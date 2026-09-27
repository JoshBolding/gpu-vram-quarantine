// vrampill - quarantine defective VRAM cells on a consumer NVIDIA GPU so the
// rest of the card stays usable.
//
// WHY THIS EXISTS
// ---------------
// A GDDR6/GDDR6X cell that cannot hold a bit makes an otherwise healthy card
// untrustworthy for anything that fills VRAM. There is no vendor remedy on
// GeForce: NVIDIA's dynamic page retirement and row remapping are explicitly
// unsupported on all GeForce products, and GDDR6X has no on-die array ECC (the
// link CRC covers transmission only), so a stuck cell returns a wrong value
// with a perfectly valid checksum. It is invisible to every reporting path on
// the card.
//
// This program locates the defective cells and holds the small allocations
// containing them for the life of the process. VRAM is a shared physical
// resource: once a process owns a page, the driver cannot hand it to any other
// process. Everything else on the card then becomes safe to use, without the
// cooperation of whatever workload you actually want to run.
//
// WHAT IT DOES NOT DO
// -------------------
// It does not repair anything. It does not detect faults it never provoked. A
// clean search is NOT proof the card is healthy - see --quiet-seconds and the
// exit codes below.
//
// Build: nvcc -O3 -arch=sm_86 -o vrampill vrampill.cu     (adjust -arch)
//
// Exit codes:
//   0  quarantine active (or the target GPU has no detectable fault to hold,
//      only when --allow-clean is given)
//   2  CUDA error
//   3  could not allocate enough VRAM to search meaningfully
//   4  no fault found. This is NOT a pass - it means nothing was quarantined.
//   5  found more failing chunks than --max-bad allows; refusing to continue.

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <unistd.h>
#include <csignal>
#include <dlfcn.h>
#include <cuda_runtime.h>

#define CK(x) do {                                                            \
    cudaError_t _e = (x);                                                     \
    if (_e != cudaSuccess) {                                                  \
        printf("[CUDA ERROR] %s:%d %s -> %s\n", __FILE__, __LINE__, #x,       \
               cudaGetErrorString(_e));                                       \
        fflush(stdout);                                                       \
        return 2;                                                             \
    }                                                                         \
} while (0)

#define MAXCHUNK 16384

__device__ __forceinline__
unsigned int genval(unsigned long long gidx, unsigned int pat, int mode)
{
    if (mode == 1) return (unsigned int)(gidx * 2654435761ULL) ^ pat;
    if (mode == 2) return (gidx & 1ULL) ? ~pat : pat;
    return pat;
}

__global__ void kfill(unsigned int* p, unsigned long long n,
                      unsigned long long base, unsigned int pat, int mode)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) p[i] = genval(base + i, pat, mode);
}

#define NO_CAND 0xFFFFFFFFFFFFFFFFULL

// Per-chunk state lives on the device so that a whole sweep costs one host copy
// rather than one per chunk. With thousands of chunks that is the difference
// between a usable search and an unusably slow one.
//
// The first error in a chunk nominates a candidate element index. Subsequent
// errors either land on that same index (corroboration) or somewhere else in the
// chunk (recorded separately). A chunk is only *confirmed* once the same element
// has failed --confirm-hits times, which is what distinguishes a genuinely stuck
// cell from a one-off transient. Credit to Olari-A for the requirement that
// discovery and confirmation must agree on the same offset and bit.
__global__ void kcheck(const unsigned int* p, unsigned long long n,
                       unsigned long long base, unsigned int pat, int mode,
                       int cidx, unsigned int* chunkhits,
                       unsigned long long* chunkidx, unsigned int* chunkxor,
                       unsigned int* samehits, unsigned int* mismatch)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        unsigned int want = genval(base + i, pat, mode);
        unsigned int got  = p[i];
        if (got != want) {
            atomicAdd(&chunkhits[cidx], 1u);
            unsigned long long prev =
                atomicCAS((unsigned long long*)&chunkidx[cidx], NO_CAND, i);
            if (prev == NO_CAND || prev == i) {
                atomicAdd(&samehits[cidx], 1u);
                atomicOr(&chunkxor[cidx], want ^ got);
            } else {
                atomicAdd(&mismatch[cidx], 1u);
            }
        }
    }
}

// Plain error counter for the watchdog. Deliberately separate from kcheck so a
// verification pass cannot pollute the discovery statistics.
__global__ void kverify(const unsigned int* p, unsigned long long n,
                        unsigned long long base, unsigned int pat, int mode,
                        unsigned int* nerr)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (; i < n; i += stride)
        if (p[i] != genval(base + i, pat, mode)) atomicAdd(nerr, 1u);
}

// ---------------------------------------------------------------------------
// Optional GPU temperature, via NVML loaded at runtime.
//
// Deliberately dlopen'd rather than linked: nvml.h is not reliably present with
// the CUDA toolkit, and requiring it would undo the point of shipping a binary
// that builds with a bare `make`. If the library is missing we simply lose
// temperature gating and say so.
//
// Why the watchdog needs temperature at all: these faults are thermally gated.
// On an idle card a small held region may never fail even though the defect is
// still very much there. Counting elapsed wall-clock time toward a warning
// would therefore fire on every healthy quarantine. Only time spent *hot*
// without a reproduction is evidence of anything.
// ---------------------------------------------------------------------------
typedef int (*nvml_init_t)(void);
typedef int (*nvml_bybus_t)(const char*, void**);
typedef int (*nvml_temp_t)(void*, int, unsigned int*);

static void*        g_nvml_lib  = NULL;
static nvml_temp_t  g_nvml_temp = NULL;
static void*        g_nvml_dev  = NULL;

static void nvml_try_open(const cudaDeviceProp& prop)
{
    g_nvml_lib = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
    if (!g_nvml_lib) g_nvml_lib = dlopen("libnvidia-ml.so", RTLD_LAZY);
    if (!g_nvml_lib) return;

    nvml_init_t  init  = (nvml_init_t)dlsym(g_nvml_lib, "nvmlInit_v2");
    nvml_bybus_t bybus = (nvml_bybus_t)dlsym(g_nvml_lib, "nvmlDeviceGetHandleByPciBusId_v2");
    g_nvml_temp        = (nvml_temp_t)dlsym(g_nvml_lib, "nvmlDeviceGetTemperature");
    if (!init || !bybus || !g_nvml_temp) { g_nvml_temp = NULL; return; }
    if (init() != 0) { g_nvml_temp = NULL; return; }

    // Resolve by PCI address, because CUDA and NVML device orderings differ.
    char bus[32];
    snprintf(bus, sizeof(bus), "%08X:%02X:%02X.0",
             prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
    if (bybus(bus, &g_nvml_dev) != 0) { g_nvml_temp = NULL; g_nvml_dev = NULL; }
}

// Returns degrees C, or -1 if unavailable.
static int gpu_temp_c(void)
{
    if (!g_nvml_temp || !g_nvml_dev) return -1;
    unsigned int t = 0;
    if (g_nvml_temp(g_nvml_dev, 0 /* NVML_TEMPERATURE_GPU */, &t) != 0) return -1;
    return (int)t;
}

static volatile sig_atomic_t g_stop = 0;
static void on_sig(int) { g_stop = 1; }

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

int main(int argc, char** argv)
{
    int    dev         = 0;
    size_t chunk_mib   = 8;
    double find_secs   = 1800.0;   // hard ceiling on the search
    double quiet_secs  = 180.0;    // stop once this long passes with nothing new
    int    max_bad     = 64;
    int    allow_clean = 0;
    int    confirm_hits = 2;   // repeats at the same offset before "confirmed"
    double verify_every = 300.0;   // seconds between watchdog passes; 0 disables
    int    verify_iters = 200;     // sweep iterations per watchdog pass
    double warn_after   = 3600.0;  // HOT seconds without reproduction before warning
    int    hot_c        = 60;      // at or above this, absence of the fault is meaningful
    const char* ready_file = NULL;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--device") && i + 1 < argc) dev = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--chunk-mib") && i + 1 < argc) chunk_mib = (size_t)atoll(argv[++i]);
        else if (!strcmp(argv[i], "--find-seconds") && i + 1 < argc) find_secs = atof(argv[++i]);
        else if (!strcmp(argv[i], "--quiet-seconds") && i + 1 < argc) quiet_secs = atof(argv[++i]);
        else if (!strcmp(argv[i], "--max-bad") && i + 1 < argc) max_bad = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--confirm-hits") && i + 1 < argc) confirm_hits = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--verify-every") && i + 1 < argc) verify_every = atof(argv[++i]);
        else if (!strcmp(argv[i], "--verify-iters") && i + 1 < argc) verify_iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warn-after") && i + 1 < argc) warn_after = atof(argv[++i]);
        else if (!strcmp(argv[i], "--hot-c") && i + 1 < argc) hot_c = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--allow-clean")) allow_clean = 1;
        else if (!strcmp(argv[i], "--ready-file") && i + 1 < argc) ready_file = argv[++i];
        else {
            // Reject rather than ignore. A silently dropped flag on a tool whose
            // failure mode is undetected memory corruption is not acceptable.
            if (strcmp(argv[i], "--help") && strcmp(argv[i], "-h"))
                printf("unrecognised or incomplete argument: %s\n\n", argv[i]);
            printf(
                "vrampill - quarantine defective VRAM cells so the rest of the card stays usable\n\n"
                "  --device N          CUDA device index (default 0)\n"
                "  --chunk-mib N       quarantine granularity in MiB (default 8)\n"
                "  --find-seconds N    hard ceiling on the search (default 1800)\n"
                "  --quiet-seconds N   stop after this long with nothing new (default 180)\n"
                "  --confirm-hits N    repeats at one offset before confirming (default 2)\n"
                "  --max-bad N         refuse to continue past this many bad chunks (default 64)\n"
                "  --verify-every N    watchdog interval in seconds, 0 disables (default 300)\n"
                "  --verify-iters N    sweep iterations per watchdog pass (default 200)\n"
                "  --warn-after N      HOT seconds without reproduction before warning (default 3600)\n"
                "  --hot-c N           temperature at or above which absence is meaningful (default 60)\n"
                "  --allow-clean       exit 0 instead of 4 when no fault is found\n"
                "  --ready-file PATH   write a marker once the quarantine is active\n\n"
                "Exit codes: 0 active, 2 CUDA error, 3 insufficient VRAM,\n"
                "            4 no fault found (NOT a pass), 5 too many bad chunks\n");
            return (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) ? 0 : 1;
        }
    }
    if (ready_file) unlink(ready_file);

    signal(SIGINT, on_sig);
    signal(SIGTERM, on_sig);

    CK(cudaSetDevice(dev));
    // The context reserves a per-thread stack for every thread the card can
    // run at once: 1 KiB by default, about 124 MiB on a 3090. None of the
    // kernels here use any stack, so that reservation is pure waste for a
    // process that exists to sit on a few MiB. Credit to xiconfjs (issue #3).
    CK(cudaDeviceSetLimit(cudaLimitStackSize, 0));
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, dev));

    size_t freeB = 0, totalB = 0;
    CK(cudaMemGetInfo(&freeB, &totalB));

    printf("=== vrampill ===\n");
    printf("device       : %d (%s)\n", dev, prop.name);
    printf("total VRAM   : %.2f GiB\n", totalB / 1073741824.0);
    printf("free VRAM    : %.2f GiB\n", freeB  / 1073741824.0);
    printf("chunk size   : %zu MiB\n", chunk_mib);
    printf("search       : up to %.0fs, stop after %.0fs with nothing new\n",
           find_secs, quiet_secs);
    fflush(stdout);

    static unsigned int*      chunk[MAXCHUNK];
    static unsigned long long celems[MAXCHUNK];
    static unsigned long long cbase[MAXCHUNK];
    static unsigned char      isbad[MAXCHUNK];   // confirmed: repeated at one offset
    static unsigned char      seen[MAXCHUNK];    // errored at least once, maybe transient
    int    nchunk = 0;
    size_t got_total = 0;

    size_t csize  = chunk_mib * 1048576ULL;
    size_t target = (freeB > 640ULL * 1048576ULL) ? freeB - 640ULL * 1048576ULL : 0;

    unsigned long long base = 0;
    while (nchunk < MAXCHUNK && got_total + csize <= target) {
        void* p = NULL;
        if (cudaMalloc(&p, csize) != cudaSuccess) { cudaGetLastError(); break; }
        chunk[nchunk]  = (unsigned int*)p;
        celems[nchunk] = csize / 4ULL;
        cbase[nchunk]  = base;
        isbad[nchunk]  = 0;
        seen[nchunk]   = 0;
        base      += celems[nchunk];
        got_total += csize;
        nchunk++;
    }
    printf("allocated    : %.2f GiB in %d chunks\n", got_total / 1073741824.0, nchunk);

    // A search that covers only part of the card can only ever quarantine
    // faults inside the part it covered.
    double cover = (double)got_total / (double)totalB;
    printf("coverage     : %.1f%% of total VRAM\n", cover * 100.0);
    if (cover < 0.80) {
        printf("\nABORT: only %.1f%% of VRAM could be allocated. Free the card first;\n"
               "anything holding VRAM hides the region it occupies from this search.\n",
               cover * 100.0);
        return 3;
    }
    printf("\n");
    fflush(stdout);

    unsigned int       *d_hits, *d_xor, *d_same, *d_mism;
    unsigned long long *d_idx;
    CK(cudaMalloc(&d_hits, sizeof(unsigned int) * nchunk));
    CK(cudaMalloc(&d_xor,  sizeof(unsigned int) * nchunk));
    CK(cudaMalloc(&d_same, sizeof(unsigned int) * nchunk));
    CK(cudaMalloc(&d_mism, sizeof(unsigned int) * nchunk));
    CK(cudaMalloc(&d_idx,  sizeof(unsigned long long) * nchunk));
    CK(cudaMemset(d_hits, 0, sizeof(unsigned int) * nchunk));
    CK(cudaMemset(d_xor,  0, sizeof(unsigned int) * nchunk));
    CK(cudaMemset(d_same, 0, sizeof(unsigned int) * nchunk));
    CK(cudaMemset(d_mism, 0, sizeof(unsigned int) * nchunk));
    // 0xFF... = NO_CAND, i.e. no candidate offset nominated yet.
    CK(cudaMemset(d_idx, 0xFF, sizeof(unsigned long long) * nchunk));

    unsigned int* h_hits = (unsigned int*)calloc(nchunk, sizeof(unsigned int));
    unsigned int* h_xor  = (unsigned int*)calloc(nchunk, sizeof(unsigned int));
    unsigned int* h_same = (unsigned int*)calloc(nchunk, sizeof(unsigned int));
    unsigned int* h_mism = (unsigned int*)calloc(nchunk, sizeof(unsigned int));
    unsigned long long* h_idx =
        (unsigned long long*)calloc(nchunk, sizeof(unsigned long long));
    if (!h_hits || !h_xor || !h_same || !h_mism || !h_idx) {
        printf("ABORT: host alloc failed\n"); return 2;
    }

    // Walking one and walking zero drive every bit position both high and low
    // at every address. A cell that only fails in one direction is invisible to
    // partial bit coverage.
    unsigned int pats[8 + 64];
    int npat = 0;
    const unsigned int base_pats[] = {
        0x00000000u, 0xFFFFFFFFu, 0xAAAAAAAAu, 0x55555555u,
        0xA5A5A5A5u, 0x5A5A5A5Au, 0xDEADBEEFu, 0x00FF00FFu
    };
    for (int i = 0; i < 8; i++) pats[npat++] = base_pats[i];
    for (int b = 0; b < 32; b++) pats[npat++] = (1u << b);
    for (int b = 0; b < 32; b++) pats[npat++] = ~(1u << b);

    int threads = 256;
    int blocks  = prop.multiProcessorCount * 16;

    printf("searching (faults are often temperature-gated; the card must warm up)...\n");
    fflush(stdout);

    double t0 = now_s();
    double last_new = t0;
    int    nbad = 0;
    unsigned int iter = 0;

    // Keep searching after the first hit. Stopping early was the single most
    // dangerous behaviour this tool could have: it would quarantine one cell,
    // report success, and leave any others live with no warning.
    while (!g_stop) {
        double el = now_s() - t0;
        if (el >= find_secs) break;
        if (nbad > 0 && (now_s() - last_new) >= quiet_secs) break;

        unsigned int pat  = pats[iter % npat];
        int          mode = (iter / npat) % 3;

        for (int c = 0; c < nchunk; c++)
            kfill<<<blocks, threads>>>(chunk[c], celems[c], cbase[c], pat, mode);
        CK(cudaDeviceSynchronize());

        for (int c = 0; c < nchunk; c++)
            kcheck<<<blocks, threads>>>(chunk[c], celems[c], cbase[c], pat, mode,
                                        c, d_hits, d_idx, d_xor, d_same, d_mism);
        CK(cudaDeviceSynchronize());

        CK(cudaMemcpy(h_hits, d_hits, sizeof(unsigned int) * nchunk, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h_same, d_same, sizeof(unsigned int) * nchunk, cudaMemcpyDeviceToHost));
        for (int c = 0; c < nchunk; c++) {
            if (!h_hits[c]) continue;

            // Announce a chunk the first time it errors at all, but do not treat
            // it as confirmed until the same offset has failed repeatedly. One
            // hit could be a transient rather than a stuck cell.
            if (!seen[c]) {
                seen[c] = 1;
                CK(cudaMemcpy(h_idx, d_idx, sizeof(unsigned long long) * nchunk, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(h_xor, d_xor, sizeof(unsigned int) * nchunk, cudaMemcpyDeviceToHost));
                unsigned long long off = (cbase[c] + h_idx[c]) * 4ULL;
                printf("  [%.0fs] candidate: chunk %d, offset 0x%llX (%.2f GiB), xor 0x%08X"
                       " — needs %d hits to confirm\n",
                       now_s() - t0, c, off, off / 1073741824.0, h_xor[c], confirm_hits);
                fflush(stdout);
            }

            if (!isbad[c] && (int)h_same[c] >= confirm_hits) {
                isbad[c] = 1;
                nbad++;
                last_new = now_s();
                CK(cudaMemcpy(h_idx, d_idx, sizeof(unsigned long long) * nchunk, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(h_xor, d_xor, sizeof(unsigned int) * nchunk, cudaMemcpyDeviceToHost));
                unsigned long long off = (cbase[c] + h_idx[c]) * 4ULL;
                printf("  [%.0fs] CONFIRMED #%d: chunk %d, offset 0x%llX (%.2f GiB),"
                       " xor 0x%08X, %u hits at that offset\n",
                       now_s() - t0, nbad, c, off, off / 1073741824.0, h_xor[c], h_same[c]);
                fflush(stdout);
                if (nbad > max_bad) {
                    printf("\nABORT: more than --max-bad (%d) confirmed failing chunks.\n"
                           "This card is too damaged for quarantine to be sensible.\n", max_bad);
                    return 5;
                }
            }
        }
        iter++;
        if ((iter % 500) == 0) {
            printf("  [%.0fs] %u iters, %d confirmed\n", now_s() - t0, iter, nbad);
            fflush(stdout);
        }
    }

    double searched = now_s() - t0;

    CK(cudaMemcpy(h_idx,  d_idx,  sizeof(unsigned long long) * nchunk, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_xor,  d_xor,  sizeof(unsigned int) * nchunk, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_same, d_same, sizeof(unsigned int) * nchunk, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_mism, d_mism, sizeof(unsigned int) * nchunk, cudaMemcpyDeviceToHost));

    int nseen = 0, nunconf = 0;
    for (int c = 0; c < nchunk; c++) {
        if (!seen[c]) continue;
        nseen++;
        if (!isbad[c]) nunconf++;
    }

    if (nseen == 0) {
        printf("\nNo fault found in %.0fs / %u iterations.\n", searched, iter);
        printf("\nThis is NOT a clean bill of health. It means nothing was quarantined.\n"
               "A temperature-gated fault can need several minutes of load to appear,\n"
               "and a fault outside the allocated region cannot be seen at all.\n");
        if (!allow_clean) {
            printf("Exiting 4 so that callers fail closed. Pass --allow-clean to treat\n"
                   "this as success on a card you believe is healthy.\n");
            return 4;
        }
        printf("--allow-clean given; exiting 0 with no quarantine held.\n");
        if (ready_file) {
            FILE* f = fopen(ready_file, "w");
            if (f) { fprintf(f, "status=no-fault-found\ndevice=%d\n", dev); fclose(f); }
        }
        return 0;
    }

    // Quarantine anything that ever errored, confirmed or not. Holding an extra
    // 8 MiB costs nothing; releasing a chunk that turns out to be genuinely bad
    // costs silent corruption. Confirmation governs what gets *reported* as
    // established, not what gets held.
    size_t held = 0, freed = 0;
    for (int c = 0; c < nchunk; c++) {
        if (seen[c]) { held += celems[c] * 4ULL; continue; }
        cudaFree(chunk[c]);
        freed += celems[c] * 4ULL;
    }
    cudaFree(d_hits); cudaFree(d_xor); cudaFree(d_idx);
    cudaFree(d_same); cudaFree(d_mism);

    size_t f2 = 0, t2 = 0;
    cudaMemGetInfo(&f2, &t2);

    printf("\n=== QUARANTINE ACTIVE ===\n");
    printf("searched     : %.0fs, %u iterations\n", searched, iter);
    printf("confirmed    : %d chunk(s) (>= %d hits at one offset)\n", nbad, confirm_hits);
    if (nunconf)
        printf("unconfirmed  : %d chunk(s) errored but never repeated — held anyway\n", nunconf);
    printf("held         : %.1f MiB across %d chunk(s)\n", held / 1048576.0, nseen);
    printf("released     : %.2f GiB\n", freed / 1073741824.0);
    printf("free now     : %.2f GiB\n", f2 / 1073741824.0);

    printf("\n%-8s %-20s %-12s %-8s %-9s %s\n",
           "chunk", "offset", "xor", "hits", "elsewhere", "state");
    for (int c = 0; c < nchunk; c++) {
        if (!seen[c]) continue;
        printf("%-8d 0x%016llX  0x%08X   %-8u %-9u %s\n",
               c, (unsigned long long)((cbase[c] + h_idx[c]) * 4ULL), h_xor[c],
               h_same[c], h_mism[c], isbad[c] ? "CONFIRMED" : "unconfirmed");
    }
    if (nunconf)
        printf("\nUnconfirmed chunks errored once and never again at the same offset.\n"
               "That may be a transient rather than a stuck cell. They are quarantined\n"
               "regardless, but a longer run would settle it.\n");

    // ---- watchdog state -------------------------------------------------
    // The quarantine was just established from a live fault, so "last
    // reproduced" starts now.
    time_t last_repro = time(NULL);
    unsigned long long verify_passes = 0, verify_reproduced = 0;
    double hot_secs_no_repro = 0.0;   // only time spent hot counts toward a warning

    nvml_try_open(prop);
    int have_temp = (gpu_temp_c() >= 0);

    auto write_marker = [&]() {
        if (!ready_file) return;
        FILE* f = fopen(ready_file, "w");
        if (!f) { printf("WARNING: could not write ready marker %s\n", ready_file); return; }
        fprintf(f, "status=active\ndevice=%d\nchunk_mib=%zu\n"
                   "confirmed_chunks=%d\nunconfirmed_chunks=%d\nheld_chunks=%d\n"
                   "held_mib=%.1f\nconfirm_hits=%d\n",
                dev, chunk_mib, nbad, nunconf, nseen,
                held / 1048576.0, confirm_hits);
        fprintf(f, "verify_passes=%llu\nverify_reproduced=%llu\n"
                   "last_reproduced_unix=%lld\nlast_reproduced_age_s=%lld\n"
                   "hot_secs_without_repro=%.0f\nhot_threshold_c=%d\n"
                   "gpu_temp_c=%d\ntemp_source=%s\n",
                verify_passes, verify_reproduced,
                (long long)last_repro, (long long)(time(NULL) - last_repro),
                hot_secs_no_repro, hot_c, gpu_temp_c(),
                have_temp ? "nvml" : "unavailable");
        for (int c = 0; c < nchunk; c++)
            if (seen[c])
                fprintf(f, "bad_offset=0x%llX xor=0x%08X hits=%u state=%s\n",
                        (unsigned long long)((cbase[c] + h_idx[c]) * 4ULL),
                        h_xor[c], h_same[c], isbad[c] ? "confirmed" : "unconfirmed");
        fclose(f);
    };

    write_marker();
    if (ready_file) printf("ready marker : %s\n", ready_file);

    if (verify_every > 0) {
        printf("watchdog     : re-checking held memory every %.0fs\n", verify_every);
        if (have_temp)
            printf("               warns after %.0fs at >=%d C with no reproduction\n",
                   warn_after, hot_c);
        else
            printf("               NVML unavailable, no temperature gating — absence of\n"
                   "               the fault on an idle card is normal and not reported\n");
    }
    printf("\nHolding until killed. Verify the rest of the card separately.\n");
    fflush(stdout);

    unsigned int* d_verr = NULL;
    CK(cudaMalloc(&d_verr, sizeof(unsigned int)));

    double next_verify = now_s() + verify_every;

    while (!g_stop) {
        sleep(2);
        if (verify_every <= 0 || now_s() < next_verify) continue;
        next_verify = now_s() + verify_every;

        // Re-run the sweep across only the held chunks. If the driver ever
        // relocated this allocation, we would still own the virtual range but
        // no longer the defective physical page — and the fault would stop
        // appearing here while reappearing in memory handed to someone else.
        unsigned int total = 0;
        for (int it = 0; it < verify_iters && !g_stop; it++) {
            unsigned int pat  = pats[it % npat];
            int          mode = (it / npat) % 3;
            for (int c = 0; c < nchunk; c++)
                if (seen[c]) kfill<<<blocks, threads>>>(chunk[c], celems[c], cbase[c], pat, mode);
            CK(cudaDeviceSynchronize());
            CK(cudaMemset(d_verr, 0, sizeof(unsigned int)));
            for (int c = 0; c < nchunk; c++)
                if (seen[c]) kverify<<<blocks, threads>>>(chunk[c], celems[c], cbase[c],
                                                          pat, mode, d_verr);
            CK(cudaDeviceSynchronize());
            unsigned int e = 0;
            CK(cudaMemcpy(&e, d_verr, sizeof(e), cudaMemcpyDeviceToHost));
            total += e;
        }
        verify_passes++;

        int temp = gpu_temp_c();

        if (total) {
            verify_reproduced++;
            last_repro = time(NULL);
            hot_secs_no_repro = 0.0;
            printf("[watchdog] fault still inside quarantined memory (%u errors, %d C)\n",
                   total, temp);
        } else {
            // Only accumulate time the card was actually hot. A cool card not
            // reproducing a thermally-gated fault is expected, and counting that
            // toward a warning would fire on every healthy quarantine.
            if (have_temp && temp >= hot_c) hot_secs_no_repro += verify_every;

            long long age = (long long)(time(NULL) - last_repro);
            if (have_temp && hot_secs_no_repro > warn_after) {
                // Still a warning rather than a failure: even hot, absence is
                // suggestive rather than conclusive.
                printf("[watchdog] WARNING: %.0fs at >=%d C with no reproduction in held\n"
                       "           memory (currently %d C, last seen %llds ago). Either the\n"
                       "           cell has changed behaviour or this allocation no longer\n"
                       "           covers the defect. Re-run a full vramcheck to find out.\n",
                       hot_secs_no_repro, hot_c, temp, age);
            } else if (have_temp) {
                printf("[watchdog] no reproduction (%d C, %.0fs hot so far, last seen %llds ago)\n",
                       temp, hot_secs_no_repro, age);
            } else {
                printf("[watchdog] no reproduction (last seen %llds ago, no temperature data)\n",
                       age);
            }
        }
        write_marker();
        fflush(stdout);
    }

    printf("\nsignal received, releasing quarantine.\n");
    if (ready_file) unlink(ready_file);
    cudaFree(d_verr);
    for (int c = 0; c < nchunk; c++) if (seen[c]) cudaFree(chunk[c]);
    return 0;
}
