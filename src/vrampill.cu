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

// Per-chunk error flags live on the device so that a whole sweep costs one
// host copy rather than one per chunk. With thousands of chunks that is the
// difference between a usable search and an unusably slow one.
__global__ void kcheck(const unsigned int* p, unsigned long long n,
                       unsigned long long base, unsigned int pat, int mode,
                       int cidx, unsigned int* chunkhits,
                       unsigned long long* chunkidx, unsigned int* chunkxor)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        unsigned int want = genval(base + i, pat, mode);
        unsigned int got  = p[i];
        if (got != want) {
            if (atomicAdd(&chunkhits[cidx], 1u) == 0u) chunkidx[cidx] = i;
            atomicOr(&chunkxor[cidx], want ^ got);
        }
    }
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
    const char* ready_file = NULL;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--device") && i + 1 < argc) dev = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--chunk-mib") && i + 1 < argc) chunk_mib = (size_t)atoll(argv[++i]);
        else if (!strcmp(argv[i], "--find-seconds") && i + 1 < argc) find_secs = atof(argv[++i]);
        else if (!strcmp(argv[i], "--quiet-seconds") && i + 1 < argc) quiet_secs = atof(argv[++i]);
        else if (!strcmp(argv[i], "--max-bad") && i + 1 < argc) max_bad = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--allow-clean")) allow_clean = 1;
        else if (!strcmp(argv[i], "--ready-file") && i + 1 < argc) ready_file = argv[++i];
    }
    if (ready_file) unlink(ready_file);

    signal(SIGINT, on_sig);
    signal(SIGTERM, on_sig);

    CK(cudaSetDevice(dev));
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
    static unsigned char      isbad[MAXCHUNK];
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

    unsigned int       *d_hits, *d_xor;
    unsigned long long *d_idx;
    CK(cudaMalloc(&d_hits, sizeof(unsigned int) * nchunk));
    CK(cudaMalloc(&d_xor,  sizeof(unsigned int) * nchunk));
    CK(cudaMalloc(&d_idx,  sizeof(unsigned long long) * nchunk));
    CK(cudaMemset(d_hits, 0, sizeof(unsigned int) * nchunk));
    CK(cudaMemset(d_xor,  0, sizeof(unsigned int) * nchunk));
    CK(cudaMemset(d_idx,  0, sizeof(unsigned long long) * nchunk));

    unsigned int* h_hits = (unsigned int*)calloc(nchunk, sizeof(unsigned int));
    unsigned int* h_xor  = (unsigned int*)calloc(nchunk, sizeof(unsigned int));
    unsigned long long* h_idx =
        (unsigned long long*)calloc(nchunk, sizeof(unsigned long long));
    if (!h_hits || !h_xor || !h_idx) { printf("ABORT: host alloc failed\n"); return 2; }

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
                                        c, d_hits, d_idx, d_xor);
        CK(cudaDeviceSynchronize());

        CK(cudaMemcpy(h_hits, d_hits, sizeof(unsigned int) * nchunk, cudaMemcpyDeviceToHost));
        for (int c = 0; c < nchunk; c++) {
            if (h_hits[c] && !isbad[c]) {
                isbad[c] = 1;
                nbad++;
                last_new = now_s();
                CK(cudaMemcpy(h_idx, d_idx, sizeof(unsigned long long) * nchunk, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(h_xor, d_xor, sizeof(unsigned int) * nchunk, cudaMemcpyDeviceToHost));
                unsigned long long off = (cbase[c] + h_idx[c]) * 4ULL;
                printf("  [%.0fs] BAD CELL #%d: chunk %d, offset 0x%llX (%.2f GiB), xor 0x%08X\n",
                       now_s() - t0, nbad, c, off, off / 1073741824.0, h_xor[c]);
                fflush(stdout);
                if (nbad > max_bad) {
                    printf("\nABORT: more than --max-bad (%d) failing chunks. This card is\n"
                           "too damaged for quarantine to be sensible.\n", max_bad);
                    return 5;
                }
            }
        }
        iter++;
        if ((iter % 500) == 0) {
            printf("  [%.0fs] %u iters, %d bad chunk(s) so far\n", now_s() - t0, iter, nbad);
            fflush(stdout);
        }
    }

    double searched = now_s() - t0;

    if (nbad == 0) {
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

    // Release everything except the chunks holding defects.
    size_t held = 0, freed = 0;
    for (int c = 0; c < nchunk; c++) {
        if (isbad[c]) { held += celems[c] * 4ULL; continue; }
        cudaFree(chunk[c]);
        freed += celems[c] * 4ULL;
    }
    cudaFree(d_hits); cudaFree(d_xor); cudaFree(d_idx);

    size_t f2 = 0, t2 = 0;
    cudaMemGetInfo(&f2, &t2);

    printf("\n=== QUARANTINE ACTIVE ===\n");
    printf("searched     : %.0fs, %u iterations\n", searched, iter);
    printf("bad chunks   : %d\n", nbad);
    printf("held         : %.1f MiB\n", held / 1048576.0);
    printf("released     : %.2f GiB\n", freed / 1073741824.0);
    printf("free now     : %.2f GiB\n", f2 / 1073741824.0);

    if (ready_file) {
        FILE* f = fopen(ready_file, "w");
        if (f) {
            fprintf(f, "status=active\ndevice=%d\nchunk_mib=%zu\nbad_chunks=%d\nheld_mib=%.1f\n",
                    dev, chunk_mib, nbad, held / 1048576.0);
            for (int c = 0; c < nchunk; c++)
                if (isbad[c])
                    fprintf(f, "bad_offset=0x%llX xor=0x%08X\n",
                            (unsigned long long)((cbase[c] + h_idx[c]) * 4ULL), h_xor[c]);
            fclose(f);
            printf("ready marker : %s\n", ready_file);
        } else {
            printf("WARNING: could not write ready marker %s\n", ready_file);
        }
    }
    printf("\nHolding until killed. Verify the rest of the card separately.\n");
    fflush(stdout);

    while (!g_stop) sleep(2);

    printf("\nsignal received, releasing quarantine.\n");
    if (ready_file) unlink(ready_file);
    for (int c = 0; c < nchunk; c++) if (isbad[c]) cudaFree(chunk[c]);
    return 0;
}
