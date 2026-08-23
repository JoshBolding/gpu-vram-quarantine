// vramcheck.cu - full-VRAM write/verify integrity sweep for a single CUDA device.
//
// Companion to vrampill. Finds and characterises defective VRAM cells, and is
// also how you verify that a quarantine actually worked.
//
// Two properties of real single-cell faults drive the design:
//
//   1. COVERAGE. A defect lives at a fixed physical location, which may sit
//      anywhere in the card. A sweep that allocates only part of VRAM can only
//      find faults in the part it allocated. This tool takes everything it can
//      and reports what fraction that was.
//
//   2. PROVOCATION. Such faults are usually intermittent and frequently
//      temperature-gated. On the card this was written for, zero errors appeared
//      during the first 143 seconds of load, every time, and then the fault
//      errored steadily once hot. A single sweep - or a short one on a cold card -
//      proves nothing. Run many iterations, and let the card heat up.
//
// Patterns include a walking-one and walking-zero sweep so that every bit
// position is driven both high and low at every address; a cell that only fails
// in one direction is invisible to partial bit coverage.
//
// Build: nvcc -O3 -arch=sm_86 -o vramcheck vramcheck.cu   (set -arch for your GPU)

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
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

#define MAXCHUNK 128
#define MAXREC   8192

struct ErrRec {
    unsigned long long off;   // byte offset in the global linear space
    unsigned int expected;
    unsigned int got;
    unsigned int iter;
};

// Pattern generator. mode 0: constant. mode 1: address-derived (catches
// aliasing / wrong-row returns). mode 2: parity-alternating.
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

__global__ void kcheck(const unsigned int* p, unsigned long long n,
                       unsigned long long base, unsigned int pat, int mode,
                       unsigned long long* nerr, ErrRec* recs, unsigned int iter)
{
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        unsigned int want = genval(base + i, pat, mode);
        unsigned int got  = p[i];
        if (got != want) {
            unsigned long long k = atomicAdd(nerr, 1ULL);
            if (k < MAXREC) {
                recs[k].off      = (base + i) * 4ULL;
                recs[k].expected = want;
                recs[k].got      = got;
                recs[k].iter     = iter;
            }
        }
    }
}

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

int main(int argc, char** argv)
{
    int    dev      = 0;
    double seconds  = 300.0;
    size_t reserve  = 640;   // MiB left unallocated for context/driver
    double max_gib  = 0.0;   // 0 = uncapped; else cap allocation to this many GiB

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--device")  && i + 1 < argc) dev     = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seconds") && i + 1 < argc) seconds = atof(argv[++i]);
        else if (!strcmp(argv[i], "--reserve") && i + 1 < argc) reserve = (size_t)atoll(argv[++i]);
        else if (!strcmp(argv[i], "--max-gib") && i + 1 < argc) max_gib = atof(argv[++i]);
    }

    CK(cudaSetDevice(dev));

    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, dev));

    char serial[64] = "unknown";
    // nvidia-smi is the only reliable source of board serial; caller logs it too.

    size_t freeB = 0, totalB = 0;
    CK(cudaMemGetInfo(&freeB, &totalB));

    printf("=== vramcheck ===\n");
    printf("device        : %d (%s)\n", dev, prop.name);
    printf("total VRAM    : %.2f GiB\n", totalB / 1073741824.0);
    printf("free VRAM     : %.2f GiB\n", freeB  / 1073741824.0);
    printf("reserve       : %zu MiB\n", reserve);
    printf("duration      : %.0f s\n", seconds);
    fflush(stdout);

    // --- allocate as much as we can, in chunks ---
    unsigned int*      chunk[MAXCHUNK];
    unsigned long long celems[MAXCHUNK];
    unsigned long long cbase[MAXCHUNK];
    int    nchunk = 0;
    size_t got_total = 0;

    size_t target = (freeB > reserve * 1048576ULL) ? freeB - reserve * 1048576ULL : 0;
    if (max_gib > 0.0) {
        size_t cap = (size_t)(max_gib * 1073741824.0);
        if (cap < target) target = cap;
        printf("cap           : %.2f GiB (--max-gib)\n", max_gib);
    }
    size_t csize  = 512ULL * 1048576ULL;   // 512 MiB chunks

    unsigned long long base = 0;
    while (nchunk < MAXCHUNK && got_total + csize <= target) {
        void* p = NULL;
        if (cudaMalloc(&p, csize) != cudaSuccess) { cudaGetLastError(); break; }
        chunk[nchunk]  = (unsigned int*)p;
        celems[nchunk] = csize / 4ULL;
        cbase[nchunk]  = base;
        base      += celems[nchunk];
        got_total += csize;
        nchunk++;
    }
    // top up with smaller chunks
    for (size_t s = csize / 2; s >= 32ULL * 1048576ULL && nchunk < MAXCHUNK; s /= 2) {
        while (nchunk < MAXCHUNK && got_total + s <= target) {
            void* p = NULL;
            if (cudaMalloc(&p, s) != cudaSuccess) { cudaGetLastError(); break; }
            chunk[nchunk]  = (unsigned int*)p;
            celems[nchunk] = s / 4ULL;
            cbase[nchunk]  = base;
            base      += celems[nchunk];
            got_total += s;
            nchunk++;
        }
    }

    printf("allocated     : %.2f GiB in %d chunks\n", got_total / 1073741824.0, nchunk);

    const double GIB = 1073741824.0;
    double cover = (double)got_total / (double)totalB;
    if (max_gib > 0.0) {
        printf("coverage      : CAPPED RUN - deliberately testing a %.2f GiB subset\n"
               "                (%.1f%% of the card). A clean result means only that\n"
               "                THIS allocation held no bad cell. It says nothing\n"
               "                about the rest of the card.\n",
               got_total / GIB, cover * 100.0);
    } else if (cover < 0.90) {
        printf("\n*** WARNING: allocated %.2f GiB, only %.1f%% of this card's %.2f GiB.\n"
               "*** Faults outside that region cannot be detected. Free the card of\n"
               "*** other work and re-run before treating a clean result as meaningful.\n",
               got_total / GIB, cover * 100.0, totalB / GIB);
    } else {
        printf("coverage      : %.1f%% of total VRAM\n", cover * 100.0);
    }
    printf("\n");
    fflush(stdout);

    unsigned long long* d_nerr = NULL;
    ErrRec*             d_recs = NULL;
    CK(cudaMalloc(&d_nerr, sizeof(unsigned long long)));
    CK(cudaMalloc(&d_recs, sizeof(ErrRec) * MAXREC));
    CK(cudaMemset(d_nerr, 0, sizeof(unsigned long long)));
    CK(cudaMemset(d_recs, 0, sizeof(ErrRec) * MAXREC));

    // Base patterns, then a walking-one and walking-zero sweep so that every one
    // of the 32 bit positions is driven both high and low at every address. A
    // stuck or weak cell only misbehaves when its bit is driven to the value it
    // cannot hold, so partial bit coverage can hide a fault entirely.
    unsigned int pats[8 + 64];
    int npat = 0;
    const unsigned int base_pats[] = {
        0x00000000u, 0xFFFFFFFFu, 0xAAAAAAAAu, 0x55555555u,
        0xA5A5A5A5u, 0x5A5A5A5Au, 0xDEADBEEFu, 0x00FF00FFu
    };
    for (int i = 0; i < 8; i++) pats[npat++] = base_pats[i];
    for (int b = 0; b < 32; b++) pats[npat++] = (1u << b);      // walking one
    for (int b = 0; b < 32; b++) pats[npat++] = ~(1u << b);     // walking zero
    const int modes[] = {0, 1, 2};
    const int nmode = 3;

    int  threads = 256;
    int  blocks  = prop.multiProcessorCount * 16;

    double t0 = now_s();
    unsigned long long total_err = 0, last_reported = 0;
    unsigned int iter = 0;
    double bytes_moved = 0.0;

    while (now_s() - t0 < seconds) {
        unsigned int pat  = pats[iter % npat];
        int          mode = modes[(iter / npat) % nmode];

        // Fill every chunk first, then verify every chunk. The gap between the
        // write and the first read back is what reproduces an INITIAL_READ-class
        // fault rather than an immediate write-read hazard.
        for (int c = 0; c < nchunk; c++)
            kfill<<<blocks, threads>>>(chunk[c], celems[c], cbase[c], pat, mode);
        CK(cudaDeviceSynchronize());

        for (int c = 0; c < nchunk; c++)
            kcheck<<<blocks, threads>>>(chunk[c], celems[c], cbase[c], pat, mode,
                                        d_nerr, d_recs, iter);
        CK(cudaDeviceSynchronize());

        bytes_moved += (double)got_total * 2.0;
        iter++;

        CK(cudaMemcpy(&total_err, d_nerr, sizeof(total_err), cudaMemcpyDeviceToHost));
        if (total_err != last_reported) {
            printf("[iter %6u] *** ERRORS: %llu total (pattern 0x%08X mode %d)\n",
                   iter, total_err, pat, mode);
            fflush(stdout);
            last_reported = total_err;
        }
        if ((iter % 100) == 0) {
            double el = now_s() - t0;
            printf("[iter %6u] %6.1fs elapsed, %llu errors, %.0f GB/s effective\n",
                   iter, el, total_err, bytes_moved / el / 1e9);
            fflush(stdout);
        }
    }

    double elapsed = now_s() - t0;
    CK(cudaMemcpy(&total_err, d_nerr, sizeof(total_err), cudaMemcpyDeviceToHost));

    ErrRec recs[MAXREC];
    CK(cudaMemcpy(recs, d_recs, sizeof(recs), cudaMemcpyDeviceToHost));

    printf("\n=== RESULT device %d (%s) ===\n", dev, prop.name);
    printf("iterations    : %u\n", iter);
    printf("elapsed       : %.1f s\n", elapsed);
    printf("coverage      : %.2f GiB per iteration\n", got_total / GIB);
    printf("data verified : %.1f TiB\n", bytes_moved / 2.0 / 1099511627776.0);
    printf("effective BW  : %.0f GB/s\n", bytes_moved / elapsed / 1e9);
    printf("ERRORS        : %llu\n", total_err);

    if (total_err) {
        unsigned long long shown = total_err < MAXREC ? total_err : MAXREC;

        // Distinct failing addresses are the whole point of this run: with more
        // than one, the spacing between them exposes the memory interleave and
        // can point at a channel. With only one, there is no structure to read.
        unsigned long long uoff[MAXREC];
        unsigned long long ucnt[MAXREC];
        unsigned int       ubits[MAXREC];
        int nu = 0;
        for (unsigned long long i = 0; i < shown; i++) {
            int f = -1;
            for (int j = 0; j < nu; j++) if (uoff[j] == recs[i].off) { f = j; break; }
            if (f < 0 && nu < MAXREC) {
                uoff[nu] = recs[i].off; ucnt[nu] = 0; ubits[nu] = 0; f = nu++;
            }
            if (f >= 0) { ucnt[f]++; ubits[f] |= (recs[i].expected ^ recs[i].got); }
        }

        printf("\nDISTINCT failing addresses: %d\n", nu);
        printf("  %-20s %-10s %-12s %s\n", "byte_offset", "hits", "bits_seen", "bit_positions");
        for (int j = 0; j < nu; j++) {
            printf("  0x%016llX %-10llu 0x%08X   ", uoff[j], ucnt[j], ubits[j]);
            for (int b = 0; b < 32; b++) if (ubits[j] & (1u << b)) printf("%d ", b);
            printf("\n");
        }

        if (nu > 1) {
            printf("\naddress deltas between distinct failures (interleave clues):\n");
            for (int j = 1; j < nu; j++) {
                long long d = (long long)uoff[j] - (long long)uoff[0];
                printf("  [%d]-[0] = %lld bytes (0x%llX)\n", j, d, d < 0 ? -d : d);
            }
        } else {
            printf("\nOnly one distinct address failed. No spacing information is\n"
                   "available, so the memory interleave cannot be inferred and the\n"
                   "failing chip cannot be narrowed down from this data.\n");
        }

        printf("\nVERDICT: FAIL - VRAM integrity errors detected.\n");
    } else {
        printf("\nVERDICT: PASS - no VRAM integrity errors in %u iterations.\n", iter);
    }
    fflush(stdout);

    for (int c = 0; c < nchunk; c++) cudaFree(chunk[c]);
    cudaFree(d_nerr);
    cudaFree(d_recs);
    return total_err ? 1 : 0;
}
