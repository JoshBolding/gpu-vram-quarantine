# gpu-vram-quarantine

Salvage an NVIDIA GPU that has a few defective VRAM cells, by holding the memory
that contains them so nothing else can be given it.

Two small CUDA programs:

- **`vramcheck`** — full-VRAM integrity sweep. Finds and characterises bad cells.
- **`vrampill`** — locates the bad cells, holds only the small allocations
  containing them, and releases the rest of the card for normal use.

## The problem

A single GDDR6/GDDR6X cell that cannot hold a bit makes an otherwise healthy card
untrustworthy for anything that fills VRAM. The card games fine, benchmarks fine,
and passes most memory tests — then silently returns one wrong value to a workload
that happens to touch that address.

There is no vendor remedy on consumer cards:

- NVIDIA's **dynamic page retirement** is documented as *"No GeForce products are
  currently supported."* **Row remapping** requires ECC plus an InfoROM ECC object,
  which GeForce cards do not have.
- **GDDR6X has no on-die array ECC.** The link CRC covers transmission only. A
  stuck cell returns a wrong value with a perfectly valid checksum, so it is
  invisible to every error-reporting path on the card. `nvidia-smi -q -d ECC`,
  `-d ROW_REMAPPER` and `-d PAGE_RETIREMENT` all return `N/A`.

So the memory controller will keep handing out the bad page forever, and nothing
will ever tell you.

## The approach

VRAM is a shared physical resource. Once a process owns a page, the driver cannot
hand that page to any other process.

`vrampill` allocates the whole card in small chunks, runs a walking-bit sweep to
find which chunks contain defective cells, frees every chunk that tested clean,
and holds the rest for the life of the process. Whatever you actually want to run
then physically cannot be given the bad memory — with no cooperation needed from
that workload.

It **locates the faults on every run** rather than hardcoding an address. This
matters: the offset a given cell appears at depends on where the allocator places
the buffer. On the card this was developed against, the same physical cell showed
up anywhere between 20.67 GiB and 21.85 GiB depending on how much VRAM was free.
Hardcoding would have been wrong.

## Build

Requires the CUDA toolkit. Set `-arch` for your card (`sm_86` = Ampere/RTX 30xx,
`sm_89` = Ada/RTX 40xx, `sm_75` = Turing/RTX 20xx).

```bash
make ARCH=sm_86
```

or directly:

```bash
nvcc -O3 -arch=sm_86 -o vramcheck src/vramcheck.cu
nvcc -O3 -arch=sm_86 -o vrampill  src/vrampill.cu
```

## Usage

**Find out whether you have a problem.** Run with the card otherwise idle. Give it
real time — see the temperature note below.

```bash
./vramcheck --device 0 --seconds 1200
```

**Quarantine the bad cells.** Must run before anything else fills VRAM, since it
needs to allocate the whole card to search it.

```bash
./vrampill --device 0 --chunk-mib 8 --find-seconds 1800 --quiet-seconds 180
```

It holds until killed. Then verify what's left is clean, in another shell:

```bash
./vramcheck --device 0 --seconds 900
```

### `vrampill` options

| Flag | Default | Meaning |
|---|---|---|
| `--device N` | 0 | CUDA device index |
| `--chunk-mib N` | 8 | Quarantine granularity. Smaller wastes less, searches slower. |
| `--find-seconds N` | 1800 | Hard ceiling on the search |
| `--quiet-seconds N` | 180 | Stop once this long passes with no *new* bad chunk found |
| `--max-bad N` | 64 | Refuse to continue past this many bad chunks |
| `--allow-clean` | off | Exit 0 instead of 4 when no fault is found |
| `--ready-file P` | — | Write a marker file once quarantine is active |

### Exit codes

| Code | Meaning |
|---|---|
| 0 | Quarantine active (or no fault found, with `--allow-clean`) |
| 2 | CUDA error |
| 3 | Could not allocate enough VRAM to search meaningfully |
| 4 | **No fault found — nothing was quarantined.** Not a pass. |
| 5 | More bad chunks than `--max-bad`; card too damaged for this to make sense |

Exit 4 is deliberately a failure. If you wire this into a startup sequence, treat
it as fatal and refuse to start VRAM workloads — otherwise a run that simply
failed to provoke the fault looks identical to a healthy card.

## Read this before trusting it

**A clean search is not a clean card.** It means nothing was found in the time
allowed, in the region that was allocated. Those are different claims.

**Faults are often temperature-gated.** On the card this was built for, the defect
produced zero errors for the first 143 seconds of load, every time, and then
errored steadily once hot. A short run on a cold card will pass a broken card.
This is why `--quiet-seconds` exists and why the defaults are generous.

**It assumes allocation placement is stable while it runs.** The quarantine holds
physical pages, so it is robust once established. But the search only covers what
it managed to allocate — anything else holding VRAM hides the region it occupies.
`vrampill` aborts if it cannot allocate at least 80% of the card.

**Validated on one card.** One RTX 3090 Founders Edition, one defective cell, one
driver version (595.71.05, Linux). The multi-cell code path is exercised but has
never been tested against a card that genuinely has several bad cells, because I
do not have one. Reports welcome.

**This is not a repair.** The card is still defective. You are routing around the
damage, and you are trusting that the damage does not spread. Nobody publishes a
wear-out model for a single weak GDDR6X cell, so re-run `vramcheck` periodically
and watch whether the count stays where it was.

**Linux and CUDA only.** No Windows support, not tested on WSL.

**Don't use this for anything where a wrong answer is unacceptable** without
understanding all of the above. It makes a salvaged card *usable*; it does not
make it *trustworthy* in the way a healthy card is.

## Running it at boot

`examples/systemd/` has a working user-service setup: a guard that resolves the
target GPU **by board serial** rather than index (indexes change when cards move
slots), a readiness gate, and a drop-in that stops VRAM workloads from starting
unless the quarantine is confirmed active.

```bash
BAD_SERIAL=1234567890123 ./vrampill-guard.sh
```

If the target serial is not installed, the guard writes the marker and exits 0, so
removing the card does not block your services.

## Evidence from the card this was built for

RTX 3090 FE, one defective cell. Reported by `memtest_vulkan` as `INITIAL_READ`
errors at a repeating address suffix; independently reproduced here.

| | |
|---|---|
| Failing bit | 9 (`xor 0x00000200`), reads back 0 when a 1 is written |
| Walking-bit sweep | 685.5 TiB verified, 31,329 iterations |
| Errors found | 1,866 — **all at one address, all the same bit** |
| Distinct bad cells | 1 |
| Controls | Two healthy RTX 3090s, same host, same run: 0 errors over ~456 TiB each |
| Usable VRAM, naive cap | 20.00 GiB |
| Usable VRAM, quarantined | 22.8 GiB (holding 8 MiB) |
| Verification with quarantine active | 15,877 iterations, 342.6 TiB, peak 64 °C, **0 errors** |

The 64 °C matters: that is the temperature at which this card fails constantly
without the quarantine.

The card had previously been through a professional VRAM reball, which did not
change the fault at all — consistent with a defect inside the die rather than the
interconnect. A reball fixes joints; it cannot touch the memory array.

## Prior art, and why this didn't already exist

The idea has been raised at least twice and never built:
[BadMemory#6](https://github.com/prsyahmi/BadMemory/issues/6) (2018, no replies) and an
[NVIDIA forum thread](https://forums.developer.nvidia.com/t/gpu-memory-corrupted-because-of-broken-vram-possible-to-disable-that-memory/50088)
(2017, no replies).

Identifying *which physical chip* holds a bad cell from its address is a separate
and much harder problem. The address-to-chip mapping on modern NVIDIA GPUs is an
undocumented non-linear hash; the maintainer of `memtest_vulkan` describes relating
an address to a specific chip as
["nearly impossible"](https://github.com/GpuZelenograd/memtest_vulkan/discussions/89).
This tool does not attempt it — it makes the question unnecessary.

Credit to [memtest_vulkan](https://github.com/GpuZelenograd/memtest_vulkan) for
being the tool that identifies this class of fault in the first place, and for the
maintainer's clear public explanations of what a single-bit repeating-address
failure actually means.

## License

MIT
