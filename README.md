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

Requires the CUDA toolkit (SDK, not just the runtime).

```bash
make
```

That produces a fat binary with SASS for every architecture from Maxwell to
Hopper plus PTX, so it runs on newer GPUs via JIT too. You don't need to know
your compute capability. It takes about ten seconds and costs a few hundred KB.

For a faster single-arch build while developing:

```bash
make ARCH=sm_86      # or sm_89 for Ada, sm_75 for Turing
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
| `--confirm-hits N` | 2 | Repeats at the *same* offset before a chunk counts as confirmed |
| `--verify-every N` | 300 | Watchdog interval in seconds; `0` disables |
| `--verify-iters N` | 200 | Sweep iterations per watchdog pass |
| `--warn-after N` | 3600 | **Hot** seconds with no reproduction before warning |
| `--hot-c N` | 60 | Temperature at or above which absence of the fault is meaningful |
| `--allow-clean` | off | Exit 0 instead of 4 when no fault is found |
| `--ready-file P` | — | Write a marker once the quarantine is active |

Unrecognised arguments are rejected with exit 1 rather than ignored. A silently
dropped flag on a tool whose failure mode is undetected corruption is not a
tradeoff worth making.

### Watchdog

Once the quarantine is established, the holder periodically re-runs the sweep
across only the chunks it holds. If the driver ever relocated the allocation, the
fault would stop appearing in held memory and resurface in memory handed to
something else — this is the only way that failure becomes visible from inside the
process.

The subtlety is that **absence of the fault is not evidence of a problem.** These
defects are thermally gated: on the card this was built for, the cell fires roughly
1% of the times it is written at 64 °C under load, and essentially never on an idle
card. A watchdog that warned on elapsed time alone would fire on every healthy
quarantine.

So only time spent *hot* counts. Temperature comes from NVML, loaded at runtime
with `dlopen` so there is no build dependency — if it is unavailable the watchdog
still runs and still reports reproductions, it just never warns.

```
[watchdog] fault still inside quarantined memory (3 errors, 64 C)
[watchdog] no reproduction (53 C, 30s hot so far, last seen 30s ago)
```

Warnings are warnings, never failures. Even hot, absence is suggestive rather than
conclusive, and the right response is to re-run a full `vramcheck` rather than to
assume anything.

The marker file carries the freshness data for external monitoring:
`verify_passes`, `verify_reproduced`, `last_reproduced_age_s`,
`hot_secs_without_repro`, `gpu_temp_c`, `temp_source`.

### Confirmed vs unconfirmed

A chunk that errors once is a **candidate**, not a fault — a single flip could be a
transient rather than a stuck cell. A chunk becomes **confirmed** only when the same
four-byte offset fails `--confirm-hits` times. This requirement came from
[Olari-A](https://github.com/GpuZelenograd/memtest_vulkan/discussions/89), who
arrived at the same quarantine design independently and insists that discovery and
confirmation agree on allocation, offset and bit index.

Where this implementation deliberately diverges: **unconfirmed chunks are still
quarantined.** Confirmation governs what gets *reported* as established, not what
gets *held*. Retaining an extra 8 MiB costs nothing; releasing a chunk that turns
out to be genuinely bad costs silent corruption. Unconfirmed chunks are labelled as
such in the output and the marker file, and a longer run will usually settle them.
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

**The search only covers what it managed to allocate.** Anything else holding VRAM
hides the region it occupies. `vrampill` aborts if it cannot allocate at least 80%
of the card, and reports the coverage it achieved.

**It assumes the driver will not relocate the held allocation.** This is the most
serious open risk, raised by @galkinvv (memtest_vulkan's author). CUDA on Windows
can move allocated data within physical memory — VRAM-to-VRAM migration when
several applications compete for it. If that happened to the held chunk, the
process would keep its virtual range but stop owning the defective physical page,
and would go on reporting the quarantine as active while protecting nothing.
Silent failure of exactly the kind this tool exists to prevent.

Linux CUDA does not appear to do this today, which is why the approach works at
all. But it is luck rather than design.

The **watchdog** above is partial mitigation — it makes relocation *detectable*
rather than silent, though only while the card is hot enough for the fault to be
firing. The proper fix, not yet implemented, is to move the holder onto the
**CUDA VMM API** (`cuMemCreate` / `cuMemMap`), which yields an explicit physical
allocation handle rather than a relocatable mapping.

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
| Verification with quarantine active | 15,877 iterations, 342.6 TiB, peak 64 °C, **0 errors** |

How much of the card you keep:

| | Usable | vs a healthy card |
|---|---:|---:|
| Healthy RTX 3090, same host | 24,117 MiB | 100% |
| **Quarantined** | **23,853 MiB** | **98.9%** |
| Naive allocation cap below the fault | 20,480 MiB | 84.9% |

The quarantine itself is 8 MiB. The rest of the 264 MiB cost in that table is
the holder process's CUDA context. About 124 MiB of it was the default per-thread
stack reservation, which vrampill no longer asks for (its kernels use no stack),
so the holder now costs about 142 MiB on a 3090. The remaining ~134 MiB is fixed
driver overhead that any process touching CUDA pays.

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

Two people materially improved this after it was published, both in
[discussion #91](https://github.com/GpuZelenograd/memtest_vulkan/discussions/91)
and [#89](https://github.com/GpuZelenograd/memtest_vulkan/discussions/89):

- **@Olari-A** built the same mechanism independently and contributed the
  requirement that discovery and confirmation must agree on the same allocation,
  offset and bit before a quarantine is declared — now `--confirm-hits`.
- **@galkinvv** identified the allocation-relocation risk documented above, which
  is the most serious open issue with this approach, and pointed out that a fat
  binary removes the SDK-and-`-arch` burden from users. He also supplied the
  physical explanation for the delayed onset: GPUs take 2–5 minutes to reach
  thermal stabilisation under load, which is why memtest_vulkan's standard test
  runs for five.

## License

MIT
