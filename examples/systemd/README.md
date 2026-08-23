# Running the quarantine at boot (systemd user units)

Copy `vrampill.service` and `vrampill-ready.service` into
`~/.config/systemd/user/`, set `BAD_SERIAL` in `vrampill.service` to your card's
board serial, then:

```bash
systemctl --user daemon-reload
systemctl --user enable --now vrampill.service vrampill-ready.service
```

Make every service that allocates VRAM depend on the **gate**, via a drop-in:

```bash
mkdir -p ~/.config/systemd/user/my-gpu-workload.service.d
cp 10-vrampill.conf ~/.config/systemd/user/my-gpu-workload.service.d/
systemctl --user daemon-reload
```

## Why the two-unit split

`vrampill.service` runs for the life of the machine holding the bad memory, so it
never "finishes" and cannot itself signal readiness. `vrampill-ready.service` is a
oneshot that blocks until the marker file appears and then stays active, which is
what your workloads can actually order against.

## Fail-closed behaviour

If the quarantine cannot be established, `vrampill` exits 4, the gate fails, and
dependent services do not start. That is intentional. A search that failed to
provoke the fault looks exactly like a healthy card, and starting a VRAM workload
on that assumption is how you get silent corruption.

Verified: with the find budget forced to 5 seconds so the search could not
succeed, the dependent services stayed inactive with "A dependency job failed".

## Turn it off if the card is repaired

A **repaired** card will make `vrampill` search, find nothing, exit 4, and block
your services — indistinguishable from a broken quarantine. Disable the units:

```bash
systemctl --user disable --now vrampill.service vrampill-ready.service
rm ~/.config/systemd/user/*.service.d/10-vrampill.conf
systemctl --user daemon-reload
```

A **different** replacement card needs no action: the guard sees an unknown serial,
reports the card absent, and gets out of the way.
