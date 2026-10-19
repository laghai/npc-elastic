"""python -m npc_elastic SPEC.json OUT_DIR [--workers N]  |  python -m npc_elastic --schema DIR"""

from __future__ import annotations

import sys
import time


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["--schema"]:
        from .schema import write_all

        for p in write_all(argv[1] if len(argv) > 1 else "elastic/schema"):
            print(p)
        return 0
    workers = 1
    if "--workers" in argv:
        j = argv.index("--workers")
        workers = int(argv[j + 1])
        argv = argv[:j] + argv[j + 2 :]
    if len(argv) != 2:
        print(__doc__)
        return 2
    from . import run

    t0 = time.time()

    def progress(sym, i, n, md):
        print(
            f"[{time.time() - t0:7.1f}s] {sym} k {i + 1}/{n}: "
            f"{md.freqs_hz[0] / 1e9:.3f}..{md.freqs_hz[-1] / 1e9:.3f} GHz",
            flush=True,
        )

    res = run(argv[0], out_dir=argv[1], progress=progress, workers=workers)
    print(
        f"done: {sum(len(r.k) for r in res.runs)} k-points, {res.mesh.n_elements} elements, "
        f"{time.time() - t0:.0f} s -> {argv[1]}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
