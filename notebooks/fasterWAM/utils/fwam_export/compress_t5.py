#!/usr/bin/env python
"""umT5-xxl text encoder: fp16 IR -> INT8_SYM weight-only IR (NNCF, data-free, per-channel).

The second storage arm of the text encoder.  Input is the IR ``export_t5.py`` wrote (fp16 weights,
untouched here); output is ``t5_int8.xml/.bin`` next to it plus ``t5_int8_manifest.json``.

Recipe, measured 11 Nov on an integrated GPU (notebook page `fasterwam-t5-int8-igpu`):

  * ``nncf.compress_weights(mode=INT8_SYM)`` with the defaults: every weight (24 x 7 FCs + the
    token-embedding Gather = 169), per-channel, ratio 1.0, no calibration data.  ASYM and "keep the
    embedding fp16" were measured too and lose on the action chunk.
  * saved with ``compress_to_fp16=False`` (the int8 weights stay int8; scales stay as NNCF made them),
    5.69 GB against 11.4 GB.
  * runs in **bf16 execution on a GPU** like the fp16 arm (f16 execution NaNs).  Context cos_min
    0.96375 / mean 0.99583 against the shipped bank (fp16 weights: 0.98975 / 0.99918), about 2x faster
    per prompt; the action chunk moves to cos_min 0.999857 against the shipped-context chunk.
  * NNCF peaks at about 17 GB of host memory for about 17 s, which is why this is a subprocess.

Usage:
    python compress_t5.py --src <notebook>/model/ir/fp16/t5/t5_fp16.xml \
        --out-dir <notebook>/model/ir/int8/t5
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

NNCF_VERSION = "3.4.0"
TAG = "t5_int8"


def _nncf_or_die():
    """Import nncf, or fail loudly with the exact remedy (never silently ship fp16 as 'int8')."""
    try:
        import nncf  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"[t5] int8: `import nncf` failed in THIS interpreter ({sys.executable}): "
            f"{type(exc).__name__}: {exc}\n"
            "     Remedy (one of):\n"
            f"       1. {sys.executable} -m pip install 'nncf=={NNCF_VERSION}'   # then re-run\n"
            "       2. keep the fp16 text encoder (weights='fp16', the default)"
        )
    import nncf

    return nncf


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True, help="the fp16 IR written by export_t5.py")
    ap.add_argument("--out-dir", required=True, help="where t5_int8.xml/.bin and its manifest go")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    out_dir = Path(args.out_dir).resolve()
    if not src.is_file() or not src.with_suffix(".bin").is_file():
        raise SystemExit(f"[t5] int8: no fp16 IR at {src} - export it first (ensure_t5)")
    out_dir.mkdir(parents=True, exist_ok=True)
    xml = out_dir / f"{TAG}.xml"
    manifest_path = out_dir / f"{TAG}_manifest.json"
    manifest_path.unlink(missing_ok=True)  # a manifest only ever describes a finished IR

    nncf = _nncf_or_die()
    import openvino as ov

    print(f"[t5] int8: NNCF {nncf.__version__} INT8_SYM (per-channel, data-free) on {src.name}", flush=True)
    t0 = time.perf_counter()
    model = ov.Core().read_model(src)
    model = nncf.compress_weights(model, mode=nncf.CompressWeightsMode.INT8_SYM)
    n_int8 = sum(1 for op in model.get_ops() if op.get_type_name() == "Constant" and op.get_output_element_type(0).get_type_name() == "i8")
    ov.save_model(model, xml, compress_to_fp16=False)
    del model
    seconds = round(time.perf_counter() - t0, 1)
    size_gb = round((xml.stat().st_size + xml.with_suffix(".bin").stat().st_size) / 1e9, 2)
    peak_gb = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 1)

    manifest = {
        "xml": xml.name,
        "source_xml": str(src),
        "recipe": {
            "tool": "nncf.compress_weights",
            "mode": "INT8_SYM",
            "granularity": "per-channel",
            "group_size": -1,
            "ratio": 1.0,
            "data_free": True,
            "ignored_scope": None,
            "compress_to_fp16": False,
            "execution": "bf16 on a GPU (f16 NaNs); CPU uses the fp16 IR",
        },
        "int8_weights": n_int8,
        "nncf": nncf.__version__,
        "openvino": ov.get_version(),
        "seconds": seconds,
        "size_gb": size_gb,
        "peak_rss_gb": peak_gb,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[t5] int8: {xml} ({size_gb:.2f} GB, {n_int8} int8 weights) in {seconds} s, " f"peak host memory {peak_gb} GB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
