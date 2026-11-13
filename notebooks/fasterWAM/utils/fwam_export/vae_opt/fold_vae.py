#!/usr/bin/env python
"""Fold the VAE-encoder's dead depth taps at IR level.

Every Conv3D with kernel[0]==3 in vae_fp16.xml sits behind a Pad on a D=1 tensor that
puts the single frame at depth index p (p=2 for this export).  The Pad's two *other*
depth taps therefore multiply exact zeros.  =>  Conv3D(kd=3, W[:,:,p]) == Conv3D(kd=1)
with W[:,:,p:p+1], the Pad's spatial pads absorbed into the convolution, and the Pad
deleted.

Two entry points
----------------
`fold_graph(model, log=print)` -- the LIBRARY call, used by `scripts/export_openvino.py`
    `--unit vae` (default ON since the 31 Oct promotion) right after the ONNX->OV
    conversion.  It rebuilds the convolutions in an in-memory `ov.Model` and returns the
    MAC accounting; it does NO file I/O (the caller owns save/reload).
`main()` -- the standalone CLI (re)generates the durable artifact `vae_fold.xml|.bin`
    from a source IR and re-verifies the SAVED file.

Implementation notes
--------------------
* The standalone script reads its source from a COPY in /tmp (never from ~/fasterWAM/ir*).
* OpenVINO stores Pad's pads_begin/pads_end as *runtime* int64 inputs computed from a
  small constant sub-graph (Concat -> Reshape -> Slice -> Transpose -> Reshape ->
  Convert -> Split).  A tiny numpy evaluator (ops: Constant/ConstantOfShape/Broadcast/
  Concat/Reshape/Transpose/Slice/Convert/Split/Squeeze/Unsqueeze/Gather/ShapeOf/Add/
  Sub/Mul/Div) recovers the literal pad values so p is *read from the IR*, not guessed.
* The weight slice is exact (no re-quantisation): the filter Constant keeps its dtype
  and is sliced on its depth-kernel axis (2 for Convolution, 3 for GroupConvolution).
* Dead nodes (old Convolutions, their constants, orphaned Pads) are simply not
  reachable from the Results any more, so ov.save_model drops them.

usage:  $FW_PY fold_vae.py [<src xml> [<out xml>]]
"""

from __future__ import annotations

import os
import shutil
import sys

import numpy as np
import openvino as ov

SRC_DEFAULT = "/tmp/vaeopt/vae_ctrl/vae_fp16.xml"
FOLD_DIR = "/tmp/vaeopt/fold"
DOC = "/tmp/vaeopt/fold_log.txt"

# MAC total the fold must land on for THIS VAE encoder (233.0 -> 81.4 GMAC); the standalone
# script asserts it so a silently-degraded fold can never produce a durable artifact.
EXPECTED_MACS_AFTER_GMAC = 81.4
EXPECTED_MACS_TOL = 0.02


# ---------------------------------------------------------------- mini evaluator
class Ev:
    """Evaluates the constant sub-graphs we meet as Pad pad-amount producers."""

    def __init__(self):
        self.cache = {}
        self.hits = []

    def val(self, out):
        key = (out.get_node().get_instance_id(), out.get_index())
        if key in self.cache:
            return self.cache[key]
        n = out.get_node()
        t = n.get_type_name()
        v = self._eval(n, t, out)
        self.cache[key] = v
        self.hits.append(t)
        return v

    def ins(self, n):
        return [self.val(n.input(i).get_source_output()) for i in range(n.get_input_size())]

    def _eval(self, n, t, out):
        if t == "Constant":
            return np.array(n.get_data())
        a = n.get_attributes()
        if t == "Convert":
            v = self.val(n.input(0).get_source_output())
            tname = str(out.get_element_type())
            np_dt = {"f16": np.float16, "f32": np.float32, "f64": np.float64, "i8": np.int8, "u8": np.uint8, "i32": np.int32, "i64": np.int64}.get(tname)
            return v.astype(np_dt) if np_dt is not None else v
        if t == "ShapeOf":
            return np.array(self.val(n.input(0).get_source_output()).shape, dtype=np.int64)
        if t == "ConstantOfShape":
            shape = self.val(n.input(0).get_source_output()).astype(np.int64)
            fill = 0
            if n.get_input_size() > 1:
                fill = self.val(n.input(1).get_source_output()).ravel()[0]
            return np.full(shape, fill, dtype=np.float32 if "f32" in str(out.get_element_type()) else np.int64)
        if t == "Broadcast":
            d = self.val(n.input(0).get_source_output())
            shp = self.val(n.input(1).get_source_output()).astype(np.int64)
            mode = a.get("broadcast_spec", a.get("mode", "NUMPY"))
            if str(mode).lower().endswith("explicit"):
                ax = self.val(n.input(2).get_source_output()).astype(np.int64)
                s = [1] * len(shp)
                for i, ax_i in enumerate(ax):
                    s[int(ax_i)] = d.shape[i]
                d = d.reshape(s)
            return np.broadcast_to(d, tuple(int(x) for x in shp)).copy()
        if t == "Concat":
            return np.concatenate(self.ins(n), axis=int(a.get("axis", 0)))
        if t == "Reshape":
            d = self.val(n.input(0).get_source_output())
            shp = [int(x) for x in self.val(n.input(1).get_source_output()).ravel()]
            if a.get("special_zero", False):
                shp = list(d.shape)
                raw = [int(x) for x in self.val(n.input(1).get_source_output()).ravel()]
                shp = [d.shape[i] if r == 0 else r for i, r in enumerate(raw)]
            return d.reshape(shp)
        if t == "Squeeze":
            d = self.val(n.input(0).get_source_output())
            if n.get_input_size() > 1:
                ax = tuple(int(x) for x in self.val(n.input(1).get_source_output()).ravel())
            else:
                ax = tuple(int(x) for x in a.get("axes", []))
            return np.squeeze(d, axis=ax) if ax else np.squeeze(d)
        if t == "Unsqueeze":
            d = self.val(n.input(0).get_source_output())
            ax = self.val(n.input(1).get_source_output()).ravel() if n.get_input_size() > 1 else a.get("axes", [])
            for ax_i in sorted(int(x) for x in ax):
                d = np.expand_dims(d, ax_i)
            return d
        if t == "Transpose":
            d = self.val(n.input(0).get_source_output())
            order = self.val(n.input(1).get_source_output()).ravel() if n.get_input_size() > 1 else a.get("order", [])
            return np.transpose(d, axes=[int(x) for x in order])
        if t == "Slice":
            d = self.val(n.input(0).get_source_output())
            v = [self.val(n.input(i).get_source_output()) for i in range(1, n.get_input_size())]
            start = np.atleast_1d(v[0]).astype(np.int64)
            stop = np.atleast_1d(v[1]).astype(np.int64)
            step = np.atleast_1d(v[2]).astype(np.int64) if len(v) > 2 else np.ones_like(start)
            axes = np.atleast_1d(v[3]).astype(np.int64) if len(v) > 3 else np.arange(len(start))
            INT64_MAX = 2**63 - 1
            INT64_MIN = -(2**63)
            idx = [slice(None)] * d.ndim
            for s, e, st, ax in zip(start, stop, step, axes):
                ax = int(ax) % d.ndim
                if e == INT64_MIN:
                    e = None if st > 0 else INT64_MIN + 1
                idx[ax] = slice(int(s), None if e is None or e >= INT64_MAX else int(e), int(st))
            return d[tuple(idx)]
        if t == "Split":
            d = self.val(n.input(0).get_source_output())
            axis = int(a.get("axis", 0))
            k = int(a.get("num_splits", 0) or 0)
            if k <= 0 and n.get_input_size() > 1:  # lengths given as a 1-D input
                lens = [int(x) for x in self.val(n.input(1).get_source_output()).ravel()]
            else:
                k = k or 2
                lens = [d.shape[axis] // k] * k
            parts = np.split(d, np.cumsum(lens)[:-1], axis=axis)
            return parts[int(out.get_index())]
        if t == "Gather":
            d = self.val(n.input(0).get_source_output())
            i = self.val(n.input(1).get_source_output()).astype(np.int64)
            return np.take(d, i, axis=int(a.get("axis", 0)))
        if t in ("Add", "Subtract", "Multiply", "Divide"):
            x, y = self.ins(n)[:2]
            return {"Add": np.add, "Subtract": np.subtract, "Multiply": np.multiply, "Divide": np.divide}[t](x, y)
        raise RuntimeError(f"mini-evaluator: unsupported op {t} ({n.get_friendly_name()})")


def pad_amounts(pad_node, ev=None):
    """(pads_begin, pads_end) as python int lists."""
    ev = ev or Ev()
    tot = []
    for i in (1, 2):
        v = ev.val(pad_node.input(i).get_source_output())
        tot.append([int(x) for x in np.ravel(v)])
    return tot[0], tot[1]


# ---------------------------------------------------------------------- helpers
def macs(node, filt_shape):
    """MACs = out_elems * (per-output-channel taps) = prod(out) * prod(filters[1:]).

    filters are [O, I/g, kd, ky, kx] (GroupConvolution: [g, O/g, I/g, kd, ky, kx]); the
    output element count already contains O (or g*O/g), so only filters[1:] is missing."""
    o = [int(x) for x in node.output(0).get_partial_shape().get_shape()]
    return int(np.prod(o)) * int(np.prod(filt_shape[1:]))


def depth_kernel_axis(conv):
    return 3 if conv.get_type_name() == "GroupConvolution" else 2


def get_pads(conv):
    a = conv.get_attributes()
    return ([int(x) for x in a["pads_begin"]], [int(x) for x in a["pads_end"]])


def convs(model):
    return [o for o in model.get_ops() if o.get_type_name() in ("Convolution", "GroupConvolution")]


def conv_macs(model):
    return sum(macs(c, [int(x) for x in c.input(1).get_source_output().get_partial_shape().get_shape()]) for c in convs(model))


def depth_pad_conv(conv, ev):
    """True if `conv` is a rank-5 Conv3D with kernel[0]==3 on a depth-only-padded D=1 input.

    This is exactly the shape the fold removes, so after a successful fold it must be False
    for every convolution in the model -- that is the assertion the exporter makes.
    """
    if len(conv.input(0).get_partial_shape()) != 5:
        return False
    fshape = [int(x) for x in conv.input(1).get_source_output().get_partial_shape().get_shape()]
    if fshape[depth_kernel_axis(conv)] != 3:
        return False
    prod = conv.input(0).get_source_output().get_node()
    if prod.get_type_name() != "Pad":
        return False
    in_shape = [int(x) for x in prod.input(0).get_partial_shape().get_shape()]
    if len(in_shape) < 3 or in_shape[2] != 1:
        return False
    try:
        pb, pe = pad_amounts(prod, ev)
    except Exception:  # noqa: BLE001
        return False
    return len(pb) == 5 and len(pe) == 5 and pb[0] == pb[1] == 0 and pe[0] == pe[1] == 0


def remaining_depth_taps(model, ev=None):
    """Names of convolutions that STILL have a kd==3 depth tap on a D=1 padded input."""
    ev = ev or Ev()
    return [c.get_friendly_name() for c in convs(model) if depth_pad_conv(c, ev)]


# -------------------------------------------------------------------- the fold
def fold_graph(model, log=print):
    """Fold every dead-depth-tap convolution of `model` in place; no file I/O.

    Returns {"model", "mac_before", "mac_after", "folded", "skipped", "details"} where
    `skipped` is [(conv name, reason, GMAC)] for the convolutions left alone.
    """
    ev = Ev()
    mac_before = mac_after = 0
    folded = 0
    skipped = []
    details = []

    def note(s=""):
        details.append(s)
        log(s)

    note("=== per-node ===")
    for conv in list(model.get_ops()):
        if conv.get_type_name() not in ("Convolution", "GroupConvolution"):
            continue
        filt = conv.input(1).get_source_output()
        fshape = [int(x) for x in filt.get_partial_shape().get_shape()]
        kaxis = depth_kernel_axis(conv)
        kd = fshape[kaxis]
        rank = len(conv.input(0).get_partial_shape())
        mb = macs(conv, fshape)
        mac_before += mb
        if rank != 5 or kd != 3:
            mac_after += mb
            skipped.append((conv.get_friendly_name(), f"rank={rank} kd={kd}", mb))
            continue
        src_out = conv.input(0).get_source_output()
        prod = src_out.get_node()
        if prod.get_type_name() != "Pad":
            skipped.append((conv.get_friendly_name(), "input is not a Pad", mb))
            mac_after += mb
            continue
        pb, pe = pad_amounts(prod, ev)
        in_shape = [int(x) for x in prod.input(0).get_partial_shape().get_shape()]
        pad_out = [int(x) for x in prod.output(0).get_partial_shape().get_shape()]
        cpads = get_pads(conv)
        strides = [int(x) for x in conv.get_attributes()["strides"]]
        dils = [int(x) for x in conv.get_attributes()["dilations"]]
        auto_pad = str(conv.get_attributes()["auto_pad"])
        ok = (
            len(pb) == 5
            and len(pe) == 5
            and len(cpads[0]) == 3 == len(cpads[1])
            and pb[0] == pb[1] == 0
            and pe[0] == pe[1] == 0
            and in_shape[0] == 1
            and in_shape[2] == 1
            and auto_pad.lower() == "explicit"
            and strides[0] == 1
        )
        if not ok:
            skipped.append((conv.get_friendly_name(), f"pad/attr mismatch {pb}{pe} {auto_pad}", mb))
            mac_after += mb
            continue
        p = pb[2]
        # the single frame must sit at depth index p of the padded tensor
        assert pad_out[2] == 1 + pb[2] + pe[2], (pad_out, pb, pe)
        # conv pads_begin/end are per-SPATIAL-dim (len 3 for a 3-D conv); the Pad op's are
        # full-rank (len 5, dims N,C,D,H,W).  The convolution now does the spatial padding
        # itself, so its new pads = Pad's spatial pads + its own previous pads (depth -> 0).
        new_pb = [0, cpads[0][1] + pb[3], cpads[0][2] + pb[4]]
        new_pe = [0, cpads[1][1] + pe[3], cpads[1][2] + pe[4]]

        # ---- slice the filter Constant (exact) -----------------------------
        wnode = filt.get_node()
        wdt = filt.get_element_type()
        if wnode.get_type_name() == "Convert":  # f16 Constant -> Convert -> f32
            bnode = wnode.input(0).get_source_output().get_node()
            assert bnode.get_type_name() == "Constant", bnode.get_type_name()
            w = np.array(bnode.get_data())
            base_type = wnode.input(0).get_source_output().get_element_type()
            slicer = np.ascontiguousarray(np.take(w, [p], axis=kaxis))
            new_const = ov.opset1.constant(slicer)
            filters = ov.opset1.convert(new_const, wdt).output(0)
        elif wnode.get_type_name() == "Constant":
            w = np.array(wnode.get_data())
            slicer = np.ascontiguousarray(np.take(w, [p], axis=kaxis))
            filters = ov.opset1.constant(slicer).output(0)
        else:
            raise RuntimeError(f"unexpected filter producer {wnode.get_type_name()}")
        assert slicer.dtype == w.dtype, (slicer.dtype, w.dtype)

        # ---- rebuild the convolution --------------------------------------
        data = prod.input(0).get_source_output()  # bypass the (now useless) Pad
        if conv.get_type_name() == "Convolution":
            new = ov.opset1.convolution(data, filters, strides, new_pb, new_pe, dils)
        else:
            new = ov.opset1.group_convolution(data, filters, strides, new_pb, new_pe, dils)
        new.set_friendly_name(conv.get_friendly_name() + "/fold")
        conv.output(0).replace(new.output(0))

        fnew = list(slicer.shape)
        ma = macs(new, fnew)
        mac_after += ma
        folded += 1
        note(
            f"{conv.get_friendly_name()}\n"
            f"   in {prod.input(0).get_partial_shape()} ->pad {pb}/{pe}-> {prod.output(0).get_partial_shape()}"
            f"  live tap p={p}\n"
            f"   W {fshape} -> {fnew}   kernel [{kd},{fshape[kaxis+1]},{fshape[kaxis+2]}] -> [1,{fnew[kaxis+1]},{fnew[kaxis+2]}]\n"
            f"   pads {cpads[0]}/{cpads[1]} + pad {pb}/{pe} -> {new_pb}/{new_pe}\n"
            f"   MAC {mb/1e9:8.3f} -> {ma/1e9:8.3f} GMAC   (running total {mac_after/1e9:8.3f})"
        )

    note("=== skipped (not folded) ===")
    for name, why, mb in skipped:
        note(f"  {name}  [{why}]  {mb/1e9:.3f} GMAC")
    note(
        "MAC total before = %.3f GMAC   after = %.3f GMAC   reduction %.2fx  folded=%d"
        % (mac_before / 1e9, mac_after / 1e9, mac_before / max(mac_after, 1e-9), folded)
    )

    return {"model": model, "mac_before": mac_before, "mac_after": mac_after, "folded": folded, "skipped": skipped, "details": details}


def verify_model(model, expected_macs=None, tol=EXPECTED_MACS_TOL):
    """Re-check a (saved+reloaded) model: no depth-tap convolution left, MACs as expected."""
    kd3 = []
    for o in convs(model):
        fs = [int(x) for x in o.input(1).get_source_output().get_partial_shape().get_shape()]
        if len(fs) == 5 and fs[depth_kernel_axis(o)] == 3:
            kd3.append(o.get_friendly_name())
    mac_total = conv_macs(model)
    left = remaining_depth_taps(model)
    assert not left, f"Conv3D kd=3 on a depth-only-padded D=1 input still present: {left}"
    if expected_macs is not None:
        assert abs(mac_total / 1e9 - expected_macs) / expected_macs <= tol, f"MAC total {mac_total/1e9:.3f} not ~{expected_macs} GMAC"
    return {"n_ops": len(model.get_ops()), "n_convs": len(convs(model)), "kd3_convs": kd3, "macs_gmac": mac_total / 1e9, "depth_taps_left": left}


# ------------------------------------------------------------------------- main
def main():
    src = sys.argv[1] if len(sys.argv) > 1 else SRC_DEFAULT
    os.makedirs(os.path.join(FOLD_DIR, "src"), exist_ok=True)
    copy_xml = os.path.join(FOLD_DIR, os.path.basename(src))
    assert os.path.abspath(copy_xml) != os.path.abspath(src), "refuse to fold in place"
    if not os.path.exists(copy_xml):
        shutil.copy2(src, copy_xml)
        shutil.copy2(src.replace(".xml", ".bin"), copy_xml.replace(".xml", ".bin"))
    copy_bin = copy_xml.replace(".xml", ".bin")

    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    log(f"# fold_vae.py  ov={ov.__version__}")
    log(f"src (copy) = {copy_xml}  bin {os.path.getsize(copy_bin)} B")

    core = ov.Core()
    model = core.read_model(copy_xml)
    n_ops_src = len(model.get_ops())
    log(f"read_model: {n_ops_src} ops, out " f"{[str(o.get_partial_shape()) for o in model.outputs]}")

    stats = fold_graph(model, log=log)

    out_xml = os.path.join(FOLD_DIR, "vae_fold.xml")
    ov.save_model(model, out_xml, compress_to_fp16=False)
    log(f"saved {out_xml}  bin {os.path.getsize(out_xml.replace('.xml', '.bin'))} B")

    # ---- re-read the SAVED model and check it -------------------------------
    m2 = core.read_model(out_xml)
    chk = verify_model(m2, expected_macs=EXPECTED_MACS_AFTER_GMAC)
    log(f"reloaded: {chk['n_ops']} ops, {chk['n_convs']} convolutions, " f"kd==3 remaining: {chk['kd3_convs']}")
    log(f"reloaded MAC total = {chk['macs_gmac']:.3f} GMAC   out={m2.outputs[0].get_any_name()} " f"{m2.outputs[0].get_partial_shape()}")
    ctrl_bin = os.path.getsize(copy_bin)
    fold_bin = os.path.getsize(out_xml.replace(".xml", ".bin"))
    log(f"bin: {ctrl_bin} -> {fold_bin} B  ({fold_bin/ctrl_bin*100:.1f}% of control, " f"-{(ctrl_bin-fold_bin)/1e6:.1f} MB)")
    log(f"ops: {n_ops_src} -> {chk['n_ops']}")
    with open(DOC, "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
