#!/usr/bin/env python
"""B1: hoist the action-context K/V tower out of the action IR (pure IR-XML surgery, no export).

Source : <src>/action_fp16.xml|.bin  (shipping, read-only)
Out    : <out>/action_ctx.xml|.bin   inputs  = {context}
                                     outputs = 16 x [1,129,3072] fp16  (tower terminals)
         <out>/action_b1.xml|.bin    inputs  = {latents_action,t,16 kv...,16 actx_00..15}
                                     outputs = pred  (unchanged)
Every layer/edge outside the tower is copied verbatim; the 16 tower terminals are replaced by
Parameter layers that KEEP the original layer id (so every surviving edge stays valid) and only
have their output port renumbered to 0 (Parameter data port).

Usage: ovrun.sh python /tmp/b1/split_action.py [--out /tmp/b1] [--src <ir dir>]
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

DEFAULT_SRC = "/home/clee30/disk2/models/fasterWAM/ir/action/ir"
STOP_TYPES = {
    "Reshape",
    "Transpose",
    "Concat",
    "MatMul",
    "Einsum",
    "Softmax",
    "MatMul2D",
    "Gather",
    "Squeeze",
    "Unsqueeze",
    "VariadicSplit",
    "Split",
    "SDPA",
    "Result",
}


def load(xml: Path):
    root = ET.parse(str(xml)).getroot()
    layers = collections.OrderedDict()
    for lay in root.find("layers").findall("layer"):
        d = lay.find("data")
        outs = []
        o = lay.find("output")
        for p in (o.findall("port") if o is not None else []):
            outs.append(dict(port=int(p.get("id")), prec=p.get("precision"), shape=tuple(int(x.text) for x in p.findall("dim")), names=p.get("names")))
        ins = []
        i = lay.find("input")
        for p in (i.findall("port") if i is not None else []):
            ins.append(dict(port=int(p.get("id")), prec=p.get("precision"), shape=tuple(int(x.text) for x in p.findall("dim"))))
        layers[int(lay.get("id"))] = dict(el=lay, name=lay.get("name"), type=lay.get("type"), data=dict(d.attrib) if d is not None else {}, ins=ins, outs=outs)
    prov = {}
    fanout = collections.defaultdict(list)
    edges = []
    for e in root.find("edges").findall("edge"):
        f, fp, ti, tp = (int(e.get("from-layer")), int(e.get("from-port")), int(e.get("to-layer")), int(e.get("to-port")))
        edges.append(dict(el=e, f=f, fp=fp, t=ti, tp=tp))
        prov[(ti, tp)] = f
        fanout[f].append((ti, tp))
    return root, layers, edges, prov, fanout


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--out", default="/tmp/b1/ir")
    args = ap.parse_args()
    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    xml_in = src / "action_fp16.xml"
    bin_in = src / "action_fp16.bin"

    root, L, edges, prov, fanout = load(xml_in)
    print(f"[src] {xml_in}  layers={len(L)} edges={len(edges)} bin={bin_in.stat().st_size/1e6:.1f} MB")

    # ---------------------------------------------------------------- 1. locate A and the 16 GEMMs
    aid = [i for i, v in L.items() if v["name"] == "/Add_5"]
    assert len(aid) == 1, aid
    aid = aid[0]
    gemms = [ti for (ti, tp) in fanout[aid] if L[ti]["type"] == "MatMul"]
    assert len(gemms) == 16, gemms
    for ti in gemms:
        assert prov[(ti, 0)] == aid, (ti, prov.get((ti, 0)))
    assert sorted(prov[(ti, 1)] for ti in gemms) == sorted(prov[(ti, 1)] for ti in gemms)

    # ---------------------------------------------------------------- 2. invariant fan-out
    inv = set()
    stack = [aid]
    while stack:  # backward closure of A
        n = stack.pop()
        if n in inv:
            continue
        inv.add(n)
        for (to, tp), f in prov.items():
            if to == n:
                stack.append(f)
    inv.update(gemms)
    fwd = set(gemms)  # only nodes fed (transitively) by a GEMM may join: A's own ancestors are
    changed = True  # invariant too, but their other consumers are NOT part of the tower
    while changed:
        changed = False
        for i in sorted(L):
            if i in inv or L[i]["type"] in STOP_TYPES or L[i]["type"] == "Parameter":
                continue
            srcs = [prov[k] for k in prov if k[0] == i]
            if not srcs or not any(s in fwd for s in srcs):
                continue
            if any(s not in inv and L[s]["type"] != "Const" for s in srcs):
                continue
            inv.add(i)
            fwd.add(i)
            changed = True
    # definitive tower = backward closure of the terminal (cut) tensors
    cut = sorted(i for i in inv if L[i]["type"] != "Result" and all(to not in inv for (to, tp) in fanout[i]))
    tower = set()
    stack = list(cut)
    while stack:
        n = stack.pop()
        if n in tower:
            continue
        tower.add(n)
        for k in prov:
            if k[0] == n:
                stack.append(prov[k])
    print(f"[tower] {len(tower)} layers, {len(cut)} terminals")
    for i in cut:
        v = L[i]
        assert v["outs"][0]["shape"] == (1, 129, 3072), (i, v["name"], v["outs"][0]["shape"])
        assert v["outs"][0]["prec"] == "FP16", (i, v["outs"][0]["prec"])
        assert len(fanout[i]) >= 1
        print(f"   cut {i:5d} {v['type']:9s} {v['name'][:16]:16s} -> {len(fanout[i])} cons " f"({', '.join(sorted({L[t]['type'] for t, _ in fanout[i]}))})")
    tparams = [L[i]["name"] for i in sorted(tower) if L[i]["type"] == "Parameter"]
    print(f"[tower] Parameters inside = {tparams}")
    assert tparams == ["context"], tparams
    ctypes = collections.Counter(L[i]["type"] for i in tower)
    print(f"[tower] op mix: {ctypes.most_common()}")
    tw = sum(int(L[i]["data"].get("size", 0)) for i in tower if L[i]["type"] == "Const")
    print(f"[tower] const bytes = {tw} ({tw/1e6:.2f} MB)  of {bin_in.stat().st_size/1e6:.1f} MB")

    # ---------------------------------------------------------------- 3. helpers
    def param_layer(lid, name, shape, prec="FP16", etype="f16"):
        el = ET.Element("layer", dict(id=str(lid), name=name, type="Parameter", version="opset1"))
        ET.SubElement(el, "data", dict(shape=",".join(map(str, shape)), element_type=etype))
        o = ET.SubElement(el, "output")
        p = ET.SubElement(o, "port", dict(id="0", precision=prec, names=name))
        for d in shape:
            ET.SubElement(p, "dim").text = str(d)
        return el

    def result_layer(lid, name, src_name, shape, prec="FP16"):
        el = ET.Element("layer", dict(id=str(lid), name=f"{name}/sink_port_0", type="Result", version="opset1", output_names=name))
        i = ET.SubElement(el, "input")
        p = ET.SubElement(i, "port", dict(id="0", precision=prec))
        for d in shape:
            ET.SubElement(p, "dim").text = str(d)
        return el

    meta = root.find("rt_info")

    def write_model(xml_out: Path, bin_out: Path, layers, edges_, netname):
        """layers: list of Element in order; edges_: list of edge dicts with el elements."""
        consts = [e for e in layers if e.tag == "layer" and e.get("type") == "Const"]
        # pack bin
        blobs = []
        off = 0
        with open(bin_in, "rb") as fh:
            for c in consts:
                d = c.find("data")
                o0, sz = int(d.get("offset")), int(d.get("size"))
                fh.seek(o0)
                b = fh.read(sz)
                assert len(b) == sz, (c.get("name"), sz)
                d.set("offset", str(off))
                blobs.append(b)
                off += sz
        tmp = bin_out.with_suffix(".bin.tmp")
        with open(tmp, "wb") as fh:
            for b in blobs:
                fh.write(b)
        os.replace(tmp, bin_out)
        net = ET.Element("net", dict(name=netname, version=root.get("version", "11")))
        ls = ET.SubElement(net, "layers")
        for e in layers:
            ls.append(e)
        es = ET.SubElement(net, "edges")
        for e in edges_:
            es.append(e["el"])
        if meta is not None:
            net.append(meta)
        tree = ET.ElementTree(net)
        ET.indent(tree, space="\t")
        tree.write(str(xml_out), encoding="utf-8", xml_declaration=True)
        print(f"[write] {xml_out.name} ({xml_out.stat().st_size/1e3:.1f} kB) + " f"{bin_out.name} ({bin_out.stat().st_size/1e6:.1f} MB), consts={len(consts)}")

    # ---------------------------------------------------------------- 4. ctx model
    cutset = set(cut)
    names = {i: f"actx_{k:02d}" for k, i in enumerate(cut)}
    ctx_layers, ctx_edges = [], []
    next_id = 100000
    for i in sorted(tower):
        ctx_layers.append(L[i]["el"])
    res_pairs = []
    for i in cut:
        rid = next_id
        next_id += 1
        v = L[i]
        ctx_layers.append(result_layer(rid, names[i], v["name"], v["outs"][0]["shape"]))
        res_pairs.append((i, rid))
    for e in edges:
        if e["f"] in tower and e["t"] in tower:
            ctx_edges.append(e)
    for i, rid in res_pairs:
        ctx_edges.append(
            dict(el=ET.Element("edge", dict(**{"from-layer": str(i), "from-port": str(L[i]["outs"][0]["port"]), "to-layer": str(rid), "to-port": "0"})))
        )
    write_model(out / "action_ctx.xml", out / "action_ctx.bin", ctx_layers, ctx_edges, "action_ctx")

    # ---------------------------------------------------------------- 5. b1 model
    b1_layers, b1_edges = [], []
    for i in sorted(L):
        v = L[i]
        if i in cutset:
            b1_layers.append(param_layer(i, names[i], v["outs"][0]["shape"]))
        elif i in tower:
            continue
        elif v["type"] == "Const":
            used_outside = any(to not in tower for (to, tp) in fanout[i])
            if used_outside:
                b1_layers.append(v["el"])
        else:
            b1_layers.append(v["el"])
    # drop the now-dead `context` Parameter (only consumer was the tower's /MatMul_1)
    cid = [i for i, v in L.items() if v["type"] == "Parameter" and v["name"] == "context"][0]
    assert all(to in tower for (to, tp) in fanout[cid])
    b1_layers = [e for e in b1_layers if not (e.get("type") == "Parameter" and e.get("name") == "context")]
    for e in edges:
        f, t = e["f"], e["t"]
        if f in tower and f not in cutset:
            continue  # internal tower edge
        if t in tower and t not in cutset:
            continue  # consumer removed
        if f == cid:
            continue  # dead context input
        ok = dict(e)
        if f in cutset:  # Parameter data port is 0, not the op's
            ok["fp"] = 0
            ok["el"] = ET.Element("edge", dict(**{"from-layer": str(f), "from-port": "0", "to-layer": str(t), "to-port": str(e["tp"])}))
        b1_edges.append(ok)
    write_model(out / "action_b1.xml", out / "action_b1.bin", b1_layers, b1_edges, "action_b1")

    # ---------------------------------------------------------------- 6. structural verification
    def canonical(xml: Path):
        _, M, E, P, F = load(xml)
        ops = collections.OrderedDict()
        for i, v in M.items():
            if v["type"] in ("Parameter", "Const"):
                continue
            ops[v["name"]] = (
                v["type"],
                tuple(sorted(v["data"].items())),
                tuple(v["outs"][0]["shape"]) if v["outs"] else (),
                tuple(sorted((f"src{s}#p{k[1]}" for k, s in P.items() if k[0] == i))),
            )
        return M, ops

    M0, O0 = canonical(xml_in)
    M1, O1 = canonical(out / "action_b1.xml")
    removed_ops = {L[i]["name"] for i in tower if L[i]["type"] not in ("Const", "Parameter")}
    expect = {k: v for k, v in O0.items() if k not in removed_ops}
    print(f"\n[verify] source non-Parameter ops={len(O0)} tower_ops_removed={len(removed_ops)} " f"expected={len(expect)} b1={len(O1)}")
    assert set(expect) == set(O1), (set(expect) - set(O1), set(O1) - set(expect))
    diffs = [k for k in expect if expect[k] != O1[k]]
    print(f"[verify] ops identical to source (type/data/shape/inputs): {len(expect) - len(diffs)}" f"/{len(expect)}   diffs={diffs}")
    assert not diffs, diffs
    pin0 = [(v["name"], v["outs"][0]["shape"], v["outs"][0]["prec"]) for i, v in sorted(M0.items()) if v["type"] == "Parameter"]
    pin1 = [(v["name"], v["outs"][0]["shape"], v["outs"][0]["prec"]) for i, v in sorted(M1.items()) if v["type"] == "Parameter"]
    print(f"[verify] b1 inputs ({len(pin1)}):")
    for p in pin1:
        print("        ", p)
    po0 = [(v["name"], v["ins"][0]["shape"]) for v in M0.values() if v["type"] == "Result"]
    po1 = [(v["name"], v["ins"][0]["shape"]) for v in M1.values() if v["type"] == "Result"]
    print(f"[verify] result: source={po0} b1={po1} equal={po0 == po1}")
    assert po0 == po1
    print(f"[verify] source params {len(pin0)} -> b1 {len(pin1)}; only difference: -context, " f"+16 actx_*")
    # ctx model check
    Mc, Oc = canonical(out / "action_ctx.xml")
    cpin = [(v["name"], v["outs"][0]["shape"]) for v in Mc.values() if v["type"] == "Parameter"]
    cpout = [(v["name"], v["ins"][0]["shape"]) for v in Mc.values() if v["type"] == "Result"]
    print(f"[verify] ctx inputs={cpin} outputs={len(cpout)}x{cpout[0][1] if cpout else None} " f"ops={len(Oc)}")
    assert len(cpin) == 1 and cpin[0][0] == "context"
    assert len(cpout) == 16
    # edge-level equality outside the tower
    ok = {(e["f"], e["fp"], e["t"], e["tp"]) for e in edges if e["f"] not in tower and e["t"] not in tower}
    Mb1, _, Eb1, _, _ = load(out / "action_b1.xml")
    nk = {(e["f"], e["fp"], e["t"], e["tp"]) for e in Eb1 if e["f"] not in cutset and e["t"] not in cutset}
    print(f"[verify] edges outside tower/replaced region: source={len(ok)} b1={len(nk)} " f"identical={ok == nk}")
    assert ok == nk
    md5 = lambda p: hashlib.md5(Path(p).read_bytes()).hexdigest()[:12]
    print(f"\n[md5] source {md5(xml_in)} / {md5(bin_in)}")
    for f in ("action_ctx.xml", "action_ctx.bin", "action_b1.xml", "action_b1.bin"):
        print(f"[md5] {f:18s} {md5(out / f)}")


if __name__ == "__main__":
    main()
