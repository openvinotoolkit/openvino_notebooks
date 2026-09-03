#!/usr/bin/env python
#
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""
Rewrite the D-FINE postprocessor TopK (top-Q of queries*classes) into a
two-stage top-k: per-query top-K classes, then global top-Q over the
queries*K survivors.

num_queries, num_classes and num_top_queries are read from the IR, so the
rewrite works for any D-FINE config. K (per-query class budget) is the only
tunable, passed in by the caller.

Exact iff no single query contributes more than K winners to the global
top-Q. With K=8 this holds in practice for D-FINE.

Usage:
    python two_stage_topk.py IN.xml OUT.xml [K=8]
"""

import sys
import numpy as np
import openvino as ov

import openvino.opset11 as opset

# TopK ports: input 0 = data, input 1 = k; output 0 = values, output 1 = indices.
TOPK_DATA_IN = 0
TOPK_K_IN = 1
TOPK_VALUES_OUT = 0
TOPK_INDICES_OUT = 1

# Divide ports: input 0 = dividend, input 1 = divisor.
DIVIDE_DIVISOR_IN = 1


def find_postproc(model):
    """Locate the postprocessor TopK and read its geometry off the graph.

    Returns (topk_node, num_queries, num_classes, num_top_queries). The
    postprocessor is the only TopK whose index output is divided by a constant
    (the `index // num_classes` that splits a flat index into (query, class)).
    """
    found = []
    for op in model.get_ordered_ops():
        if op.get_type_name() != "TopK":
            continue
        for target in op.output(TOPK_INDICES_OUT).get_target_inputs():
            div = target.get_node()
            if div.get_type_name() != "Divide":
                continue
            divisor = div.input_value(DIVIDE_DIVISOR_IN).get_node()
            if divisor.get_type_name() == "Constant":
                found.append((op, int(divisor.data.flatten()[0])))
                break

    if len(found) != 1:
        names = [op.get_friendly_name() for op, _ in found]
        raise RuntimeError("Expected exactly 1 postprocessor TopK (index output floor-divided by a " f"constant class count); found {len(found)}: {names}")

    topk, num_classes = found[0]

    shape = topk.input(TOPK_DATA_IN).get_partial_shape()
    if len(shape) != 2 or not shape[1].is_static:
        raise RuntimeError(f"Postprocessor TopK input must be [B, N] with static N; got {shape}")
    total = shape[1].get_length()
    if num_classes < 1 or total % num_classes:
        raise RuntimeError(f"TopK input width {total} is not a multiple of {num_classes} classes")

    k_node = topk.input_value(TOPK_K_IN).get_node()
    if k_node.get_type_name() != "Constant":
        raise RuntimeError("Postprocessor TopK k must be a Constant")

    return topk, total // num_classes, num_classes, int(k_node.data.flatten()[0])


def rewrite_topk(in_xml, out_xml, K=8):
    core = ov.Core()
    model = core.read_model(in_xml)
    tk, num_queries, num_classes, num_top_queries = find_postproc(model)
    if not 1 <= K <= num_classes:
        raise ValueError(f"K must be in [1, {num_classes}]; got {K}")
    print(
        f"[two-stage-topk] target TopK: {tk.get_friendly_name()}  queries={num_queries} "
        f"classes={num_classes} top={num_top_queries}  K(per-query)={K}  "
        f"survivors={num_queries * K}"
    )

    scores_flat = tk.input_value(TOPK_DATA_IN)  # [B, queries*classes], f32

    # Collect consumers before rewiring.
    val_consumers = list(tk.output(TOPK_VALUES_OUT).get_target_inputs())  # -> scores Result
    idx_consumers = list(tk.output(TOPK_INDICES_OUT).get_target_inputs())  # -> labels(Subtract), query(Divide)

    # Stage 0: reshape [B, queries*classes] -> [B, queries, classes]
    s3_shape = opset.constant(np.array([-1, num_queries, num_classes], dtype=np.int64))
    s3 = opset.reshape(scores_flat, s3_shape, special_zero=False)

    # Stage 1: per-query top-K over the class axis.
    k1 = opset.constant(np.int64(K))
    stage1 = opset.topk(s3, k1, axis=-1, mode="max", sort="value", index_element_type="i64")
    v1 = stage1.output(TOPK_VALUES_OUT)  # [B, queries, K] values
    c1 = stage1.output(TOPK_INDICES_OUT)  # [B, queries, K] class indices

    # Global flat index = class + query*classes.
    qoff = opset.constant((np.arange(num_queries, dtype=np.int64) * num_classes).reshape(1, num_queries, 1))
    flat1 = opset.add(c1, qoff)  # [B, queries, K]

    # Flatten survivors -> [B, queries*K].
    flat_shape = opset.constant(np.array([-1, num_queries * K], dtype=np.int64))
    v1f = opset.reshape(v1, flat_shape, special_zero=False)
    flat1f = opset.reshape(flat1, flat_shape, special_zero=False)

    # Stage 2: global top-num_top_queries over only queries*K survivors.
    k2 = opset.constant(np.int64(num_top_queries))
    stage2 = opset.topk(v1f, k2, axis=-1, mode="max", sort="value", index_element_type="i64")
    v2 = stage2.output(TOPK_VALUES_OUT)  # [B, top] final scores (sorted desc)
    j2 = stage2.output(TOPK_INDICES_OUT)  # [B, top] positions into the queries*K survivors

    # Recover the original flat indices for the selected survivors.
    final_index = opset.gather(flat1f, j2, opset.constant(np.int64(1)), batch_dims=1)  # [B, top], axis=1

    v2.get_node().set_friendly_name("postproc_two_stage_topk/values")
    final_index.set_friendly_name("postproc_two_stage_topk/index")

    # Rewire consumers to the new subgraph; the old TopK becomes dead and is pruned on save.
    for ci in val_consumers:
        ci.replace_source_output(v2)
    for ci in idx_consumers:
        ci.replace_source_output(final_index.output(0))

    model.validate_nodes_and_infer_types()
    ov.save_model(model, out_xml, compress_to_fp16=False)
    print(f"[two-stage-topk] wrote {out_xml}")
    return out_xml


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    in_xml, out_xml = sys.argv[1], sys.argv[2]
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    rewrite_topk(in_xml, out_xml, K)


if __name__ == "__main__":
    main()
