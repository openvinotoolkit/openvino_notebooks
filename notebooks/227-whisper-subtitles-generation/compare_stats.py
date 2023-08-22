import pickle
import numpy as np

from nncf.common.graph.transformations.commands import TargetType
from nncf.experimental.common.tensor_statistics.collectors import MinAggregator, MaxAggregator, MeanAggregator, \
    ShapeAggregator
from nncf.openvino.statistics.statistics import OVMinMaxTensorStatistic, OVMeanTensorStatistic

max_threshold = 1e-7


def compare_stat_dicts():
    with open('ptq_stats/decoder_reversed-15.pkl', 'rb') as f:
        stats1 = pickle.load(f)

    with open('ptq_stats/decoder_last-15.pkl', 'rb') as f:
        stats2 = pickle.load(f)

    skip_no_nop = bool(1)


    for node_name in stats1.keys():
        for algo_name in stats1[node_name].keys():
            # if algo_name == 'FBC' or algo_name == 'SQ':
            #     continue
            for stat_name in stats1[node_name][algo_name].keys():
                if stat_name == 'no_op' and skip_no_nop:
                    continue
                x1 = stats1[node_name][algo_name][stat_name]
                x2 = stats2[node_name][algo_name][stat_name]
                if isinstance(x1, np.ndarray):
                    max_diff = np.abs(x1 - x2).max()
                    mean_diff = np.abs(x1 - x2).mean()
                    if max_diff > max_threshold:
                        print(f"{node_name}, {algo_name}, {stat_name}, "
                              f"mean abs diff: {mean_diff:.2e}, max abs diff: {max_diff:.2e}")
                elif isinstance(x1, tuple):
                    assert x1 == x2, f"{node_name}, {algo_name}, {stat_name}: {x1} {x2}"
                elif isinstance(x1, list):
                    for i in range(len(x1)):
                        x1_i = x1[i]
                        x2_i = x2[i]
                        max_diff = np.abs(x1_i - x2_i).max()
                        mean_diff = np.abs(x1_i - x2_i).mean()
                        if max_diff > max_threshold:
                            print(f"{node_name}, {algo_name}, {stat_name}, "
                                  f"mean abs diff: {mean_diff:.2e}, max abs diff: {max_diff:.2e}")


def compare_stat_objects():
    def is_diff_to_big(x1, x2):
        max_diff = np.abs(x1 - x2).max()
        mean_diff = np.abs(x1 - x2).mean()
        rel_diff = np.mean(np.abs(x1 - x2) / np.abs(x1))
        if max_diff > max_threshold:
            print(f"{node_name}, {algo_name}, {str(type(tc1)).split('.')[-1]}, "
                  f"max abs diff: {max_diff:.2e}, mean abs diff: {mean_diff:.2e}, relative diff: {rel_diff:.2e}")
            return True
        return False

    with open('ptq_stats/decoder_obj_reversed-15_PTQ.pkl', 'rb') as f:
        stats1 = pickle.load(f)

    with open('ptq_stats/decoder_obj_last-15_PTQ.pkl', 'rb') as f:
        stats2 = pickle.load(f)

    for node_name in stats1.data.keys():
        for i in range(len(stats1.data[node_name])):
            for algo_name in stats1.data[node_name][i].algorithm_to_tensor_collectors.keys():
                assert len(stats1.data[node_name][i].algorithm_to_tensor_collectors[algo_name]) == 1
                assert len(stats2.data[node_name][i].algorithm_to_tensor_collectors[algo_name]) == 1
                sp1, sp2 = stats1.data[node_name][i], stats2.data[node_name][i]
                tc1 = stats1.data[node_name][i].algorithm_to_tensor_collectors[algo_name][0]
                tc2 = stats2.data[node_name][i].algorithm_to_tensor_collectors[algo_name][0]

                stat1 = tc1.get_statistics()
                stat2 = tc2.get_statistics()

                if isinstance(stat1, dict):
                    for stat_name in stat1.keys():
                        x1, x2 = stat1[stat_name], stat2[stat_name]
                        is_diff_to_big(x1, x2)
                elif isinstance(stat1, OVMinMaxTensorStatistic):
                    for attr in ["min_values", "max_values"]:
                        x1, x2 = getattr(stat1, attr), getattr(stat2, attr)
                        is_diff_to_big(x1, x2)
                elif isinstance(stat1, OVMeanTensorStatistic):
                    if sp1.target_point.type == TargetType.PRE_LAYER_OPERATION:
                        suffix = 'pre'
                    elif sp2.target_point.type == TargetType.POST_LAYER_OPERATION:
                        suffix = 'post'
                    else:
                        raise Exception(f"Can't handle such target point type: {sp1.target_point.type}")

                    x1, x2 = stat1.mean_values, stat2.mean_values
                    is_diff_to_big(x1, x2)
                    agg1, agg2 = next(iter(tc1.aggregators.values())), next(iter(tc2.aggregators.values()))

                    def count_unique(data):
                        counts = {}
                        for it in data:
                            if it not in counts:
                                counts[it] = 0
                            counts[it] += 1
                        return counts

                    counts1 = count_unique([np.mean(it.tensor) for it in agg1._container])
                    counts2 = count_unique([np.mean(it.tensor) for it in agg2._container])
                    assert len(counts1) == len(counts2) and len(set(counts1.keys()).difference(set(counts2.keys()))) == 0
                    for k in counts1.keys():
                        assert counts1[k] == counts2[k], f"{node_name} {algo_name} {k} {counts1[k]} != {counts2[k]}"
                else:
                    raise Exception(f"Can't handle such statistics type: {type(stat1)}")


 # compare_stat_dicts()
compare_stat_objects()
