# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC
from abc import abstractmethod
from itertools import islice
from pathlib import Path
from typing import Any, TypeVar

import nncf
from nncf.common import factory
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.logging import nncf_logger
from nncf.common.logging.track_progress import track
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.common.tensor_statistics.statistics import TensorStatistic
from nncf.common.tensor_statistics.statistics_serializer import dump_statistics
from nncf.common.tensor_statistics.statistics_serializer import load_statistics
from nncf.common.utils.backend import BackendType
from nncf.data.dataset import Dataset
from nncf.tensor import Tensor

TensorType = TypeVar("TensorType")
TModel = TypeVar("TModel")

EMPTY_DATASET_ERROR = (
    "Calibration dataset must not be empty. Please provide calibration dataset with at least one sample."
)


class StatisticsAggregator(ABC):
    """
    Base class for statistics collection.
    """

    BACKEND: BackendType

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.stat_subset_size = None
        self.statistic_points = StatisticPointsContainer()

    def _get_iterations_number(self) -> int | None:
        """
        Returns number of iterations.

        :return: Number of iterations for statistics collection.
        """
        dataset_length = self.dataset.get_length()
        if dataset_length and self.stat_subset_size:
            return min(dataset_length, self.stat_subset_size)
        return dataset_length or self.stat_subset_size

    def collect_statistics(self, model: TModel, graph: NNCFGraph) -> None:
        """
        Collects statistics for registered StatisticPoints.
        The statistics are stored in self.statistic_points.

        :param model: Backend-specific model instance.
        :param graph: Model graph.
        """
        if not self.statistic_points:
            return
        model_transformer = factory.ModelTransformerFactory.create(model)
        merged_statistics = self._get_merged_statistic_points(self.statistic_points, model, graph)
        transformation_layout = self._get_transformation_layout_extra_outputs(merged_statistics)
        model_with_outputs: TModel = model_transformer.transform(transformation_layout)
        engine = factory.EngineFactory.create(model_with_outputs)
        iterations_number = self._get_iterations_number()
        processed_samples = 0
        for input_data in track(
            islice(self.dataset.get_inference_data(), iterations_number),
            total=iterations_number,
            description="Statistics collection",
        ):
            outputs = engine.infer(input_data)
            processed_outputs = self._process_outputs(outputs)
            self._register_statistics(processed_outputs, merged_statistics)
            # Manually dereference output tensors to hint gc to remove them. Without it,
            # the processed_outputs and outputs remain during model inference,
            # increasing the peak memory consumption.
            del processed_outputs
            del outputs
            processed_samples += 1
        if processed_samples == 0:
            raise nncf.ValidationError(EMPTY_DATASET_ERROR)
        if self.stat_subset_size is not None and self.stat_subset_size > processed_samples:
            nncf_logger.warning(
                f"Dataset contains only {processed_samples} samples, "
                f"smaller than the requested subset size {self.stat_subset_size}."
            )

    def load_statistics_from_dir(self, dir_path: Path) -> None:
        """
        Loads statistics from a directory and populates the statistic points with the loaded data.

        :param dir_path: The name of the directory from which to load the statistics.
        """
        loaded_data = load_statistics(dir_path, self.BACKEND)
        self._load_statistics(loaded_data)
        nncf_logger.info(f"Statistics were successfully loaded from a directory {dir_path.absolute()}")

    def _load_statistics(self, data: dict[str, Any]) -> None:
        """
        Loads statistics into the registered statistic points from the given data.

        :param data: A dictionary containing the statistics loaded from a file.
        """
        for _, statistic_point, tensor_collector in self.statistic_points.get_tensor_collectors():
            statistics = tensor_collector.get_statistics()
            if not isinstance(statistics, TensorStatistic):
                msg = (
                    "Please use nncf.common.tensor_statistics.statistics.TensorStatistic"
                    " initializing the TensorCollector to make it possible to dump and load statistics"
                )
                raise nncf.InternalError(msg)
            statistics_key = self._get_statistics_key(statistics, statistic_point.target_point)
            if statistics_key not in data:
                msg = f"Not found statistics for {statistics_key}"
                raise nncf.ValidationError(msg)
            statistics.load_data(data[statistics_key])
            tensor_collector.set_cache(statistics)

    def dump_statistics(self, dir_path: Path) -> None:
        """
        Dumps the current statistics to a directory in a compressed format.

        :param dir_path: The path of the directory where the statistics will be saved.
        """
        data_to_dump = self._prepare_statistics()
        additional_metadata = {"subset_size": self.stat_subset_size}
        dump_statistics(data_to_dump, dir_path, self.BACKEND, additional_metadata)
        nncf_logger.info(f"Statistics were successfully saved to a directory {dir_path.absolute()}")

    def _prepare_statistics(self) -> dict[str, Any]:
        """
        Prepares the statistics data for dumping into a directory.

        :return: A dictionary containing the statistics data to be dumped.
        """
        data_to_dump = {}
        for _, statistic_point, tensor_collector in self.statistic_points.get_tensor_collectors():
            statistics = tensor_collector.get_statistics()
            if not isinstance(statistics, TensorStatistic):
                msg = (
                    "Please use nncf.common.tensor_statistics.statistics.TensorStatistic"
                    " initializing the TensorCollector to make it possible to dump and load statistics"
                )
                raise nncf.InternalError(msg)
            statistics_key = self._get_statistics_key(statistics, statistic_point.target_point)
            data = statistics.get_data(is_serialized=True)
            data_to_dump[statistics_key] = data
        return data_to_dump

    def register_statistic_points(self, statistic_points: StatisticPointsContainer) -> None:
        """
        Register statistic points for statistics collection and recalculates the maximum number samples
        for collecting statistics, based on the maximum value from the all algorithms.

        :param statistic_points: StatisticPointsContainer instance with the statistic points
        """
        for _statistic_points in statistic_points.values():
            for _statistic_point in _statistic_points:
                self.statistic_points.add_statistic_point(_statistic_point)

        for _statistic_points in self.statistic_points.values():
            for _statistic_point in _statistic_points:
                for tensor_collectors in _statistic_point.algorithm_to_tensor_collectors.values():
                    for tensor_collector in tensor_collectors:
                        if self.stat_subset_size is None:
                            self.stat_subset_size = tensor_collector.num_samples
                        elif tensor_collector.num_samples is not None:
                            self.stat_subset_size = max(self.stat_subset_size, tensor_collector.num_samples)

    @abstractmethod
    def _register_statistics(self, outputs: dict[str, Tensor], statistic_points: StatisticPointsContainer) -> None:
        """
        Process prepared raw model outputs and statistic points for the further usage.

        :param outputs: prepared raw model outputs
        :param statistic_points: StatisticPointsContainer instance with the statistic points
        """

    @abstractmethod
    def _get_transformation_layout_extra_outputs(
        self, statistic_points: StatisticPointsContainer
    ) -> TransformationLayout:
        """
        Creates backend-specific transformation layout for the further statistics collection.

        :param statistic_points: StatisticPointsContainer to add outputs
        :return: TransformationLayout with the corresponding transformations
        """

    @staticmethod
    @abstractmethod
    def _get_merged_statistic_points(
        statistic_points: StatisticPointsContainer, model: TModel, graph: NNCFGraph
    ) -> StatisticPointsContainer:
        """
        Creates a new StatisticPointContainer that has no duplicated tensor collectors for one
        unique statistic point. Alters statistic collectors in the given statistic point container so statistics
        collected by merged statistic collectors will be available in all corresponding statistic collectors
        from the given statistic point container.

        :param statistic_points: Registered statistic points with possible tensor collectors duplicates.
        :param model: Backend-specific target model.
        :param graph: Model graph.
        :return: Merged statistic points container bounded with given statistic point container.
        """

    @staticmethod
    @abstractmethod
    def _process_outputs(outputs: Any) -> dict[str, Tensor]:
        """
        Post-process model outputs for the further statistics collection.

        :param outputs: raw model outputs
        :return: processed model outputs in Dict[str, Tensor] format
        """

    @abstractmethod
    def _get_statistics_key(self, statistics: TensorStatistic, target_point: TargetPoint) -> str:
        """
        Returns key of statistics.

        :param statistics: Statistics value.
        :param target_point: Statistics target point.
        :return: Statistics key.
        """
