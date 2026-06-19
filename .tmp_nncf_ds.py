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

from typing import Any, Callable, Generator, Iterable, Iterator, cast

from nncf.common.utils.api_marker import api


@api(canonical_alias="nncf.Dataset")
class Dataset:
    """
    Wrapper for passing custom user datasets into NNCF algorithms.

    This class defines the interface by which compression algorithms
    retrieve data items from the passed data source object. These data items are used
    for different purposes, for example, model inference and model validation, based
    on the choice of the exact compression algorithm.

    If the data item has been returned from the data source per iteration and it cannot be
    used as input for model inference, the transformation function is used to extract the
    model's input from this data item. For example, in supervised learning, the data item
    usually contains both examples and labels. So transformation function should extract
    the examples from the data item.

    A special input key nncf.definitions.NNCF_DATASET_RESET_STATE_KEY can be used by OpenVINO backend to control
    resetting of internal model state between model inferences. This key can be added to a dataset sample input
    dictionary with either `True` or `False` value. With `True` value, the model state will be reset before inference
    on the corresponding sample, and with `False` the state will not be reset.

    :param data_source: The iterable object serving as the source of data items.
    :param transform_func: The function that is used to extract the model's input
        from the data item. The data item here is the data item that is returned from
        the data source per iteration. This function should be passed when
        the data item cannot be directly used as model's input. If this is not specified, then the data item
        will be passed into the model as-is.
    """

    def __init__(self, data_source: Iterable[Any], transform_func: Callable[..., Any] | None = None):
        self._data_source = data_source
        self._transform_func = transform_func

    def get_data(self, indices: list[int] | None = None) -> Iterable[Any]:
        """
        Returns the iterable object that contains selected data items from the data source as-is.

        :param indices: The zero-based indices of data items that should be selected from
            the data source. The indices should be sorted in ascending order. If indices are
            not passed all data items are selected from the data source.
        :return: The iterable object that contains selected data items from the data source as-is.
        """
        return DataProvider(self._data_source, None, indices)

    def get_inference_data(self, indices: list[int] | None = None) -> Iterable[Any]:
        """
        Returns the iterable object that contains selected data items from the data source, for which
        the transformation function was applied. The item, which was returned per iteration from this
        iterable, can be used as the model's input for model inference.

        :param indices: The zero-based indices of data items that should be selected from
            the data source. The indices should be sorted in ascending order. If indices are
            not passed all data items are selected from the data source.
        :return: The iterable object that contains selected data items from the data source, for which
            the transformation function was applied.
        """
        return DataProvider(self._data_source, self._transform_func, indices)

    def get_length(self) -> int | None:
        """
        Tries to fetch length of the underlying dataset.
        :return: The length of the data_source if __len__() is implemented for it, and None otherwise.
        """
        if hasattr(self._data_source, "__len__"):
            return cast(int, self._data_source.__len__())
        return None

    def get_batch_size(self) -> int | None:
        """
        Tries to fetch batch size of the underlying dataset.
        :return: The value of batch_size or _batch_size attributes of the data_source if exist, and None otherwise.
        """
        if hasattr(self._data_source, "batch_size"):  # Torch dataloader
            return cast(int, self._data_source.batch_size)
        return None


class DataProvider:
    def __init__(
        self,
        data_source: Iterable[Any],
        transform_func: Callable[..., Any] | None,
        indices: list[int] | None = None,
    ):
        self._data_source = data_source
        if transform_func is None:
            self._transform_func = lambda x: x
        else:
            self._transform_func = transform_func
        self._indices = indices

    def __iter__(self) -> Iterator[Any]:
        if self._indices is None:
            return map(self._transform_func, self._data_source)

        if hasattr(self._data_source, "__getitem__"):
            return DataProvider._get_iterator_for_map_style(self._data_source, self._transform_func, self._indices)

        return DataProvider._get_iterator_for_iter(self._data_source, self._transform_func, sorted(self._indices))

    @staticmethod
    def _get_iterator_for_map_style(
        data_source: Iterable[Any], transform_func: Callable[..., Any], indices: list[int]
    ) -> Generator[Any, None, None]:
        for index in indices:
            yield transform_func(data_source[index])  # type: ignore[index]

    @staticmethod
    def _get_iterator_for_iter(
        data_source: Iterable[Any], transform_func: Callable[..., Any], indices: list[int]
    ) -> Generator[Any, None, None]:
        pos = 0
        num_indices = len(indices)
        for idx, data_item in enumerate(data_source):
            if pos == num_indices:
                # All specified data items were selected.
                break
            if idx == indices[pos]:
                pos = pos + 1
                yield transform_func(data_item)
