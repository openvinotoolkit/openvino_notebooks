from collections.abc import Iterable

from openvino import Model, PartialShape, Type, op
from openvino import opset12 as opset
from openvino.utils.types import make_constant_node

from openvino_tokenizers.constants import DETOKENIZER_NAME, STRING_OUTPUT_NAME, TOKEN_IDS_INPUT_NAME, TOKENIZER_NAME
from openvino_tokenizers.tokenizer_pipeline import (
    RegexDecodingStep,
    TokenizerPipeline,
    TrieTokenizerStep,
)
from openvino_tokenizers.utils import change_inputs_type, change_outputs_type, create_string_constant_node


def build_rwkv_tokenizer(
    rwkv_vocab: Iterable[str],
    clean_up_tokenization_spaces: bool = False,
    tokenizer_output_type: Type = Type.i64,
    detokenizer_input_type: Type = Type.i64,
) -> tuple[Model, Model]:
    from openvino_tokenizers import _get_factory, _get_opset_factory

    input_node = op.Parameter(Type.string, PartialShape(["?"]))
    input_node.set_friendly_name("string_input")

    output = _get_opset_factory("opset15").create("StringTensorUnpack", input_node.outputs()).outputs()
    trie_node = TrieTokenizerStep.from_rwkv_vocab(rwkv_vocab)
    output = trie_node.get_ov_subgraph(TokenizerPipeline.add_ragged_dimension(output))

    max_length = opset.reduce_max(
        opset.subtract(output[1], output[0]),
        make_constant_node(0, Type.i32),
    )

    output = (
        _get_factory()
        .create(
            "RaggedToDense",
            [
                *output,
                *max_length.outputs(),
                *make_constant_node(0, Type.i32).outputs(),  # default value
            ],
        )
        .outputs()[:1]
    )
    output[0].tensor.add_names({TOKEN_IDS_INPUT_NAME})

    tokenizer = Model(output, [input_node], TOKENIZER_NAME)

    detokenizer_input = op.Parameter(Type.i32, PartialShape(["?", "?"]))
    *detokenizer_output, chars = (
        _get_factory()
        .create(
            "VocabDecoder",
            [*detokenizer_input.outputs(), *create_string_constant_node(trie_node.vocab)],
        )
        .outputs()
    )
    detokenizer_output = _get_factory().create("FuzeRagged", detokenizer_output).outputs() + [chars]

    if clean_up_tokenization_spaces:
        RegexDecodingStep.clean_up_tokenization_spaces().get_ov_subgraph(detokenizer_output)

    detokenizer_output = _get_opset_factory("opset15").create("StringTensorPack", detokenizer_output).outputs()
    detokenizer_output[0].tensor.add_names({STRING_OUTPUT_NAME})

    detokenizer = Model(detokenizer_output, [detokenizer_input], DETOKENIZER_NAME)

    change_outputs_type(tokenizer, tokenizer_output_type)
    change_inputs_type(detokenizer, detokenizer_input_type)

    return tokenizer, detokenizer
