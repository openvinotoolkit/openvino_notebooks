import inspect
from functools import wraps
from contextlib import contextmanager

import datasets
from openvino.runtime import Core
from pathlib import Path

from nncf.quantization.advanced_parameters import AdvancedAccuracyRestorerParameters
from nncf.quantization.range_estimator import RangeEstimatorParameters, StatisticsCollectorParameters, StatisticsType, \
    AggregatorType
from optimum.intel.openvino.modeling_seq2seq import OVEncoder, OVDecoder
from transformers import pipeline, AutoTokenizer
from optimum.intel.openvino import OVModelForSeq2SeqLM, OVModelForSequenceClassification
import re
import transformers
from tqdm import tqdm
import datetime
from evaluate import load
import pprint
import pickle

import openvino.runtime as ov
import nncf

core = Core()

wer = load("wer")

DEVICE = 'CPU'
VERBOSE = bool(1)

CALIBRATION_DATA_CACHE = Path("./calibration_data")
COLLECT_CALIBRATION_DATA = False

total_check_time = 0
total_correction_time = 0

encoder_call_count = 0
encoder_total_time = 0

decoder_call_count = 0
decoder_total_time = 0

decoder_with_past_call_count = 0
decoder_with_past_total_time = 0

total_corrected_tokens = 0


@contextmanager
def calibration_data_collection():
    global COLLECT_CALIBRATION_DATA
    try:
        COLLECT_CALIBRATION_DATA = True
        yield
    finally:
        COLLECT_CALIBRATION_DATA = False


def load_grammar_cheker():
    grammar_checker_model_id = "textattack/roberta-base-CoLA"
    grammar_checker_dir = Path("roberta-base-cola")
    grammar_checker_tokenizer = AutoTokenizer.from_pretrained(grammar_checker_model_id)

    if grammar_checker_dir.exists():
        grammar_checker_model = OVModelForSequenceClassification.from_pretrained(grammar_checker_dir, device=DEVICE)
    else:
        grammar_checker_model = OVModelForSequenceClassification.from_pretrained(grammar_checker_model_id, export=True,
                                                                                 device=DEVICE)
        grammar_checker_model.save_pretrained(grammar_checker_dir)
    grammar_checker_pipe = pipeline("text-classification", model=grammar_checker_model,
                                    tokenizer=grammar_checker_tokenizer)
    return grammar_checker_tokenizer, grammar_checker_pipe


def load_grammar_corrector():
    grammar_corrector_model_id = "pszemraj/flan-t5-large-grammar-synthesis"
    grammar_corrector_dir = Path("flan-t5-large-grammar-synthesis")
    grammar_corrector_tokenizer = AutoTokenizer.from_pretrained(grammar_corrector_model_id)

    if grammar_corrector_dir.exists():
        grammar_corrector_model = OVModelForSeq2SeqLM.from_pretrained(grammar_corrector_dir, device=DEVICE)
    else:
        grammar_corrector_model = OVModelForSeq2SeqLM.from_pretrained(grammar_corrector_model_id, export=True,
                                                                      device=DEVICE)
        grammar_corrector_model.save_pretrained(grammar_corrector_dir)
    grammar_corrector_pipe = pipeline("text2text-generation", model=grammar_corrector_model,
                                      tokenizer=grammar_corrector_tokenizer)
    return grammar_corrector_tokenizer, grammar_corrector_pipe


def split_text(text: str) -> list:
    """
    Split a string of text into a list of sentence batches.

    Parameters:
    text (str): The text to be split into sentence batches.

    Returns:
    list: A list of sentence batches. Each sentence batch is a list of sentences.
    """
    # Split the text into sentences using regex
    sentences = re.split(r"(?<=[^A-Z].[.?]) +(?=[A-Z])", text)

    # Initialize a list to store the sentence batches
    sentence_batches = []

    # Initialize a temporary list to store the current batch of sentences
    temp_batch = []

    # Iterate through the sentences
    for sentence in sentences:
        # Add the sentence to the temporary batch
        temp_batch.append(sentence)

        # If the length of the temporary batch is between 2 and 3 sentences, or if it is the last batch, add it to the
        # list of sentence batches
        if len(temp_batch) >= 2 and len(temp_batch) <= 3 or sentence == sentences[-1]:
            sentence_batches.append(temp_batch)
            temp_batch = []

    return sentence_batches


def correct_text(text: str, checker: transformers.pipelines.Pipeline, corrector: transformers.pipelines.Pipeline,
                 separator: str = " ", disable_tqdm=False) -> str:
    """
    Correct the grammar in a string of text using a text-classification and text-generation pipeline.

    Parameters:
    text (str): The inpur text to be corrected.
    checker (transformers.pipelines.Pipeline): The text-classification pipeline to use for checking the grammar quality
        of the text.
    corrector (transformers.pipelines.Pipeline): The text-generation pipeline to use for correcting the text.
    separator (str, optional): The separator to use when joining the corrected text into a single string. Default is a
        space character.

    Returns:
    str: The corrected text.
    """
    # Split the text into sentence batches
    sentence_batches = split_text(text)

    # Initialize a list to store the corrected text
    corrected_text = []

    global total_correction_time, total_check_time, total_corrected_tokens
    total_corrected_tokens = total_correction_time = total_check_time = 0

    # Iterate through the sentence batches
    for batch in tqdm(
            sentence_batches, total=len(sentence_batches), desc="correcting text..", disable=disable_tqdm
    ):
        # Join the sentences in the batch into a single string
        raw_text = " ".join(batch)

        # Check the grammar quality of the text using the text-classification pipeline
        start_time = datetime.datetime.now()
        results = checker(raw_text)
        total_check_time += (datetime.datetime.now() - start_time).total_seconds()

        # Only correct the text if the results of the text-classification are not LABEL_1 or are LABEL_1 with a score below 0.9
        if results[0]["label"] != "LABEL_1" or (
                results[0]["label"] == "LABEL_1" and results[0]["score"] < 0.9
        ):
            # Correct the text using the text-generation pipeline
            total_corrected_tokens += len(corrector.tokenizer(raw_text)["input_ids"])
            start_time = datetime.datetime.now()
            corrected_batch = corrector(raw_text)
            total_correction_time += (datetime.datetime.now() - start_time).total_seconds()
            corrected_text.append(corrected_batch[0]["generated_text"])
        else:
            corrected_text.append(raw_text)

    # Join the corrected text into a single string
    corrected_text = separator.join(corrected_text)

    return corrected_text


def arg_logger(func):
    @wraps(func)
    def new_func(*args, **kwargs):
        signature = inspect.signature(func)
        bound_args = signature.bind(*args, **kwargs)
        arguments_dict = dict(bound_args.arguments)

        # if func.__name__ == "quantize_decoder_with_accuracy_control":
        #     save_dir = Path(arguments_dict["model_path"]) / arguments_dict["save_dir"]
        # else:
        #     save_dir = Path(arguments_dict.get("save_dir", arguments_dict.get("model_path")))
        # if not save_dir.exists():
        #     save_dir.mkdir(parents=True)
        # log_file_path = Path(save_dir) / f"{func.__name__}_{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.log"
        # map(logger.removeHandler, logger.handlers)
        # logger.addHandler(logging.FileHandler(log_file_path))
        #
        # logger.info(f"{func} called with args: {pprint.pformat(arguments_dict)}")

        print(f"{func} called with args: {pprint.pformat(arguments_dict)}")
        return func(*args, **kwargs)

    return new_func


def add_encoder_decoder_wrappers(encoder, decoder, decoder_with_past):
    def wrap(model):
        original_forward = model.forward

        def wrapper(*args, **kwargs):
            global encoder_call_count, encoder_total_time, decoder_call_count, decoder_total_time, \
                decoder_with_past_call_count, decoder_with_past_total_time
            start_time = datetime.datetime.now()
            result = original_forward(*args, **kwargs)
            end_time = datetime.datetime.now()
            time_delta = (end_time - start_time).total_seconds()
            if isinstance(model, OVEncoder):
                encoder_call_count += 1
                encoder_total_time += time_delta
            elif isinstance(model, OVDecoder):
                if "past_key_values" in kwargs:
                    decoder_with_past_call_count += 1
                    decoder_with_past_total_time += time_delta
                else:
                    decoder_call_count += 1
                    decoder_total_time += time_delta
            else:
                raise Exception
            return result

        model.forward = wrapper

    wrap(encoder)
    wrap(decoder)
    wrap(decoder_with_past)


def collect_calibration_data(ov_decoder, num_calibration_samples, save_calibration_data):
    calibration_data_file_path = CALIBRATION_DATA_CACHE / f"{num_calibration_samples}.pkl"

    if calibration_data_file_path.exists():
        with open(calibration_data_file_path, 'rb') as f:
            calibration_data = pickle.load(f)
    else:
        calibration_data = []

        def wrap_for_data_collection():
            original_fn = ov_decoder.request.start_async

            def wrapper(*args, **kwargs):
                inputs = kwargs.get("inputs", args[0])
                if COLLECT_CALIBRATION_DATA:
                    calibration_data.append(inputs)
                return original_fn(*args, **kwargs)

            ov_decoder.request.start_async = wrapper

        wrap_for_data_collection()

        num_calibration_samples = min(num_calibration_samples, 755)
        dataset = datasets.load_dataset("jfleg", split=f"validation[:{num_calibration_samples}]")

        with calibration_data_collection():
            for data_item in tqdm(dataset, total=num_calibration_samples, desc="Collecting calibration data"):
                grammar_corrector_pipe(data_item["sentence"])

        if save_calibration_data:
            if not calibration_data_file_path.parent.exists():
                calibration_data_file_path.parent.mkdir()
            with open(calibration_data_file_path, 'wb') as f:
                pickle.dump(calibration_data, f)

    return calibration_data


@arg_logger
def compress(grammar_corrector_pipe, quantize, save_calibration_data=True,
             smooth_quant_alpha=0.5,
             num_calibration_samples=300,
             preset=nncf.QuantizationPreset.PERFORMANCE):
    if quantize:
        # model_path = Path(f"quantized_models/{preset.value}_{num_calibration_samples}_{smooth_quant_alpha:.2f}")
        model_path = Path(f"quantized_models/{preset.value}_{num_calibration_samples}_{smooth_quant_alpha:.2f}_max-q1e-4")
        # model_path = Path(f"quantized_models/{preset.value}_{num_calibration_samples}_{smooth_quant_alpha:.2f}_tmp")
    else:
        model_path = Path(f"compressed_model")

    # model_path = Path("quantized_models/old/performance_300_0.50")

    model_path = model_path / "openvino_model.xml"
    if model_path.exists():
        print(f"Loading existing model from {model_path}")
        compressed_model = core.read_model(model_path)
    else:
        ov_decoder: OVDecoder = grammar_corrector_pipe.model.decoder_with_past
        model: ov.Model = ov_decoder.model
        if not quantize:
            compressed_model = nncf.compress_weights(model)
        else:
            calibration_data = collect_calibration_data(ov_decoder, num_calibration_samples, save_calibration_data)
            # calibration_data[300] = None

            compressed_model = nncf.quantize(
                model,
                calibration_dataset=nncf.Dataset(calibration_data),
                preset=preset,
                subset_size=len(calibration_data),
                model_type=nncf.ModelType.TRANSFORMER,
                advanced_parameters=nncf.AdvancedQuantizationParameters(
                    smooth_quant_alpha=smooth_quant_alpha,
                    activations_range_estimator_params=RangeEstimatorParameters(
                        max=StatisticsCollectorParameters(StatisticsType.QUANTILE),
                    )
                ),
            )

        ov.serialize(compressed_model, model_path)

    grammar_corrector_pipe.model.decoder_with_past.model = compressed_model
    grammar_corrector_pipe.model.decoder_with_past.request = None
    grammar_corrector_pipe.model.decoder_with_past._compile()


def validate(grammar_corrector_pipe, verbose=True, return_per_sample=False, dataset=None):
    if dataset is None:
        dataset = datasets.load_dataset("jfleg", split="test[:50]")

    predictions = []
    ground_truths = []
    for data_item in tqdm(dataset, desc="Evaluation", disable=not verbose):
        corrected_text = correct_text(data_item["sentence"], grammar_checker_pipe, grammar_corrector_pipe,
                                      disable_tqdm=True)

        # print(data_item["sentence"])
        # print(grammar_corrector_pipe(data_item["sentence"])[0]["generated_text"])
        # print(corrected_text)
        # print()

        ground_truths.append(data_item["corrections"])
        predictions.append([corrected_text] * len(data_item["corrections"]))

    word_accuracy = 100 * (1 - wer.compute(references=sum(ground_truths, start=[]),
                                           predictions=sum(predictions, start=[])))
    if verbose:
        print(f"WER: {word_accuracy:.2f}")

    if return_per_sample:
        per_sample_metrics = []
        for gts, ps in zip(ground_truths, predictions):
            acc = 100 * (1 - wer.compute(references=gts, predictions=ps))
            per_sample_metrics.append(acc)
        return word_accuracy, per_sample_metrics
    return word_accuracy


def quantize_with_accuracy_control(grammar_corrector_pipe, num_calibration_samples):
    ov_decoder: OVDecoder = grammar_corrector_pipe.model.decoder_with_past
    model: ov.Model = ov_decoder.model

    calibration_data = collect_calibration_data(ov_decoder, num_calibration_samples, save_calibration_data=True)
    calibration_dataset = nncf.Dataset(calibration_data)
    validation_dataset = nncf.Dataset(datasets.load_dataset("jfleg", split=f"test[:50]"))

    def validate_fn(model, data):
        grammar_corrector_pipe.model.decoder_with_past.model = model
        grammar_corrector_pipe.model.decoder_with_past.request = None
        grammar_corrector_pipe.model.decoder_with_past._compile()
        return validate(grammar_corrector_pipe, verbose=False, return_per_sample=True, dataset=data)

    def prepare_model_for_inference(self, model):
        return model
    from nncf.quantization.algorithms.accuracy_control.evaluator import Evaluator
    Evaluator.prepare_model_for_inference = prepare_model_for_inference

    compressed_model = nncf.quantize_with_accuracy_control(model,
                                                           calibration_dataset,
                                                           validation_dataset,
                                                           validation_fn=validate_fn,
                                                           subset_size=len(calibration_data),
                                                           max_drop=-2,
                                                           preset=nncf.QuantizationPreset.PERFORMANCE,
                                                           model_type=nncf.ModelType.TRANSFORMER,
                                                           advanced_quantization_parameters=
                                                           nncf.AdvancedQuantizationParameters(smooth_quant_alpha=0.95),
                                                           advanced_accuracy_restorer_parameters=
                                                           AdvancedAccuracyRestorerParameters(tune_hyperparams=False))

    save_dir = Path("quantized_models/qwac")
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    ov.serialize(compressed_model, save_dir / "openvino_model.xml")

    grammar_corrector_pipe.model.decoder_with_past.model = compressed_model
    grammar_corrector_pipe.model.decoder_with_past.request = None
    grammar_corrector_pipe.model.decoder_with_past._compile()


default_text = "It's beeen a whille sinse I've went on a vacashun, and I'm realy exiceted abuot thiss trip. I've hurd " \
               "that the beech at this destenation is absolootly gorgeos. We've rentted a butiful cabben rite by the " \
               "oceen, and the veiw is brethtaking. Theres so meny fun activitees planed for this weeke, " \
               "like snorckling, parasaleeng, and bilding sandcastels. I cant wait to relax in the son with a coldd " \
               "piña colata.\n\nLast nite, we went out to a local restront for dinar. The menoo was quite exstensive, " \
               "and I orddered the lasanya, but wen it came, it was compleetely undarcoocked. I was disapointed, " \
               "but they quicklee replaced it with a new dish. The fud was delisious, and I was so ful afterword. We " \
               "also orddered a bottel of red wine to acompany our meal, and it was relly good.\n\nTommorrow, " \
               "we're planing to go to the mountins for a hike. I here the veiw from the peak is incredabel. I hoap " \
               "the wether is nice and we don't get cauhgt in any rain. We're bringing our dog along, and he loves " \
               "runing in the outdors.\n\nOveral, this vacashun is off to a grate start, despit the smal hiccups. I " \
               "cant wait to see what the rest of the weke has in store for us."

checker_tokenizer, grammar_checker_pipe = load_grammar_cheker()
corrector_tokenizer, grammar_corrector_pipe = load_grammar_corrector()

add_encoder_decoder_wrappers(grammar_corrector_pipe.model.encoder, grammar_corrector_pipe.model.decoder,
                             grammar_corrector_pipe.model.decoder_with_past)

# import numpy as np
# np.seterr(over='raise')
compress(grammar_corrector_pipe,
         quantize=bool(1),
         save_calibration_data=bool(1),
         smooth_quant_alpha=0.95,
         num_calibration_samples=10,
         preset=nncf.QuantizationPreset.PERFORMANCE,
)

# quantize_with_accuracy_control(grammar_corrector_pipe, 10)

corrected_text = correct_text(default_text, grammar_checker_pipe, grammar_corrector_pipe)
print(corrected_text)

if VERBOSE:
    print(f"Total corrected tokens: {total_corrected_tokens}")
    print(f"Total check time: {total_check_time}")
    print(f"Total correction time: {total_correction_time}")
    print(f"Encoder total calls: {encoder_call_count}; encoder total time: {encoder_total_time}")
    print(f"Decoder total calls: {decoder_call_count}; decoder total time: {decoder_total_time}")
    print(f"Decoder w/past total calls: {decoder_with_past_call_count}; decoder w/past total time: "
          f"{decoder_with_past_total_time}")

validate(grammar_corrector_pipe)

# Text length: 408 tokens
# Original time: 11.94 / 13.57
# Compressed time: 8.33 / 10.07 (1.43x / 1.27x)
# Quantized (bad) time: 6.77 / 8.33 (1.76x / 1.63x)
# Quantized (good) time: 7.87 / 9.54 (1.52x / 1.42x)
