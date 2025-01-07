import shutil
import time
import os
from nncf import compress_weights, Dataset
from nncf.parameters import CompressWeightsMode, SensitivityMetric
import openvino as ov
from pathlib import Path
import gc


from data_preprocessing import prepare_dataset_llm
from ov_mllama_helper import LANGUAGE_MODEL


def compression_widgets_helper():
    from ipywidgets import Checkbox, VBox, RadioButtons, Accordion

    radio_button = RadioButtons(options=["data-free", "data-aware"], value="data-aware")
    data_aware_settings = Accordion()

    awq = Checkbox(description="AWQ", value=True)
    scale_estimation = Checkbox(description="Scale Estimation", value=True)
    gptq = Checkbox(description="GPTQ", value=False)
    lora = Checkbox(description="LoRA correction", value=False)
    data_aware_approaches = VBox(children=[awq, scale_estimation, gptq, lora])
    data_aware_settings = Accordion(children=[data_aware_approaches], titles=["Data-Aware settings"])

    vb = VBox(children=[radio_button, data_aware_settings])

    compression_settings = {"awq": awq, "gptq": gptq, "lora": lora, "scale_estimation": scale_estimation}

    def remove_data_aware_settings(button):
        if radio_button.value == "data-free":
            vb.children = [radio_button]
            awq.value = False
            scale_estimation.value = False
            gptq.value = False
            lora.value = False
        else:
            vb.children = [radio_button, data_aware_settings]
            awq.value = True
            scale_estimation.value = True
            gptq.value = False
            lora.value = False

    radio_button.observe(remove_data_aware_settings, names="value")
    return vb, compression_settings


def vision_encoder_selection_widget(device):
    from ipywidgets import Dropdown

    optimizations = ["FP16", "INT8 quantization", "INT8 weights compression"]

    options = Dropdown(
        description="Vision Encoder", options=optimizations, value=optimizations[0] if "GPU" in device else optimizations[1], disabled="GPU" in device
    )

    return options


core = ov.Core()


def compress(
    model_dir: Path,
    algo=CompressWeightsMode.INT4_ASYM,
    ratio=1.0,
    sm=SensitivityMetric.MAX_ACTIVATION_VARIANCE,
    awq=True,
    scale_estimation=True,
    lora=False,
    gptq=False,
    group_size=64,
    all_layers=True,
    dataset_size=64,
    force=False,
):
    postfix = f"llm_{algo}_r{ratio}_gs{group_size}_{sm}"
    if awq:
        postfix += "_awq"
    if scale_estimation:
        postfix += "_scale"
    if gptq:
        postfix += "_gptq"
    if lora:
        postfix += "_lora"
    if all_layers:
        postfix += "_all_layers"

    postfix = postfix.replace(".", "")
    dst_name = postfix + ".xml"
    saving_path = Path(model_dir) / dst_name

    if saving_path.exists():
        if force:
            shutil.rmtree(saving_path)
            shutil.rmtree(saving_path.with_suffix(".bin"))
        else:
            print(f"✅ Compressed model already exists and can be found in {saving_path}")
            return saving_path
    nncf_dataset = None
    if awq or lora or gptq or scale_estimation:
        print("⌛ Dataset preparation started")
        dataset = prepare_dataset_llm(model_dir, dataset_size)
        nncf_dataset = Dataset(dataset)
        gc.collect()
        print("✅ Dataset preparation finished")

    print("⌛ Model compression started")
    print(
        f"Compression parameters:\n\t\n\talgorithm {algo}\n\tgroup size - {group_size}\n\tratio - {ratio}\n\tawq - {awq}\n\tscale estimation - {scale_estimation}\n\tlora correction - {lora}\n\tgptq - {gptq}\n\tall_layers - {all_layers}"
    )
    lm_model = core.read_model(model_dir / LANGUAGE_MODEL)

    lm_model = compress_weights(
        lm_model,
        mode=algo,
        group_size=group_size,
        ratio=ratio,
        dataset=nncf_dataset,
        sensitivity_metric=sm,
        awq=awq,
        scale_estimation=scale_estimation,
        lora_correction=lora,
        gptq=gptq,
        all_layers=all_layers,
    )
    ov.save_model(lm_model, saving_path)
    del lm_model
    gc.collect()

    print(f"✅ Model compression finished. Compressed model can be found in {saving_path}")

    return saving_path
