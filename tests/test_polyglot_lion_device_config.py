import ast
import json
import unittest
from itertools import product
from pathlib import Path
from types import SimpleNamespace

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/polyglot-lion/polyglot-lion.ipynb"


def notebook_cell(marker):
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code" and marker in "".join(cell["source"])]
    if len(cells) != 1:
        raise AssertionError(f"Expected one code cell containing {marker!r}, found {len(cells)}")
    return cells[0]


class PolyglotDeviceConfigTest(unittest.TestCase):
    def test_device_configs_apply_before_compile_and_match_evaluation_metadata(self):
        source = notebook_cell("def load_task_model(")
        function = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "load_task_model")
        loader_module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
        evaluation = compile(notebook_cell("numerical_rows.append("), str(NOTEBOOK), "exec")
        override = {"decoder": {"INFERENCE_PRECISION_HINT": "f32"}}

        cases = (
            ("CPU", {"CPU": {}, "GPU": override}, {}),
            ("GPU", {"CPU": {}, "GPU": override}, override),
            ("GPU.0", {"CPU": {}, "GPU": override}, override),
            ("GPU.1", {"CPU": {}, "GPU": override}, override),
            ("GPU", {"CPU": {}}, {}),
            ("GPU.0", {"CPU": {}}, {}),
            ("GPU.1", {"CPU": {}}, {}),
        )
        for (device, configs, expected), diagnostics in product(cases, (False, True)):
            with self.subTest(device=device, configs=configs, diagnostics=diagnostics):
                loaded = []

                class Model:
                    def __init__(self):
                        self.decoder = SimpleNamespace(ov_config={"existing": "value"})
                        self.compiled = False

                    @classmethod
                    def from_pretrained(cls, folder, **kwargs):
                        self = cls()
                        self.device = kwargs["device"]
                        self.compiled_on_load = kwargs["compile"]
                        loaded.append(self)
                        return self

                    def compile(self):
                        self.compiled = True
                        self.compile_config = dict(self.decoder.ov_config)

                loader_globals = {"Path": Path, "OVModelForSpeechSeq2Seq": Model, "configure_audio_encoder": lambda model: None}
                exec(compile(loader_module, str(NOTEBOOK), "exec"), loader_globals)
                loader = loader_globals["load_task_model"]
                result = loader(Path("model"), device=device, ov_config={}, component_configs=configs)
                self.assertEqual(result.device, device)
                self.assertFalse(result.compiled_on_load)
                self.assertTrue(result.compiled)
                self.assertEqual(result.compile_config, {"existing": "value", **expected.get("decoder", {})})

                logits = SimpleNamespace(float=lambda: SimpleNamespace(numpy=lambda: [1.0]))
                namespace = {
                    "inference_device": SimpleNamespace(value=device),
                    "variants": {"FP16": Path("model")},
                    "load_task_model": loader,
                    "CONFIG": {},
                    "COMPONENT_CONFIGS": configs,
                    "transcribe": lambda *args, **kwargs: {"complete": True},
                    "processor": object(),
                    "prepared": [({}, {})],
                    "MAX_NEW_TOKENS": 64,
                    "RUN_F32_DIAGNOSTICS": diagnostics,
                    "generate": lambda *args, **kwargs: SimpleNamespace(logits=[logits]),
                    "logit_inputs": {},
                    "logit_metrics": lambda **kwargs: {},
                    "pt_logits": [1.0],
                }
                exec(evaluation, namespace)
                self.assertEqual(namespace["numerical_rows"][0]["component_config"], expected)
                for model in loaded:
                    self.assertEqual(model.device, device)
                    self.assertEqual(model.compile_config, {"existing": "value", **expected.get("decoder", {})})


if __name__ == "__main__":
    unittest.main()
