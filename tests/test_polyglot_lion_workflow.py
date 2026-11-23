import json
import os
import shutil
import tempfile
import unittest
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from utils.cmd_helper import optimum_cli

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/polyglot-lion/polyglot-lion.ipynb"


def notebook_cell(marker):
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code" and marker in "".join(cell["source"])]
    if len(cells) != 1:
        raise AssertionError(f"Expected one code cell containing {marker!r}, found {len(cells)}")
    return compile(cells[0], str(NOTEBOOK), "exec")


class PolyglotWorkflowTest(unittest.TestCase):
    @unittest.skipIf(os.name == "nt", "The fake CLI is a POSIX executable")
    def test_shared_cli_runs_notebook_environment_executable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scripts = root / "venv with spaces"
            scripts.mkdir()
            executable = scripts / "optimum-cli"
            executable.write_text(
                "#!/usr/bin/python3\n"
                "import json, sys\n"
                "from pathlib import Path\n"
                "output = Path(sys.argv[5])\n"
                "output.mkdir()\n"
                "(output / 'args.json').write_text(json.dumps(sys.argv[1:]))\n"
            )
            executable.chmod(0o755)
            fp16 = root / "model with spaces"
            context = {
                "MODEL_DIR": root,
                "FP16": fp16,
                "IR_FILES": (),
                "MODEL_ID": "model",
                "MODEL_REVISION": "revision",
                "snapshot_download": lambda **kwargs: "checkpoint with spaces",
                "optimum_cli": partial(optimum_cli, show_command=False),
                "sys": SimpleNamespace(executable=str(scripts / "python")),
                "shutil": shutil,
                "os": os,
                "Path": Path,
            }
            with patch.dict(os.environ, {"PATH": "/usr/bin"}):
                exec(notebook_cell("fresh_export = False"), context)
            self.assertEqual(
                json.loads((fp16 / "args.json").read_text()),
                [
                    "export",
                    "openvino",
                    "--model",
                    "checkpoint with spaces",
                    str(fp16),
                    "--task",
                    "automatic-speech-recognition-with-past",
                    "--trust-remote-code",
                    "--weight-format",
                    "fp16",
                ],
            )

    def test_accuracy_gate_requires_complete_fp16_baseline(self):
        complete = {"transcription": "test", "complete": True}
        incomplete = {"transcription": "test", "complete": False}
        code = notebook_cell("accuracy_rows = []")
        for baseline, expected in ((complete, True), (incomplete, False)):
            with self.subTest(baseline_complete=baseline["complete"]):
                context = {
                    "results": {"FP16 GPU.0": [baseline], "INT8 weights GPU.0": [complete]},
                    "pt_results": [complete],
                    "configs": ("en_us",),
                    "samples": [{"config": "en_us", "reference": "test"}],
                    "LANGUAGES": {"en_us": "English"},
                    "ERROR_BUDGET": 0.02,
                    "corpus_error": lambda **kwargs: 0.0,
                    "pd": SimpleNamespace(DataFrame=lambda rows: rows),
                    "display": lambda value: None,
                    "numerical_rows": [],
                }
                exec(code, context)
                self.assertEqual(context["accuracy_rows"][1]["local_slice_pass"], expected)

    def test_export_rejects_unproven_existing_decoder(self):
        export_cell = notebook_cell("fresh_export = False")
        encoder_cell = notebook_cell('export_record = FP16 / "export.json"')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fp16 = root / "fp16"
            files = ("config.json", "generation_config.json", "openvino_encoder_model.xml", "openvino_decoder_model.xml")
            calls = []

            def exporter(checkpoint, output, additional_args):
                calls.append((checkpoint, output, additional_args))
                output.mkdir()
                for name in files:
                    (output / name).write_text("test")

            context = {
                "MODEL_DIR": root,
                "FP16": fp16,
                "Path": Path,
                "IR_FILES": files,
                "MODEL_ID": "model",
                "MODEL_REVISION": "revision",
                "snapshot_download": lambda **kwargs: "checkpoint",
                "optimum_cli": exporter,
                "sys": SimpleNamespace(executable=str(root / "venv" / "bin" / "python")),
                "shutil": SimpleNamespace(which=lambda name, path: str(Path(path) / name)),
                "os": SimpleNamespace(environ={"PATH": "/usr/bin"}, pathsep=":"),
                "versions": {"optimum-intel": "1", "openvino": "1", "qwen-asr": "1"},
                "AUDIO_ENCODER_FIX": "test",
                "pt_model": SimpleNamespace(thinker=SimpleNamespace(audio_tower=SimpleNamespace(config=SimpleNamespace(_attn_implementation="eager")))),
                "export_audio_encoder": lambda encoder, path: calls.append(("encoder", path)),
                "model_fingerprint": lambda folder: {name: (folder / name).read_text() for name in files},
                "json": json,
            }
            exec(export_cell, context)
            self.assertTrue(context["fresh_export"])
            self.assertEqual(context["os"].environ["PATH"], f"{root / 'venv' / 'bin'}:/usr/bin")
            self.assertEqual(calls[0][2], {"task": "automatic-speech-recognition-with-past", "trust-remote-code": None, "weight-format": "fp16"})
            exec(encoder_cell, context)
            self.assertEqual(context["variants"], {"FP16": fp16})

            exec(export_cell, context)
            self.assertFalse(context["fresh_export"])
            exec(encoder_cell, context)
            self.assertEqual(len(calls), 2)

            (fp16 / "export.json").unlink()
            with self.assertRaisesRegex(ValueError, "provenance is missing"):
                exec(encoder_cell, context)
            self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
