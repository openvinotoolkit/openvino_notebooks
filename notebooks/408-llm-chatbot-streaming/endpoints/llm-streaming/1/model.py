#
# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import json
import threading
import numpy as np

import triton_python_backend_utils as pb_utils

from optimum.intel.openvino import OVModelForCausalLM

from transformers import AutoTokenizer, AutoConfig, TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList

MODEL_HF_NAME = "togethercomputer/RedPajama-INCITE-Chat-3B-v1"
MODEL_PATH = "/model" 

OV_CONFIG = {'PERFORMANCE_HINT': 'LATENCY', 'NUM_STREAMS': '1'}

class StopOnTokens(StoppingCriteria):
    def __init__(self, token_ids):
        self.token_ids = token_ids
    def __call__(self, input_ids, scores, **kwargs) -> bool:
        for stop_id in self.token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])

        # OpenVINO
        print('Loading OV model...')
        ov_config = {'PERFORMANCE_HINT': 'LATENCY', 'NUM_STREAMS': '1'}
        self.ov_model = OVModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device="CPU",
            ov_config=OV_CONFIG,
            config=AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True),
            trust_remote_code=True)
        print('OV model loaded')

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_HF_NAME, trust_remote_code=True);

    def execute(self, requests):
        if len(requests) != 1:
            raise pb_utils.TritonModelException(
                "unsupported batch size " + len(requests)
            )
        request = requests[0]
        responses = None

        text = pb_utils.get_input_tensor_by_name(request, "text")
        thread = threading.Thread(
            target=self.response_thread,
            args=(request.get_response_sender(), text),
        )

        thread.daemon = True
        thread.start()

        return responses

    def response_thread(self, response_sender, text):
        text_string = text.as_numpy()[0].decode('utf-8', 'ignore')
        try:
            print(f"Received request string: {text_string}")
        except:
            print('Cannot decode received request string, will tokenize anyway')
        input_ids = self.tokenizer([text_string], return_tensors="pt", add_special_tokens=False).input_ids

        streamer = TextIteratorStreamer(self.tokenizer, timeout=30.0, skip_prompt=True, skip_special_tokens=False)
        generate_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=1024,
            temperature=0.1,
            do_sample=True,
            top_p=1.0,
            top_k=50,
            repetition_penalty=1.1,
            streamer=streamer,
            stopping_criteria=StoppingCriteriaList([
                StopOnTokens([29, 0])  # TODO: Red-pajama specific criteria
            ])
        )

        def generate():
            print('generate_and_signal_complete start')
            self.ov_model.generate(**generate_kwargs)
            print('generate_and_signal_complete generate end')

        t1 = threading.Thread(target=generate)
        t1.start()

        print("Generating response...")
        for new_text in streamer:
            try:
                print(new_text, flush=True, end='')
            except UnicodeEncodeError:
                pass
            out_t = pb_utils.Tensor("partial_response", np.array([new_text], dtype=np.object_))
            response = pb_utils.InferenceResponse(output_tensors=[out_t])
            response_sender.send(response)

        response_sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
        print('TRITONSERVER_RESPONSE_COMPLETE_FINAL sent')

    def finalize(self):
        print("Cleaning up...")
