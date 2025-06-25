# Copyright 2022-2025 XProbe Inc.
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
import importlib.util
import logging
import torch
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .....core.model import register_batching_multimodal_models
from .....core.scheduler import InferenceRequest
from .....model.utils import select_device
from .....types import PytorchModelConfig
from ...llm_family import LLMFamilyV1, LLMSpecV1, register_transformer
from ..core import register_non_default_model
from .core import PytorchMultiModalModel

logger = logging.getLogger(__name__)


@register_transformer
@register_non_default_model(
    "mistral-small-3.1-instruct"
)
class Mistral3ChatModel(PytorchMultiModalModel):
    def _sanitize_model_config(
        self, pytorch_model_config: Optional[PytorchModelConfig]
    ) -> PytorchModelConfig:
        pytorch_model_config = super()._sanitize_model_config(pytorch_model_config)
        assert pytorch_model_config is not None
        pytorch_model_config.setdefault("min_pixels", 256 * 28 * 28)
        pytorch_model_config.setdefault("max_pixels", 1280 * 28 * 28)
        return pytorch_model_config

    @classmethod
    def match_json(
        cls, model_family: "LLMFamilyV1", model_spec: "LLMSpecV1", quantization: str
    ) -> bool:
        if model_spec.model_format not in ["pytorch", "gptq", "awq"]:
            return False
        llm_family = model_family.model_family or model_family.model_name
        if "mistral-small-3.1-instruct".lower() in llm_family.lower():
            return True
        return False

    def decide_device(self):
        device = self._pytorch_model_config.get("device", "auto")
        device = select_device(device)
        # for multiple GPU, set back to auto to make multiple devices work
        self._device = device

    def load_processor(self):
        from transformers import AutoProcessor

        min_pixels = self._pytorch_model_config.get("min_pixels")
        max_pixels = self._pytorch_model_config.get("max_pixels")
        self._processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        self._tokenizer = self._processor.tokenizer

    def load_multimodal_model(self):
        from transformers import AutoModelForImageTextToText
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_path, 
            device_map="auto",
            attn_implementation="flash_attention_2",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).eval()

    def build_inputs_from_messages(
        self,
        messages: List[Dict],
        generate_config: Dict,
    ):
        inputs = self._processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(self._model.device, dtype=torch.bfloat16)

        return inputs

    def build_generate_kwargs(self, generate_config: Dict) -> Dict[str, Any]:
        max_new_tokens = generate_config.get("max_tokens", 1000)
        temperature = generate_config.get("temperature", 1)
        bos_token_id = generate_config.get("bos_token_id", 1)
        eos_token_id = generate_config.get("eos_token_id", 2)
        return {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "bos_token_id": bos_token_id,
            "eos_token_id": eos_token_id,
        }

    def build_streaming_iter(
        self,
        messages: List[Dict],
        generate_config: Dict,
    ) -> Tuple[Iterator, int]:
        from threading import Thread

        from transformers import TextIteratorStreamer

        tokenizer = self._tokenizer
        streamer = TextIteratorStreamer(
            tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True
        )

        inputs = self.build_inputs_from_messages(messages, generate_config)
        config = self.build_generate_kwargs(generate_config)

        def model_generate():
            try:
                input_len = inputs["input_ids"].shape[-1]
                with torch.inference_mode():
                    print("input_ids shape:", inputs["input_ids"].shape)
                    print("max_new_tokens:", config.get("max_new_tokens"))
                    print("device:", self._model.device)
                    generation = self._model.generate(**inputs, do_sample=False,**config, streamer=streamer)
                    generation = generation[0][input_len:]
                    return generation
            except Exception:
                streamer.end()
                raise

        thread = Thread(target=model_generate)
        thread.start()
        return streamer, len(inputs.input_ids[0])

    def prepare_sanitize_generate_config(self, req: InferenceRequest):
        """
        This file corresponds to multiple models,
        so the corresponding configuration is read directly through the transformers interface.
        """
        from transformers import GenerationConfig

        gen_config = GenerationConfig.from_pretrained(self.model_path).to_dict()
        raw_config = req.inference_kwargs.get("raw_params", {})
        gen_config.update(raw_config)
        return gen_config

    def _get_full_prompt(self, messages: List[Dict], tools, generate_config: dict):
        return self._transform_messages(messages)

    def build_prefill_kwargs(self, prompts: List, req_list: List[InferenceRequest]):
        import torch

        inputs = self._processor.apply_chat_template(
            prompts, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(self._model.device, dtype=torch.bfloat16)
        return inputs
