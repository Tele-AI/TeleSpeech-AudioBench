import logging
import re
import torch
from typing import Dict, Any
import soundfile as sf
from src.models.base import Model

logger = logging.getLogger(__name__)
try:
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    from qwen_omni_utils import process_mm_info
except:
    logger.info(f"Need transformers version >= 4.51.3")

class Qwen2Omni(Model):
    def __init__(self, path: str, sample_params: Dict[str, Any] = None):
        super().__init__(sample_params)
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        # self.model.disable_talker()
        self.system_prompt = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
        self.processor = Qwen2_5OmniProcessor.from_pretrained(path)
        self.REGEX_HEAD = re.compile(r".*assistant\n", re.DOTALL | re.IGNORECASE)
        
        config = {
            "greedy": {
                "do_sample": False,
                "max_new_tokens": 1024,
                "top_k": None,
                "num_beams": 1,
                "temperature": None,
                "top_p": None,
                "pad_token_id": self.processor.tokenizer.pad_token_id
            },
            "default":{}
        }
        self.generation_config = config.get(self.sample_params.get("gen_type", "greedy"), None)
        logger.info("generation_config: {}".format(self.generation_config))


    def _generate(self, conversation, output_audio_path):
        return_audio = output_audio_path is not None
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)
        inputs = self.processor(
            text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=True
        )
        inputs = inputs.to(self.model.device).to(self.model.dtype)

        outputs = self.model.generate(
            **inputs, 
            use_audio_in_video=True,
            return_audio=return_audio, 
            speaker="Chelsie",
            **self.generation_config
        ) # speaker="Chelsie" or "Ethan"

        if return_audio:
            response = self.processor.batch_decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]  # batch
            audio = outputs[1]
            sf.write(output_audio_path, audio.reshape(-1).detach().cpu().numpy(), samplerate=24000)
        else:
            response = self.processor.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]  # batch
        pred = self.REGEX_HEAD.sub("", response)
        
        return pred

    def generate_once(self, audio, **kwargs):
        output_audio_path = kwargs.get("pred_audio", None)
        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": self.system_prompt}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio},
                ],
            },
        ]
        pred = self._generate(conversation, output_audio_path)
        return {"pred": pred, "pred_audio": kwargs.get("pred_audio")}

    def generate_multiturn(self, audio, user_history, assistant_history, **kwargs):
        instruction = kwargs.get("instruct", "")
        output_audio_path = kwargs.get("pred_audio", None)
        conversation = [
           {
                "role": "system",
                "content": [
                    {
                        "type": "text", 
                        "text": self.system_prompt
                    }
                ],
            },
        ]
        for uh, ah in zip(user_history, assistant_history):
            conversation.append({"role": "user", "content":  [{"type": "audio", "audio": uh}]})
            conversation.append({"role": "assistant", "content": ah})
        conversation.append({"role": "user", "content":  [{"type": "audio", "audio": audio}]})
        pred = self._generate(conversation, output_audio_path)

        return {"pred": pred, "pred_audio": kwargs.get("pred_audio")}