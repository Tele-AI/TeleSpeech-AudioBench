import logging
import os
import soundfile as sf
from typing import Dict, Any

from src.models.base import Model
from src.models.src_kimi.kimia_infer.api.kimia import KimiAudio

logger = logging.getLogger(__name__)

class Kimi(Model):
    def __init__(self, path: str, whisper_path: str, glm4_tokenizer: str, sample_params: Dict[str, Any] = None):
        super().__init__(sample_params)
        self.model = KimiAudio(
            model_path=path,
            whisper_path=whisper_path,
            glm4_tokenizer=glm4_tokenizer,
            load_detokenizer=True,
            split_device=False,  # split need 4.48.3
        )

        config = {
            "default": {
                "audio_temperature": 0.8,
                "audio_top_k": 10,
                "text_temperature": 0.0,
                "text_top_k": 5,
                "audio_repetition_penalty": 1.0,
                "audio_repetition_window_size": 64,
                "text_repetition_penalty": 1.0,
                "text_repetition_window_size": 16
            },
            "greedy": {
                "audio_temperature": 1e-7,
                "text_temperature": 1e-7,
                "audio_repetition_penalty": 1.0,
                "text_repetition_penalty": 1.0
            }  # NOTE (TTTdas): temerature > 1e-6 will do sampling
        }
        self.generation_config = config.get(self.sample_params.get("gen_type", "greedy"), None)
        logger.info("generation_config: {}".format(self.generation_config))

    def generate_once(self, audio, **kwargs):
        messages = []
        instruction = kwargs.get("instruct", "")
        if len(instruction) > 0:
            messages.append({"role": "user", "message_type": "text", "content": instruction})

        messages.append({"role": "user", "message_type": "audio", "content": audio})
        wav, text = self.model.generate(messages, **self.generation_config, output_type="both")
        if kwargs.get("pred_audio"):
            sf.write(
                kwargs["pred_audio"],
                wav.detach().cpu().view(-1).numpy(),
                24000,
            )
   
        return {"pred": text, "pred_audio": kwargs.get("pred_audio")}
    
    def generate_multiturn(self, audio, user_history, assistant_history, **kwargs):
        messages = []
        assistant_his_audio = kwargs.get("assistant_his_audio", [])
        for uh, ah, aha in zip(user_history, assistant_history, assistant_his_audio):
            messages.append({"role": "user", "message_type": "audio", "content": uh})
            messages.append({"role": "assistant", "message_type": "audio-text", "content": [aha, ah]})
        messages.append({"role": "user", "message_type": "audio", "content": audio})

        wav, text = self.model.generate(messages, **self.generation_config, output_type="both")
        if kwargs.get("pred_audio"):
            sf.write(
                kwargs["pred_audio"],
                wav.detach().cpu().view(-1).numpy(),
                24000,
            )
   
        return {"pred": text, "pred_audio": kwargs.get("pred_audio")}