import logging
from typing import Dict, Any

from src.models.base import Model
from src.models.src_stepaudio2.stepaudio2 import StepAudio2 as StepAudio2Model
from src.models.src_stepaudio2.token2wav import Token2wav

logger = logging.getLogger(__name__)

class StepAudio2(Model):
    def __init__(self, path: str, token2wav_path: str, sample_params: Dict[str, Any] = None):
        super().__init__(sample_params)

        self.model = StepAudio2Model(path)
        self.token2wav = Token2wav(token2wav_path)

        config = {
            "default": {
                "do_sample": True,
                "temperature": 0.7,
                "max_new_tokens": 2048,
                "repetition_penalty": 1.05, 
                "top_p": 0.9
            },
            "greedy": {
                "do_sample": False,
                "max_new_tokens": 1024,
                "top_k": None,
                "num_beams": 1,
                "temperature": None,
                "top_p": None,
                "repetition_penalty": 1.05,  # TTTdas: If set to 1.0, many of the generated audio outputs will be empty
            }
        }
        # self.sys_prompt = "你的名字叫做小跃，是由阶跃星辰公司训练出来的语音大模型。\n你情感细腻，观察能力强，擅长分析用户的内容，并作出善解人意的回复，说话的过程中时刻注意用户的感受，富有同理心，提供多样的情绪价值。\n今天是2025年10月28日，星期二\n请用默认女声与用户交流。"
        self.system_prompt = "You are a helpful assistant."  # stable prompt
        self.generation_config = config.get(self.sample_params.get("gen_type", "greedy"), None)
        logger.info("generation_config: {}".format(self.generation_config))

    def _generate(self, messages, output_audio_path):
        tokens, text, audio = self.model(messages, **self.generation_config)

        if output_audio_path:
            audio = [x for x in audio if x < 6561] # remove audio padding
            audio = self.token2wav(audio, prompt_wav="src/models/src_stepaudio2/assets/default_female.wav")
            with open(output_audio_path, "wb") as f:
                f.write(audio)

        return tokens, text, audio

    def generate_once(self, audio, **kwargs):
        if kwargs.get("pred_audio"):
            # speech-to-speech
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "human", "content": [{"type": "audio", "audio": audio}]},
                {"role": "assistant", "content": "<tts_start>", "eot": False}
            ]
        else:
            # speech-to-text
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "human", "content": [{"type": "audio", "audio": audio}]},
                {"role": "assistant", "content": None}
            ]
        
        tokens, text, audio = self._generate(messages, kwargs.get("pred_audio"))
        return {"pred": text, "pred_audio": kwargs.get("pred_audio")}

    def generate_multiturn(self, audio, user_history, assistant_history, **kwargs):
        messages = [{"role": "system", "content": self.system_prompt}]
        if len(user_history) > 0:
            for inp_audio, history_tokens in zip(user_history, kwargs["cache"]):
                messages.append({"role": "human", "content": [{"type": "audio", "audio": inp_audio}]})
                messages.append(
                    {
                        "role": "assistant",
                        "content":[
                            {"type": "text", "text":"<tts_start>"},
                            {"type":"token", "token": history_tokens}
                        ]
                    }
                )
        messages.append({"role": "human", "content": [{"type": "audio", "audio": audio}]})
        messages.append({"role": "assistant", "content": "<tts_start>", "eot": False})

        tokens, text, audio = self._generate(messages, kwargs.get("pred_audio"))

        return {"pred": text, "pred_audio": kwargs.get("pred_audio"), "cache": [tokens]}