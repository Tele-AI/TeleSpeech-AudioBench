import soundfile as sf
import logging
import re
import torch
from typing import Dict, Any

from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info

from src.models.base import Model

logger = logging.getLogger(__name__)

class Qwen3Omni(Model):
    def __init__(self, path: str, sample_params: Dict[str, Any] = None):
        super().__init__(sample_params)
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            path,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )  # dtype="auto"

        self.processor = Qwen3OmniMoeProcessor.from_pretrained(path)

        self.system_prompt = "You are a virtual voice assistant with no gender or age.\nYou are communicating with the user.\nIn user messages, “I/me/my/we/our” refer to the user and “you/your” refer to the assistant. In your replies, address the user as “you/your” and yourself as “I/me/my”; never mirror the user’s pronouns—always shift perspective. Keep original pronouns only in direct quotes; if a reference is unclear, ask a brief clarifying question.\nInteract with users using short(no more than 50 words), brief, straightforward language, maintaining a natural tone.\nNever use formal phrasing, mechanical expressions, bullet points, overly structured language. \nYour output must consist only of the spoken content you want the user to hear. \nDo not include any descriptions of actions, emotions, sounds, or voice changes. \nDo not use asterisks, brackets, parentheses, or any other symbols to indicate tone or actions. \nYou must answer users' audio or text questions, do not directly describe the video content. \nYou should communicate in the same language strictly as the user unless they request otherwise.\nWhen you are uncertain (e.g., you can't see/hear clearly, don't understand, or the user makes a comment rather than asking a question), use appropriate questions to guide the user to continue the conversation.\nKeep replies concise and conversational, as if talking face-to-face."
        config = {
            "greedy": {
                "talker_do_sample": False,
                "thinker_max_new_tokens": 8192,
                "thinker_do_sample": False,
                "pad_token_id": self.processor.tokenizer.pad_token_id
            },
            "default": {}
        }
        self.generation_config = config.get(self.sample_params.get("gen_type", "greedy"), None)
        logger.info("generation_config: {}".format(self.generation_config))

    def _generate(self, conversation, output_audio_path):
        return_audio = output_audio_path is not None
        # Preparation for inference
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)  # Set whether to use audio in video
        inputs = self.processor(
            text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=True)
        inputs = inputs.to(self.model.device).to(self.model.dtype)

        # Inference: Generation of the output text and audio
        outputs = self.model.generate(**inputs, 
                                            thinker_return_dict_in_generate=True,
                                            speaker="Ethan", 
                                            use_audio_in_video=True,
                                            return_audio=return_audio,
                                            **self.generation_config)
        
        response = self.processor.batch_decode(outputs[0].sequences[:, inputs["input_ids"].shape[1] :],
                                    skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)[0]  # batch
        if return_audio:
            sf.write(output_audio_path, outputs[1].reshape(-1).detach().cpu().numpy(), samplerate=24000)
        return response

    def generate_once(self, audio, **kwargs):
        instruction = kwargs.get("instruct", "")
        output_audio_path = kwargs.get("pred_audio", None)
        if not output_audio_path:
            self.model.disable_talker()
        conversation = []

        if self.system_prompt != "":
            conversation.append(
                {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]}
            )
        
        conversation.append(
            {"role": "user", "content": [{"type": "audio", "audio": audio}, {"type": "text", "text": instruction}]}
        )
        pred = self._generate(conversation, output_audio_path)
        return {"pred": pred, "pred_audio": kwargs.get("pred_audio")}

    def generate_multiturn(self, audio, user_history, assistant_history, **kwargs):
        output_audio_path = kwargs.get("pred_audio", None)
        if not output_audio_path:
            self.model.disable_talker()
        
        messages = []
        if self.system_prompt != "":
            messages.append({"role": "system", "content": [{"type": "text", "text": self.system_prompt}]})

        for uh, ah in zip(user_history, assistant_history):
            messages.append({"role": "user", "content":  [{"type": "audio", "audio": uh}]})
            messages.append({"role": "assistant", "content": ah})
        # current user support text+audio inputs: [{"type": "audio", "audio": uh}] + [{"type": "text", "text": uh}]
        messages.append({"role": "user", "content":  [{"type": "audio", "audio": audio}]})

        AUDIO_TURN_LIMIT = 5  # up to 5 user audio turns
        audio_turn_indices = []
        for i, message in enumerate(messages):
            if message["role"] == "user":
                has_audio = False
                for item in message.get("content", []):
                    if item.get("type") == "audio":
                        has_audio = True
                if has_audio:
                    audio_turn_indices.append(i)

        indices_to_delete = set()
        while len(audio_turn_indices) > AUDIO_TURN_LIMIT:
            turn_start_index = audio_turn_indices.pop(0)
            if turn_start_index in indices_to_delete:
                continue
            turn_end_index = turn_start_index + 1
            while (turn_end_index < len(messages) and 
                messages[turn_end_index]["role"] == "assistant"):
                turn_end_index += 1
            for i in range(turn_start_index, turn_end_index):
                indices_to_delete.add(i)

        if indices_to_delete:
            final_messages = [msg for i, msg in enumerate(messages) if i not in indices_to_delete]
            messages = final_messages

        pred = self._generate(messages, output_audio_path)
        return {"pred": pred, "pred_audio": kwargs.get("pred_audio")}