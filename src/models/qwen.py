import logging
from typing import Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.models.base import Model

logger = logging.getLogger(__name__)

class Qwen2Instruct(Model):
    def __init__(self, path: str, sample_params: Dict[str, Any] = None):
        super().__init__(sample_params)
        logger.info("start load model from {}".format(path))
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype="auto",
            device_map="auto",
        ).eval()
        logger.info("successfully load model from {}".format(path))

        self.tokenizer = AutoTokenizer.from_pretrained(path)
        config = {
            "greedy": {
                "do_sample": False,
                "max_new_tokens": 1024,
                "top_k": None,
                "num_beams": 1,
                "temperature": None,
                "top_p": None
            }
        }
        self.think = self.sample_params.get("think", False)
        self.generation_config = config.get(self.sample_params.get("gen_type", "greedy"), None)
        logger.info("generation_config: {}".format(self.generation_config))
        logger.info("using think: {}".format(self.think))

        self.system_prompt_qwen2 = "You are a helpful assistant."
        self.system_prompt_qwen2d5 = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

    def generate_once(self, audio, **kwargs):
        system_prompt = self.system_prompt_qwen2d5
        content = kwargs.get("instruct", "") + kwargs["query"]
            
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            **self.generation_config
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return {"pred": response}

class Qwen3Instruct(Qwen2Instruct):
    def __init__(self, path: str, sample_params: Dict[str, Any] = None):
        # transformers>=4.51.0
        super().__init__(path, sample_params)
        self.system_prompt = None

    def generate_once(self, audio, **kwargs):
        content = kwargs.get("instruct", "") + kwargs["query"]
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": content}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.think
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            **self.generation_config
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
        thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        response = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        return {"pred": response}