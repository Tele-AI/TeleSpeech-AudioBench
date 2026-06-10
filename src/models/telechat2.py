import logging
import torch
from typing import Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.models.base import Model

logger = logging.getLogger(__name__)

class TeleChat2(Model):
    def __init__(self, path: str, sample_params: Dict[str, Any] = None):
        super().__init__(sample_params)
        logger.info("start load model from {}".format(path))
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            device_map="auto",
            trust_remote_code=True, 
            torch_dtype=torch.float16
        ).eval()
        logger.info("successfully load model from {}".format(path))

        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
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
        self.generation_config = config.get(self.sample_params.get("gen_type", "greedy"), None)
        logger.info("generation_config: {}".format(self.generation_config)) 

    
    def generate_once(self, audio, **kwargs):
        content = kwargs["query"]
        
        messages = [
            {"role": "system", "content": self.system_prompt},
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
        
