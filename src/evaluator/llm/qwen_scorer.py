import logging
import os
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from src.evaluator.llm.llm_offline import LLMOfflineScorer

logger = logging.getLogger(__name__)

class QwenScorer(LLMOfflineScorer):
    def __init__(self, llm_name: str, judge_task: str, generate_params: dict, builder_name=None, parser_name=None):
        super().__init__(llm_name, judge_task, generate_params, builder_name, parser_name)

        sampling_params = {k: v for k, v in generate_params.items() if k not in ["ngpus"]}
        self.sampling_params = SamplingParams(**sampling_params)

        self.llm = LLM(model=self.model_path,
                       trust_remote_code=True,
                       gpu_memory_utilization=0.95,
                       tensor_parallel_size=generate_params.get("ngpus", 8)
                    )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

    def adapt_for_llm(self, judge_info, prompt):
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # Set to False to strictly disable thinking
        )
        return text

class Qwen3OmniScorer(LLMOfflineScorer):
    def __init__(self, llm_name: str, judge_task: str, generate_params: dict, builder_name=None, parser_name=None):
        super().__init__(llm_name, judge_task, generate_params, builder_name, parser_name)
        try:
            from transformers import Qwen3OmniMoeProcessor
            from qwen_omni_utils import process_mm_info
            os.environ['VLLM_USE_V1'] = '0'
        except:
            from importlib.metadata import version
            logging.error(
                f"Failed to import dependencies for Qwen3OmniScorer. "
                f"transformers=={version('transformers')} may be too old or qwen_omni_utils not installed."
            )
            raise RuntimeError
        self.process_mm_info = process_mm_info
        self.sampling_params = SamplingParams(
                                    temperature=generate_params.get("temperature", 0.6),
                                    top_p=generate_params.get("top_p", 0.95),
                                    top_k=generate_params.get("top_k", 20),
                                    max_tokens=generate_params.get("max_tokens", 16384)
                                )

        self.llm = LLM(model=self.model_path,
                       trust_remote_code=True,
                       gpu_memory_utilization=0.95,
                       tensor_parallel_size=generate_params.get("ngpus", 8),
                       limit_mm_per_prompt={"image": 3, "video": 3, "audio": 3},
                       max_num_seqs=8,
                       max_model_len=generate_params.get("max_model_len", 32768),
                       seed=1234)
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(self.model_path)
    
    def adapt_for_llm(self, judge_info, prompt):
        if "pred_audio" in judge_info:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": judge_info["pred_audio"]},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        else:
            messages = [{"role": "user", "content": prompt}]
        
        formatted_prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        if "pred_audio" in judge_info:
            audios, images, videos = self.process_mm_info(messages, use_audio_in_video=True)
            inputs = {
                "prompt": formatted_prompt,
                "multi_modal_data": {},
                "mm_processor_kwargs": {
                    "use_audio_in_video": True,
                },
            }
            
            if images is not None:
                inputs["multi_modal_data"]["image"] = images
            if videos is not None:
                inputs["multi_modal_data"]["video"] = videos
            if audios is not None:
                inputs["multi_modal_data"]["audio"] = audios
            return inputs
        
        return formatted_prompt   