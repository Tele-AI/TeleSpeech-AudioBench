import logging
from src.evaluator.base import Evaluator
from src.evaluator.llm.llm_handlers import LLMHandlerRegistry
from src.evaluator.llm.llm_handlers import LLMProcessor
from src.registry import registry
from src.prompt.llm_judge import TASK_PROMPT_MAP
from src.utils import retry

logger = logging.getLogger(__name__)

class LLMOfflineScorer(Evaluator):
    def __init__(self, llm_name: str, judge_task: str, generate_params: dict, builder_name=None, parser_name=None):
        logging.info(f"Using vllm to run {llm_name} offline model for judgement...")
        logging.info(f"Supported LLMProcessor type: {LLMHandlerRegistry.list_registered()}")
        logging.info(f"Using parser name: {parser_name}, builder name: {builder_name} for judgement")
        self.parser_name = parser_name
        self.builder_name = builder_name
        self.render_prompt = TASK_PROMPT_MAP.get(judge_task)
        self.model_path = registry.get_model_cfg(llm_name).get("path")
        if self.model_path is None:
            raise ValueError(f"{llm_name} model path is required for LLMOfflineScorer")
        self.sampling_params = None
        self.llm = None

        """for the latest vllm, no need of SamplingParams:
        self.sampling_params = self.llm.get_default_sampling_params()
        if generate_params.get("max_tokens"):
            self.sampling_params.max_tokens = generate_params["max_tokens"]
        """

    def evaluate(self, pred_info_list, fields, **kwargs):
        # prepare judge prompts
        batch_inputs, keys = [], []
        pred_info_map = dict()
        f = self.get_fields(fields)

        for pred_info in pred_info_list:
            pred = pred_info[f["prediction"]]
            key = pred_info[f["key"]]

            judge_info = LLMProcessor.prepare_judge_info(pred_info, builder_name=self.builder_name)
            prompt = self.render_prompt(**judge_info)
            formatted_prompt = self.adapt_for_llm(judge_info, prompt)
            keys.append(key)
            pred_info_map[key] = (pred, formatted_prompt)
            batch_inputs.append(formatted_prompt)

        results, failed_keys = self._generate_and_extract(batch_inputs, keys, pred_info_map, return_failed=True)
        if failed_keys:
            retry_inputs = [pred_info_map[k][1] for k in failed_keys]
            retry_keys = failed_keys
            retry_results = self._retry_generate_batch(retry_inputs, retry_keys, pred_info_map)
            # 更新原始结果
            key_to_result = {r["key"]: r for r in retry_results}
            for i, r in enumerate(results):
                if r["key"] in key_to_result:
                    results[i] = key_to_result[r["key"]]

        return results

    def adapt_for_llm(self, judge_info, prompt):
        raise NotImplementedError("Different for different models, need to rewrite")

    def _generate_and_extract(self, batch_inputs, keys, pred_info_map, return_failed=False):
        outputs = self.llm.generate(batch_inputs, self.sampling_params, use_tqdm=False)
        results = []
        failed_keys = []

        for output, key in zip(outputs, keys):
            generated_text = output.outputs[0].text.strip()
            try:
                score, reason, meta = LLMProcessor.extract(generated_text, parser_name=self.parser_name)
            except Exception:
                if return_failed:
                    failed_keys.append(key)
                score, reason, meta = None, None, None
            pred, _ = pred_info_map[key]
            results.append({
                "key": key,
                "score": score,
                "reason": reason,
                "meta": meta
            })
        return results, failed_keys

    @retry(max_retries=5, sleep_second=2)
    def _retry_generate_batch(self, batch_inputs, keys, pred_info_map):
        """批量重跑失败样本"""
        outputs = self.llm.generate(batch_inputs, self.sampling_params, use_tqdm=False)
        results = []
        for output, key in zip(outputs, keys):
            generated_text = output.outputs[0].text.strip()
            score, reason, meta = LLMProcessor.extract(generated_text, parser_name=self.parser_name)
            pred, _ = pred_info_map[key]
            results.append({
                "key": key, 
                "score": score, 
                "reason": reason,
                "meta": meta
            })
        return results
