import requests
import json
import logging
import threading
import itertools
from src.evaluator.base import Evaluator
from src.evaluator.llm.llm_handlers import LLMHandlerRegistry
from src.evaluator.llm.llm_handlers import LLMProcessor
from src.prompt.llm_judge import TASK_PROMPT_MAP
from src.utils import retry, parallel_batch

logger = logging.getLogger(__name__)

class LLMAPIScorer(Evaluator):
    REQUIRED_FIELDS = {"key", "prediction"}
    def __init__(self, llm_name: str, judge_task: str, api_keys: dict, max_key_jobs=None, max_workers=None, builder_name=None, parser_name=None):
        logging.info(f"Using {llm_name} API for judgement...")
        logging.info(f"Supported LLMProcessor type: {LLMHandlerRegistry.list_registered()}")
        logging.info(f"Using parser name: {parser_name}, builder name: {builder_name} for judgement")
        assert len(api_keys) > 0
        self.render_prompt = TASK_PROMPT_MAP.get(judge_task)
        if self.render_prompt is None:
            raise ValueError(f"Unsupported task: {judge_task}")
        self.llm_name = llm_name
        self.api_keys = api_keys
        self.max_key_jobs = max_key_jobs or 1
        self.key_job_count = {key: 0 for key in api_keys}
        self.max_workers = max_workers or len(api_keys) * self.max_key_jobs
        self.parser_name = parser_name
        self.builder_name = builder_name
        self.urls = {
            key: (
                f"https://{key}.openai.azure.com/"
                f"openai/deployments/{llm_name}/chat/completions?api-version=2025-01-01-preview"
            )
            for key in api_keys
        }
        self.key_cycle = itertools.cycle(self.api_keys.items())
        self.lock = threading.Lock()

    def get_next_key(self):
        with self.lock:
            key_name, key_value = next(self.key_cycle)
            return key_name, key_value, self.urls[key_name]

    def api_generate(self, temp, api_key, url):
        headers = {
            "Content-Type": "application/json",
            "api-key": api_key
        }
        input_data = {
            "model": self.llm_name,
            "messages": [
                {"role": "user", "content": temp}
            ]
        }
        response = requests.post(url, headers=headers, data=json.dumps(input_data))
        response.raise_for_status()
        response_data = response.json()
        return response_data["choices"][0]["message"]["content"].strip()

    @retry(max_retries=5, sleep_second=3)
    def get_server_response(self, temp, api_key, url):
        api_response = self.api_generate(temp, api_key, url)
        score, reason, meta = LLMProcessor.extract(api_response, parser_name=self.parser_name)
        return score, reason, meta

    @parallel_batch(default_workers=4)
    def evaluate(self, pred_info, fields, **kwargs):
        f = self.get_fields(fields)
        pred = pred_info[f["prediction"]]
        key = pred_info[f["key"]]

        key_name, api_key, url = self.get_next_key()
        judge_info = LLMProcessor.prepare_judge_info(pred_info, builder_name=self.builder_name)
        if judge_info is None:
            logger.info("Failed to prepare judge info, the judge_info is None.")
            return {"key": key, "score": 0, "reason": "Failed to prepare judge info."}

        prompt = self.render_prompt(**judge_info)
        score, reason, meta = self.get_server_response(prompt, api_key, url)
        return {"key": key, "score": score, "reason": reason, "meta": meta}
