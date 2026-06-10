import json
import re
import importlib
import inspect
import json_repair
from typing import Callable, Tuple, Any, Dict, Optional

class LLMHandlerRegistry:
    _builders: Dict[str, Callable[[Dict[str, Any]], Tuple[Any, Any]]] = {}
    _parsers: Dict[str, Callable[[Dict[str, Any]], Tuple[Any, Any]]] = {}

    @classmethod
    def register_builder(cls, name=None):
        def decorator(func):
            key = name or func.__name__
            cls._builders[key] = func
            return func
        return decorator

    @classmethod
    def register_parser(cls, name=None):
        def decorator(func):
            key = name or func.__name__
            cls._parsers[key] = func
            return func
        return decorator

    @classmethod
    def register(cls, name: str, builder: Callable = None, parser: Callable = None):
        if builder:
            cls._builders[name] = builder
        if parser:
            cls._parsers[name] = parser
        return (builder, parser)

    @classmethod
    def get_builder(cls, name):
        return cls._builders.get(name, None)

    @classmethod
    def get_parser(cls, name):
        return cls._parsers.get(name, None)

    @classmethod
    def auto_register_from_module(cls, module_name):
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as e:
            raise ImportError(f"Failed to import module '{module_name}': {e}")

    @classmethod
    def list_registered(cls):
        return {
            "preprocessors": list(cls._builders.keys()),
            "parsers": list(cls._parsers.keys()),
        }


class LLMProcessor:
    LLM_EXPLAIN = re.compile(r'["\s]*Explanation["\s]*:\s*(["“”]?)([\s\S]*?)\1(?=\s*(?:["\s]*Score["\s]*:|\}|\n|$))', re.IGNORECASE)
    LLM_SCORE = re.compile(r'["\s]*Score["\s]*:\s*"?([0-9]+(?:\.[0-9]+)?)"?', re.IGNORECASE)  # support float type

    @staticmethod
    def extract_codeblock_json(text: str) -> str:
        """
        - ```json ... ``` 或 ``` ... ```
        - 结尾缺失 ``` 或写成两个 `
        """
        pattern = r"```(?:json)?\s*(.*?)(?:```|$)"
        match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
        if match:
            content = match.group(1)
            if content.lower().startswith("json"):
                content = content[4:].strip()
            return content.strip()
        return text.strip()

    @classmethod
    def prepare_judge_info(
        cls, pred_info: Dict[str, Any], builder_name: str = None
    ) -> Dict[str, Any]:
        pre_fn = LLMHandlerRegistry.get_builder(builder_name)
        return pre_fn(pred_info) if pre_fn else pred_info

    @classmethod
    def extract(
        cls,
        llm_output: str,
        parser_name: str = None,
        explain_col: str = "Explanation",
        score_col: str = "Score",
        response_type: str = "json",
        custom_parser: Optional[Callable[[Dict[str, Any]], Tuple[Any, Any]]] = None,
    ) -> Tuple[Any, Any, Dict[str, Any]]:
        
        llm_output = llm_output.strip()
        if response_type == "str":
            parsed_output = llm_output
        else:
            try:
                # cleaned_str = cls.extract_codeblock_json(llm_output)
                # parsed_output = json.loads(cleaned_dict, ensure_ascii=False)
                parsed_output = json_repair.repair_json(llm_output, return_objects=True, ensure_ascii=False)
            except Exception:
                parsed_output = llm_output

        parser_fn = custom_parser or (LLMHandlerRegistry.get_parser(parser_name) if parser_name else None)

        if parser_fn:
            try:
                result = parser_fn(parsed_output)
                if not isinstance(result, dict) or "score" not in result:
                    raise ValueError(f"Parser '{parser_name}' must return a dict with 'score' key.")

                return result["score"], result.get("reason", ""), result.get("meta", None)
            except:
                pass

        # default
        if isinstance(parsed_output, dict):
            if explain_col in parsed_output and score_col in parsed_output:
                try:
                    explain = parsed_output[explain_col]
                    score = int(parsed_output[score_col])
                    return score, explain, None
                except Exception:
                    pass

        # fallback to regex
        explain_match = cls.LLM_EXPLAIN.search(llm_output)
        score_match = cls.LLM_SCORE.search(llm_output)
        if explain_match and score_match:
            try:
                score = int(score_match.group(1))
                explain = explain_match.group(2)
                return score, explain, None
            except Exception as e:
                pass
        raise ValueError(f"Failed to extract score and explanation from LLM output: {llm_output}")

LLMHandlerRegistry.auto_register_from_module("src.evaluator.llm.handlers_impl")
