from typing import Dict, Any, Union, List
from src.evaluator.process import SimpleTokenizer, OptionExtractor
from src.utils import parallel_batch

class Evaluator:
    DEFAULT_FIELDS = {
        "key": "key",
        "prediction": "pred",
        "reference": "ref",
        "pred_audio": "pred_audio"
    }
    REQUIRED_FIELDS: set[str] = set()

    def get_fields(self, fields: Dict | None) -> Dict[str, str] | None:
        """
        provide a mapping from dataset to inside usage
        """
        resolved = dict(self.DEFAULT_FIELDS)
        if fields:
            resolved.update(fields)

        missing = self.REQUIRED_FIELDS - resolved.keys()
        if missing:
            raise ValueError(
                f"{self.__class__.__name__} missing the required fields {sorted(missing)}, "
                f"but got {sorted(resolved.keys())}"
            )

        return resolved
    
    def evaluate(self, pred_info_list: List[Dict], fields: Dict | None = None, **kwargs) -> List[Dict]:
        raise NotImplementedError

class ExistMatch(Evaluator):
    """
    referred to https://github.com/DevSinghSachan/emdr2/blob/main/tasks/openqa/dense_retriever/evaluation/qa_validation.py
    """
    REQUIRED_FIELDS = {"prediction", "reference"}
    def __init__(self, keep_punc=False, max_workers=None):
        self.keep_punc = keep_punc
        if max_workers is not None:
            self.max_workers = max_workers

    @parallel_batch(default_workers=4)
    def evaluate(self, pred_info: Dict, fields: Dict, **kwargs):
        # NOTE (TTTdas): If strict sequential matching is required, set keep_punc=False and simply put the ref into a string
        f = self.get_fields(fields)
        pred = pred_info.get(f["prediction"])
        ref  = pred_info.get(f["reference"])
        assert pred is not None and ref is not None, print(f"Must need prediction and reference for ExistMatch: {pred_info}")

        if isinstance(ref, str):
            ref = [ref]
        if not isinstance(ref, list):
            raise ValueError(f"Need List type ref, but got {type(ref)}")

        match = SimpleTokenizer.has_answer(ref, str(pred), uncased=True, keep_punc=self.keep_punc)
        return {"key": pred_info[f["key"]], "score": 1 if match else 0}


class SingleOptionMatch(Evaluator):
    def __init__(self, max_workers=None, cushion=False):
        self.cushion = cushion
        if max_workers is not None:
            self.max_workers = max_workers
                
    @parallel_batch(default_workers=4)
    def evaluate(self, pred_info: Dict, fields: Dict, **kwargs):
        f = self.get_fields(fields)
        pred = pred_info[f["prediction"]]
        ref  = pred_info[f["reference"]]
        if isinstance(ref, list):
            assert len(ref) == 1
            ref = ref[0]
        match_dict = OptionExtractor.has_answer(ref, str(pred), pred_info.get("query", None), cushion=self.cushion)
        return {"key": pred_info[f["key"]], "score": match_dict}
