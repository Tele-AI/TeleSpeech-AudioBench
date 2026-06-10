import logging
from typing import Dict, List, Union, Any
from collections import Counter

logger = logging.getLogger(__name__)

class Summarizer:
    def __init__(self, rescale="base", power=2, score_scale=5, reverse=False):
        logger.info(f"Using rescale type: {rescale}")
        self.rescale = rescale
        self.power = power
        self.score_scale = score_scale
        self.reverse = reverse

        if rescale == "base":
            self.rescale_func = self._rescale_base
        elif rescale == "power":
            self.rescale_func = self._rescale_power
        else:
            raise ValueError(f"Unknown rescale type: {rescale}")
    
    def _check_scores(self, scores: List[Any]):
        if any(s is None for s in scores if not isinstance(s, dict)):
            raise ValueError("Scores list contains None values, need re-run evaluator.")
    
    def _rescale_base(self, score):
        return score
    
    def _rescale_power(self, score):
        if self.score_scale <= 0:
            raise ValueError("score_scale must be positive")
        return ((score / self.score_scale) ** self.power) * 100

    def statistic(self, scores: List[Any], **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

class Avg(Summarizer):
    def statistic(self, scores: List[float], **kwargs):
        values = [self.rescale_func(float(s)) for s in scores]
        avg = sum(values) / len(values)
        return {"score": "AVG: {:.2f}%".format(avg)}

class AvgInfo(Summarizer):
    def statistic(self, scores: List[Union[float, Dict[str, float]]], **kwargs):
        keys = next(s for s in scores if isinstance(s, dict)).keys()
        result = {}
        for k in keys:
            values = [
                self.rescale_func(float(s[k])) if isinstance(s, dict) else 0.0
                for s in scores
            ]
            avg = sum(values) / len(values)
            result[k] = "{:.2f}%".format(avg)

        return result

class AvgHumanDialChallenge(Summarizer):
    def statistic(self, scores: List[Union[float, Dict[str, float]]], **kwargs):
        if isinstance(scores[0], dict):
            keys = scores[0].keys()
            result = {}
            avg_cnt = 0
            for key in keys:
                values = [float(s[key]) for s in scores if key in s]
                avg = sum(values) / len(values)
                result[key] = "{:.2f}".format(avg)
                avg_cnt += avg
            
            result["avg_all_score"] = "{:.2f}".format(avg_cnt / len(keys))
            return result
        else:
            raise ValueError(f"Error format for scores: {type(scores[0])}")


class AvgThreshold(Summarizer):
    def __init__(self, rescale, threshold=60, power=2):
        super().__init__(rescale, power)
        self.threshold = threshold
    
    def statistic(self, scores: List[float], **kwargs):
        self._check_scores(scores)
        # scores = list(map(lambda x: self.rescale_func(float(x)), scores))
        scores = [self.rescale_func(float(x)) for x in scores]
        score_count = Counter(scores)

        avg = sum(scores) / len(scores)
        above_threshold = sum(count for score, count in score_count.items() if score > self.threshold)
        return {"score": "AVG: {:.2f}".format(avg), "above_threshold": "above{}: {}".format(self.threshold, above_threshold)}

class AvgMOS(Summarizer):
    def statistic(self, scores: List[float], **kwargs):
        avg = sum(map(float, scores)) / len(scores)
        return {"score": "DNSMOS: {:.2f}".format(avg)}

class AvgWER(Summarizer):
    def statistic(self, scores: List[Dict], **kwargs):
        """
        score = {"ref_len": ref_len, "subs": subs, "dele": dele, "inse": inse, "wer": wer}
        """
        total_ref_len = 0
        total_subs = 0.0
        total_dele = 0.0
        total_inse = 0.0

        for score in scores:
            total_ref_len += score.get("ref_len", 0)
            total_subs += score.get("subs", 0.0)
            total_dele += score.get("dele", 0.0)
            total_inse += score.get("inse", 0.0)

        if total_ref_len == 0:
            raise ValueError("Not enough ref_len to static")

        avg_wer = (total_subs + total_dele + total_inse) / total_ref_len * 100
        if self.reverse:
            return {"score": "Consis: {:.2f}%".format(100 - avg_wer)}
        else:
            return {"score": "WER_or_CER: {:.2f}%".format(avg_wer)}