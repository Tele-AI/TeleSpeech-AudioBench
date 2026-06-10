from jiwer import process_words
from typing import Dict
from src.evaluator.base import Evaluator
from src.evaluator.text_utils import TextProcessor
from src.evaluator.process import SimpleTokenizer
from src.utils import parallel_batch

class BaseASREvaluator(Evaluator):
    """
    Part from https://github.com/BytedanceSpeech/seed-tts-eval/tree/main
    Base class providing:
      - unified text preprocessing
      - CER/WER computation
      - optional ASR inference (implemented in subclasses when needed)
    """

    def __init__(self, language="zh"):
        self.language = language
        self.text_processor = TextProcessor(language)

    def compute_score(self, hypo: str, truth: str, uncased=False, simplified_zh=False):
        truth_norm = self.text_processor.normalize_and_clean(truth, simplified_zh=simplified_zh)
        hypo_norm = self.text_processor.normalize_and_clean(hypo, simplified_zh=simplified_zh)

        truth_tokens = SimpleTokenizer.tokenize(truth_norm, uncased=uncased, keep_punc=False)
        hypo_tokens = SimpleTokenizer.tokenize(hypo_norm, uncased=uncased, keep_punc=False)

        truth_joined = " ".join(truth_tokens)
        hypo_joined = " ".join(hypo_tokens)
        measures = process_words(truth_joined, hypo_joined)

        return {
            "clean_ref": truth_joined,
            "clean_pred": hypo_joined,
            "score": {
                "cer_or_wer": measures.wer,
                "subs": measures.substitutions,
                "dele": measures.deletions,
                "inse": measures.insertions,
                "ref_len": len(truth_tokens)
            }
        }

    def run_asr(self, audio_path: str):
        raise NotImplementedError

class ASRAccuracyEvaluator(BaseASREvaluator):
    """
    Evaluate pure ASR ability: compare model output vs ground truth.
    """
    REQUIRED_FIELDS = {"prediction", "reference"}
    def __init__(self, language="zh", max_workers=None):
        super().__init__(language)
        self.max_workers = max_workers or 4

    @parallel_batch(default_workers=4)
    def evaluate(self, pred_info: Dict, fields: Dict, **kwargs):
        f = self.get_fields(fields)
        hypo = pred_info[f["prediction"]]
        truth  = pred_info[f["reference"]]
        score = self.compute_score(hypo, truth)
        return {
            "key": pred_info[f["key"]],
            **score,
        }
