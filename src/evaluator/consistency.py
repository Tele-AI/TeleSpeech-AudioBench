import soundfile as sf
import scipy.signal
from typing import Dict
from src.evaluator.asr_eval import BaseASREvaluator
from src.utils import parallel_batch

class SpeechTextConsistencyEvaluator(BaseASREvaluator):
    """
    Compare model's text output vs ASR transcription of generated audio.
    """
    REQUIRED_FIELDS = {"key", "prediction", "pred_audio"}
    def __init__(self, model: str, language="zh", max_workers=None):
        super().__init__(language)
        self.max_workers = max_workers or 4

        if language == "zh":
            from funasr import AutoModel
            self.asr_model = AutoModel(model=model, disable_update=True)
        else:
            from transformers import WhisperProcessor, WhisperForConditionalGeneration
            self.processor = WhisperProcessor.from_pretrained(model)
            self.asr_model = WhisperForConditionalGeneration.from_pretrained(model).to("cuda")

    def run_asr(self, audio_path: str):
        if self.language == "zh":
            res = self.asr_model.generate(input=audio_path, batch_size_s=300)
            return res[0]["text"]
        else:
            wav, sr = sf.read(audio_path)
            if sr != 16000:
                wav = scipy.signal.resample_poly(wav, 16000, sr)
            feats = self.processor(wav, sampling_rate=16000, return_tensors="pt").input_features.to("cuda")
            forced_ids = self.processor.get_decoder_prompt_ids(language="english", task="transcribe")
            pred_ids = self.asr_model.generate(feats, forced_decoder_ids=forced_ids)
            return self.processor.batch_decode(pred_ids, skip_special_tokens=True)[0]

    @parallel_batch(default_workers=4)
    def evaluate(self, pred_info: Dict, fields: Dict, **kwargs):
        f = self.get_fields(fields)
        truth = pred_info[f["prediction"]]
        audio_path = pred_info[f["pred_audio"]]
        
        hypo = self.run_asr(audio_path)
        score = self.compute_score(hypo, truth, uncased=True, simplified_zh=True)
        return {
            "key": pred_info[f["key"]],
            **score,
        }