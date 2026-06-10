import soundfile as sf
from typing import List, Dict
from src.evaluator.base import Evaluator

class Emo2vec(Evaluator):
    REQUIRED_FIELDS = {"key", "pred_audio", "answer_emo"}
    
    def __init__(self, model: str, strict: bool = True):
        from funasr import AutoModel
        self.model = AutoModel(model=model, hub="ms", disable_update=True)
        self.strict = strict
        self.MAX_DURATION = 300  # 5min

    def evaluate(self, pred_info_list: List[Dict], fields: Dict, **kwargs):
        # emo2vec model support batch generate, do not need @parallel_batch
        f = self.get_fields(fields)
       
        results = []
        model_input_audios = []
        model_input_infos = []
        do_filter = self.MAX_DURATION is not None

        for info in pred_info_list:
            audio_path = info.get(f["pred_audio"])
            audio, sr = sf.read(audio_path)
            if do_filter:
                duration = len(audio) / sr
                if duration > self.MAX_DURATION:
                    results.append({"key": info.get(f["key"]) , "score": 0})
                    continue

            model_input_audios.append(audio_path)
            model_input_infos.append(info)

        if model_input_audios:
            model_outputs = self.model.generate(
                model_input_audios,
                output_dir=None,
                granularity="utterance",
                extract_embedding=False
            )
        else:
            model_outputs = []

        for output, info in zip(model_outputs, model_input_infos):
            label_scores = {
                label.split("/")[-1].lower(): score
                for label, score in zip(output["labels"], output["scores"])
            }
            ref_emotions = [emo.lower() for emo in info[f["answer_emo"]]]

            if self.strict:
                neutral_count = sum(1 for emo in ref_emotions if emo == "neutral")
                if neutral_count <= len(ref_emotions) // 2:
                    # remove "neutral"
                    filtered_ref_emotions = [emo for emo in ref_emotions if emo != "neutral"]
                else:
                    filtered_ref_emotions = ref_emotions
            else:
                filtered_ref_emotions = ref_emotions

            score = max((label_scores.get(emo, 0) for emo in filtered_ref_emotions), default=0)
            results.append({"key": info[f["key"]], "score": score})
        return results
