from typing import Dict
import kaldifeat
import logging
import numpy as np
import onnxruntime as ort
from scipy.special import softmax

from src.evaluator.base import Evaluator
from src.utils import parallel_batch, preprocess_audio

logger = logging.getLogger(__name__)

class DialectSession:
    """
    from https://github.com/Tele-AI/TeleSpeech-DialectIdentify
    """
    def __init__(self, onnx_file: str, device: str = "cpu"):
        self.session = self._init_session(onnx_file, device)
        self.mfcc_extractor = self._init_mfcc_extractor()
        self.sr = 16000
        self.DIALECT_TOKENS = {
            0: "ct", 1: "kej", 2: "mand", 3: "min", 4: "wuy", 5: "zha", 6: "zhc",
            7: "zhd", 8: "zhg", 9: "zhj", 10: "zhs", 11: "zhu", 12: "zhw", 13: "zhx"
        }
        logger.info(f"Loading dialect classify model: {onnx_file} Successfully")

    def _init_session(self, onnx_file, device):
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.add_session_config_entry("session.intra_op.allow_spinning", "1")
        provider = "CPUExecutionProvider" if device == "cpu" else "CUDAExecutionProvider"
        return ort.InferenceSession(
            onnx_file,
            providers=[provider],
            sess_options=sess_options,
        )

    def _init_mfcc_extractor(self, sr: int = 16000):
        opts = kaldifeat.MfccOptions()
        opts.device = "cpu"
        opts.frame_opts.dither = 0
        opts.frame_opts.snip_edges = False
        opts.frame_opts.samp_freq = sr
        opts.use_energy = False
        opts.mel_opts.num_bins = 40
        opts.mel_opts.low_freq = 40
        opts.mel_opts.high_freq = -200
        opts.num_ceps = 40
        return kaldifeat.Mfcc(opts)

    def classify(self, wav_file: str) -> str:
        wav = preprocess_audio(wav_file, target_sr=self.sr)
        wav = wav * (1 << 15)
        feats = self.mfcc_extractor(wav.squeeze())
        out = self.session.run(
            input_feed={"feats": feats.unsqueeze(0).numpy()},
            output_names=["labels"]
        )[0]
        pred = np.argmax(softmax(out, axis=1))
        return self.DIALECT_TOKENS[int(pred)]

class DialectClassify(Evaluator):
    REQUIRED_FIELDS = {"pred_audio", "key", "dialect"}
    def __init__(self, model: str, max_workers=None):
        if max_workers is not None:
            self.max_workers = max_workers
        self.onnx_sess = DialectSession(model)
        self.dialect_mapping = {
            "ct": "粤语", "zhs": "河南话", "zhc": "四川话",
            "zhd": "东北话", "wuy": "上海话", "mand": "普通话"
        }
    
    @parallel_batch(default_workers=4)
    def evaluate(self, pred_info: Dict, fields: Dict, **kwargs):
        f = self.get_fields(fields)
        ref_dialect  = pred_info[f["dialect"]]
        pred_audio = pred_info[f["pred_audio"]]
        
        res = self.onnx_sess.classify(pred_audio)
        mapped_dialect = self.dialect_mapping.get(res, None)
        logger.info(f"key: {pred_info[f["key"]]} recognition dialect: {mapped_dialect}")

        score = int(mapped_dialect == ref_dialect) if mapped_dialect else 0
        return {"key": pred_info[f["key"]], "score": score}