import functools
import importlib
import logging
import time
import json
from typing import Dict, List, Any
import torch
import torchaudio
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import pandas as pd
from datasets import load_dataset

logger = logging.getLogger(__name__)

class TupleEncoder:
    @staticmethod
    def encode(obj):
        if isinstance(obj, tuple):
            return {"__tuple__": True, "items": [TupleEncoder.encode(i) for i in obj]}
        elif isinstance(obj, list):
            return [TupleEncoder.encode(i) for i in obj]
        elif isinstance(obj, dict):
            return {k: TupleEncoder.encode(v) for k, v in obj.items()}
        else:
            return obj

    @staticmethod
    def decode(obj):
        if isinstance(obj, dict) and obj.get("__tuple__") is True:
            return tuple(TupleEncoder.decode(i) for i in obj["items"])
        elif isinstance(obj, list):
            return [TupleEncoder.decode(i) for i in obj]
        elif isinstance(obj, dict):
            return {k: TupleEncoder.decode(v) for k, v in obj.items()}
        else:
            return obj

class TupleJSONLConverter:
    @staticmethod
    def load_file(path, tuple_decode):
        with open(path, "r", encoding="utf-8") as f:
            if tuple_decode:
                return [TupleJSONLConverter.loads(line) for line in f]
            else:
                return [json.loads(line) for line in f]
    
    @staticmethod
    def loads(line):
        return TupleEncoder.decode(json.loads(line))
    
    @staticmethod
    def dumps(obj):
        return json.dumps(TupleEncoder.encode(obj), ensure_ascii=False)

def retry(max_retries=3, sleep_second=5, default=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for _ in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    logger.error(f"retry after: {e}")
                    time.sleep(sleep_second)
            raise last_exception
        return wrapper
    return decorator

def parallel_batch(default_workers=4):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, pred_info_list: List[Dict], fields: Dict | None = None, **kwargs):
            if not isinstance(pred_info_list, list):
                raise ValueError("pred_info_list must be a list")

            workers = min(
                getattr(self, "max_workers", default_workers),
                len(pred_info_list),
            )
            results = [None] * len(pred_info_list)

            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {}
                # ensure the return order is the same as the input order
                for i, pred_info in enumerate(pred_info_list):
                    futures[
                        executor.submit(
                            func,
                            self,
                            pred_info=pred_info,
                            fields=fields,
                            **kwargs,
                        )
                    ] = i

                for future in as_completed(futures):
                    idx = futures[future]
                    results[idx] = future.result()
            return results
        return wrapper
    return decorator

def make_object(class_name: str, **kwargs: Dict[str, Any]) -> Any:
    module_name, qualname = class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, qualname)
    return cls(**kwargs)

def preprocess_audio(source, target_sr=16000, device=None):
    if isinstance(source, str):
        waveform, sr = torchaudio.load(source)
    elif isinstance(source, bytes):
        waveform, sr = torchaudio.load(BytesIO(source))
    else:
        raise ValueError("Unsupported audio source type")

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

    if device is not None:
        waveform = waveform.to(device)

    return waveform

def parse_repo_and_pattern(path: str):
    """
    Supports two types of input:
    1. Local directory (already downloaded by the user):
       Treated as is_local=True, with pattern="*.parquet"
    2. HuggingFace-style path:
       repo/dataset_dir or repo/subdir/subdir
    """
    # --- Case 1: clear local dir ---
    if os.path.exists(path):  # support file
        return {"type": "local", "repo": os.path.abspath(path), "pattern": "*.parquet"}

    # --- Case 2: split by "/" to detect HF repo ---
    parts = path.strip("/").split("/")
    if len(parts) >= 2:
        # Tele-AI/TeleSpeech-AudioBench, noise-zh/babble_-5dB-zh
        # Tele-AI/TeleSpeech-AudioBench, llamaqa-zh
        repo = "/".join(parts[:2])
        pattern = "/".join(parts[2:]) if len(parts) > 2 else None
        if pattern:
            return {"type": "hf", "repo": repo, "pattern": pattern}

    raise ValueError(f"Unrecognized dataset path: {path}")

def load_and_process_parquet_dataset(
    repo_or_path,
    data_dir_pattern,
    audio_output_dir,
    key_col="key",
    is_local=False,
    tuple_decode=True,
    extra_audio_keys: List[str]=None
):
    """
    Load from parquet (local or remote), decode audios to audio_output_dir.
    Supports multiple audio keys (e.g., for multi-turn dialog like user_audio1, bot_audio1, ...).
    """
    if extra_audio_keys is None:
        extra_audio_keys = ["user_audio1", "user_audio2", "user_audio3", "user_audio4"]  # TELEVAL multiturn-memory keys
    
    os.makedirs(audio_output_dir, exist_ok=True)
    if is_local:
        wildcard = data_dir_pattern if data_dir_pattern.endswith(".parquet") else os.path.join(data_dir_pattern, "*.parquet")
        data_files = {"test": os.path.join(repo_or_path, wildcard)}
        ds = load_dataset("parquet", data_files=data_files, split="test")
    else:
        wildcard = data_dir_pattern if data_dir_pattern.endswith(".parquet") else f"{data_dir_pattern}/*.parquet"
        ds = load_dataset(repo_or_path, data_files={"test": wildcard}, split="test")
    df = pd.DataFrame(ds)
    
    records = []
    for row in df.to_dict(orient="records"):
        answer = row.get("answer", None)
        if isinstance(answer, str):
            try:
                answer = json.loads(answer)
            except:
                pass
        if answer and tuple_decode:
            answer = TupleEncoder.decode(answer)
        row["answer"] = answer

        for audio_key in ["audio"] + extra_audio_keys:
            audio = row.get(audio_key)
            if not audio:
                continue

            audio_filename = f"{row[key_col]}.wav" if audio_key == "audio" else f"{row[key_col]}_{audio_key}.wav"  # for multi-audio
            audio_path = os.path.abspath(os.path.join(audio_output_dir, audio_filename))

            if not os.path.exists(audio_path):
                waveform = preprocess_audio(audio["bytes"])
                torchaudio.save(audio_path, waveform, 16000, encoding="PCM_S", bits_per_sample=16)

            row[audio_key] = audio_path  # update to path
        records.append(row)

    return records
