import logging
import os
import shutil
import json
import pandas as pd
import tempfile
import atexit
from tqdm import tqdm
from typing import Dict, List
from pathlib import Path
from src.utils import load_and_process_parquet_dataset, parse_repo_and_pattern, TupleJSONLConverter, TupleEncoder
tqdm.pandas()

logger = logging.getLogger(__name__)

class BatchLoader:
    def __init__(self, file, key_col="key", do_infer=True, meta_col=None, 
                 batch_size=1, start=0, limit=0, save_query_audio_dir=None, tuple_decode=True):
        self.file = file
        self.key_col = key_col

        self.meta_col = meta_col
        self.batch_size = batch_size
        self.index = 0
        self._temp_dir = None

        file = Path(file)
        if file.is_file() and file.suffix == ".jsonl":
            records = self.load_jsonl(file, tuple_decode=tuple_decode)
        elif do_infer and isinstance(file, (str, Path)) and "/" in str(file):
            records = self._process_parquet_records(str(file), save_query_audio_dir)
        else:
            raise NotImplementedError(f"Unsupported file type: {file}")

        if limit > 0:
            records = records[:limit]
        if start > 0:
            records = records[start:]
        
        self.df = pd.DataFrame(records)

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= len(self.df):
            raise StopIteration
        batch = self.df.iloc[self.index : self.index + self.batch_size].to_dict(orient="records")
        self.index += self.batch_size
        return batch
    
    def __len__(self):
        return len(self.df)

    def _process_parquet_records(self, repo_data_dir, save_audio_root):
        info = parse_repo_and_pattern(repo_data_dir)

        repo_or_path, pattern = info["repo"], info["pattern"]
        is_local = info["type"] == "local"
        base_subdir = os.path.basename(os.path.normpath(pattern))

        if save_audio_root:
            audio_output_dir = os.path.join(save_audio_root, base_subdir)
            os.makedirs(audio_output_dir, exist_ok=True)
        else:
            self._temp_dir = tempfile.mkdtemp(prefix="temp_audio_")
            atexit.register(self.cleanup)
            audio_output_dir = os.path.join(self._temp_dir, base_subdir)
            os.makedirs(audio_output_dir, exist_ok=True)
        logger.info(f"using {audio_output_dir} to save query audio")

        return load_and_process_parquet_dataset(
            repo_or_path, pattern, audio_output_dir, key_col=self.key_col, is_local=is_local
        )

    def cleanup(self):
        if self._temp_dir and os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir, ignore_errors=True)

    @staticmethod
    def load_jsonl(file, tuple_decode):
        return TupleJSONLConverter.load_file(file, tuple_decode)

    @staticmethod
    def load_json(file):
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
        else:
            raise ValueError(f"Unsupported JSON structure in {file}")


class BatchSaver:
    def __init__(self, file, overwrite=True):
        self.file = Path(file)
        if self.file.suffix == "":
            raise ValueError(f"Invalid file path: {self.file}")
        self.file.parent.mkdir(parents=True, exist_ok=True)  # os.makedirs

        if overwrite and self.file.exists():
            print(f"Saving file already exists and will be removed: {self.file}")
            self.file.unlink()  # os.remove

    def save_one(self, item: Dict):
        ext = self.file.suffix.lower()
        if ext != ".jsonl":
            raise ValueError("Only support jsonl file")
        with self.file.open("a", encoding="utf-8") as f:
            f.write(TupleJSONLConverter.dumps(item) + "\n")

    def save_all(self, items: List[Dict]):
        ext = self.file.suffix.lower()
        if ext == ".jsonl":
            with self.file.open("w", encoding="utf-8") as f:
                for item in items:
                    f.write(TupleJSONLConverter.dumps(item) + "\n")

        elif ext == ".parquet":
            df = pd.DataFrame([TupleEncoder.encode(item) for item in items])
            df.to_parquet(self.file, index=False)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    def save_parquet(self, df: pd.DataFrame, columns=None):
        records = df.to_dict(orient="records")
        if columns:
            records = [{k: v for k, v in item.items() if k in columns} for item in records]
        self.save_all(records)
