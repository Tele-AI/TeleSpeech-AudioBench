import os
import logging
import copy
import json
from tqdm import tqdm
from typing import List, Dict
from src.registry import registry
from src.dataset import BatchLoader, BatchSaver
from src.config import TaskRuntime

logger = logging.getLogger(__name__)

class Pipeline:
    @staticmethod
    def create(mode: str, task: str, save_dir: str, **kwargs):
        if mode == "infer":
            return InferenceTask(mode, task, save_dir, **kwargs)
        elif mode == "eval":
            return EvalTask(mode, task, save_dir, **kwargs)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

class BaseTask:
    def __init__(self, mode: str, task: str, save_dir: str, **kwargs):
        self.mode = mode
        self.save_dir = save_dir
        self.kwargs = copy.deepcopy(kwargs)
        self.infer_task_name = task

        self.infer_task_cfg = registry.get_infer_task(task)
        self.model_name = self.kwargs.get("model") or self.infer_task_cfg.model
        
        if self.kwargs.get("save_pred_audio"):
            self.save_pred_audio = eval(self.kwargs.get("save_pred_audio"))
        else:
            self.save_pred_audio = self.infer_task_cfg.save_pred_audio

        self.runtimes: List[TaskRuntime] = []
        if mode == "infer":
            self._build_infer_runtimes()
        elif mode == "eval":
            self._build_eval_runtimes()

    def _get_save_file(self, stage, filename=None, suffix=None):
        save_path = os.path.join(
            self.save_dir,
            stage,
            self.model_name,
            self.infer_task_name,
        )
        os.makedirs(save_path, exist_ok=True)

        if suffix is None and self.save_pred_audio:
            save_audio_dir = os.path.join(save_path, filename)
            os.makedirs(save_audio_dir, exist_ok=True)
            self.kwargs["save_audio_dir"] = save_audio_dir

        fname = f"{filename}.{suffix}.jsonl" if suffix else f"{filename}.jsonl"
        return os.path.join(save_path, fname)

    def _build_infer_runtimes(self):
        datasets = self.infer_task_cfg.dataset  # 支持按list给多个dataset
        if not isinstance(datasets, list):
            datasets = [datasets]

        self.template = registry.get_template(self.infer_task_cfg.template)
        self.predictor = registry.get_model(self.model_name)

        for dataset_name in datasets:
            loader = registry.get_dataset(dataset_name)
            if self.kwargs.get("bsz"):
                loader.batch_size = int(self.kwargs["bsz"])
            saver = BatchSaver(self._get_save_file("prediction", filename=dataset_name))
            self.runtimes.append(TaskRuntime(dataset_name=dataset_name, loader=loader, saver=saver))

    def _build_eval_runtimes(self):
        datasets = self.infer_task_cfg.dataset
        if not isinstance(datasets, list):
            datasets = [datasets]

        eval_task_names = self.kwargs.get("eval_task") or self.infer_task_cfg.eval_task  # support list
        if not isinstance(eval_task_names, list):
            eval_task_names = [eval_task_names]

        for dataset_name in datasets:
            pred_file = self._get_save_file("prediction", filename=dataset_name)
            for eval_task_name in eval_task_names:
                cfg = registry.get_eval_task(eval_task_name)
                loader = BatchLoader(
                    pred_file,
                    batch_size=int(self.kwargs.get("bsz") or cfg.batch_size or 1),
                    do_infer=False
                )
                saver = BatchSaver(self._get_save_file("result", filename=dataset_name, suffix=eval_task_name))
                summary_file = self._get_save_file("summary", filename=dataset_name, suffix=eval_task_name)
                self.runtimes.append(TaskRuntime(
                    dataset_name=dataset_name,
                    loader=loader,
                    saver=saver,
                    summary_file=summary_file,
                    eval_task_name=eval_task_name,
                    evaluator=registry.get_evaluator(cfg.evaluator),
                    summarizer=registry.get_summarizer(cfg.summarizer),
                    fields=getattr(cfg, "fields", None)
                ))

    @staticmethod
    def save_summary(summary_file, stat: Dict, save_all=False):
        if summary_file:
            with open(summary_file, "w", encoding="utf-8") as f:
                outputs = stat if save_all else {"score": stat["score"]}
                print(json.dumps(outputs, ensure_ascii=False), file=f)
            logger.info(f"Total score saved to {summary_file}")

    def run(self):
        raise NotImplementedError

class InferenceTask(BaseTask):
    def run(self):
        for rt in self.runtimes:
            with tqdm(total=len(rt.loader), desc="Infer", unit="it") as pbar:  # , disable=True
                for batch_data in rt.loader:
                    inputs = [self.template.load(**sample) for sample in batch_data]
                    keys = [data.get(rt.loader.key_col, i) for i, data in enumerate(batch_data)]  # key_col is defined in dataset.yaml

                    pred_args = {
                        "reverse_spkr": self.infer_task_cfg.reverse_spkr,
                        "use_model_history": self.infer_task_cfg.use_model_history,
                        "save_latest_only": self.infer_task_cfg.save_latest_only,
                        "task_prompt": self.infer_task_cfg.task_prompt,
                    }
                    save_audio_dir = self.kwargs.get("save_audio_dir")
                    if save_audio_dir:
                        pred_args["pred_audio"] = [f"{save_audio_dir}/{key}_pred.wav" for key in keys]

                    outputs = self.predictor.inference(inputs, **pred_args)
                    for key, data, output in zip(keys, batch_data, outputs):
                        extra_log = {info: data.get(info) for info in (rt.loader.meta_col or [])}  # extra informataion for eval, egs. emotion
 
                        if isinstance(output, dict):
                            out_log = {"key": key, **output, **extra_log}
                            rt.saver.save_one(out_log)
                        elif isinstance(output, list):
                            for round_log in output:
                                out_log = {"key": key, **round_log, **extra_log}
                                rt.saver.save_one(out_log)
                        else:
                            raise ValueError(f"Not supported output type: {type(output)}")
                    pbar.update(len(batch_data))

class EvalTask(BaseTask):
    def run(self):
        for rt in self.runtimes:
            scores = []
            for batch_data in rt.loader:
                # sinlge_data: {"key":xxx, "pred":xxx, "ref":xxx, "query":xxx, ...}

                results = rt.evaluator.evaluate(
                    pred_info_list=batch_data,
                    fields=rt.fields,
                )

                for res, sample in zip(results, batch_data):
                    if self.kwargs.get("keep_meta_info", True):
                        # Keep evaluator-updated values in `res`; only fill missing metadata from `sample`.
                        for k, v in sample.items():
                            if k not in res:
                                res[k] = v

                    scores.append(res["score"])
                    rt.saver.save_one(res)

            stat = rt.summarizer.statistic(scores)
            logger.info(f"[{rt.dataset_name} | {rt.eval_task_name}] total_score: {stat}")
            self.save_summary(rt.summary_file, stat, save_all=True)
