from dataclasses import dataclass
from typing import Dict, List, Union, Tuple, Any
from src.dataset import BatchLoader, BatchSaver
from src.summarizer.summarizer import Summarizer
from src.evaluator.base import Evaluator

TemplateStruct = Union[str, Dict[str, Any], List[Dict[str, Union[str, List[Dict[str, str]]]]]]
RefType = Union[str, List["RefType"], Tuple["RefType", ...]]
RefsType = List[RefType]

@dataclass
class EvalTaskCfg:
    evaluator: str
    summarizer: str
    fields: Dict[str, str] | None = None
    batch_size: int = None

@dataclass
class InferTaskCfg:
    dataset: Union[str, List[str]]
    template: str
    model: str
    eval_task: Union[str, Dict[str, str]]
    save_pred_audio: bool = False
    reverse_spkr : bool = False  # for multiturn
    use_model_history: bool = True  # for multiturn
    save_latest_only: bool = False  # for multiturn_memory
    task_prompt: str | None = None

@dataclass
class TaskRuntime:
    dataset_name: str
    loader: BatchLoader                  # BatchLoader / 自定义 loader
    saver: BatchSaver                   # BatchSaver
    eval_task_name: str = None
    summary_file: str = None
    evaluator: Evaluator = None     # 只有 eval 才有
    summarizer: Summarizer = None     # 只有 eval 才有
    fields: Dict[str, str] = None  # eval task 字段映射
