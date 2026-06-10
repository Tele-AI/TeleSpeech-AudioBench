"""
Modified from https://github.com/OpenBMB/UltraEval-Audio/blob/main/src/registry.py

Functions to handle registration of evals. To add a new eval to the registry,
add an entry in one of the YAML files in the `../registry` dir.
By convention, every eval name should start with {base_eval}.{split}.
"""

import difflib
import functools
import logging
import os
from pathlib import Path
from typing import (
    Any,
    Dict,
    Generator,
    Iterator,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import yaml

from src.config import EvalTaskCfg, InferTaskCfg
from src.dataset import BatchLoader
from src.evaluator.base import Evaluator
from src.models.base import Model
from src.prompt.template import DataTemplate
from src.summarizer.summarizer import Summarizer
from src.utils import make_object

logger = logging.getLogger(__name__)

SPEC_RESERVED_KEYWORDS = ["key", "group", "cls"]


T = TypeVar("T")
RawRegistry = Dict[str, Any]
DEFAULT_PATHS = [Path(__file__).parents[1].resolve() / "registry"]


class Registry:
    def __init__(self, registry_paths=None):
        if registry_paths is None:
            registry_paths = DEFAULT_PATHS
        self._registry_paths = [
            Path(p) if isinstance(p, str) else p for p in registry_paths
        ]

    def _dereference(
        self, name: str, d: RawRegistry, object: str, config_only: bool = False, **kwargs: dict
    ) -> Optional[T]:
        if name not in d:
            logger.error(
                (
                    f"{object} '{name}' not found. "
                    f"Closest matches: {difflib.get_close_matches(name, d.keys(), n=5)}"
                )
            )
            raise ValueError(f"Error object {object} '{name}'")
            # return None

        logger.debug(f"Looking for {name}")

        spec = d[name]
        if kwargs:  # specified parameters take precedence over default one
            spec["args"].update(kwargs)
        if config_only:  # only get config, not object
            return spec["args"]
        
        try:
            return make_object(spec["cls"], **spec["args"])
        except TypeError as e:
            raise TypeError(f"Error while processing {object} '{name}': {e}")

    def get_model(self, name: str, **kwargs) -> Optional[Model]:
        return self._dereference(name, self._model, "model", **kwargs)
    
    def get_model_cfg(self, name: str, **kwargs) -> Optional[dict]:
        return self._dereference(name, self._model, "model", config_only=True, **kwargs)

    def get_evaluator(self, name: str, **kwargs) -> Optional[Evaluator]:
        return self._dereference(name, self._evaluator, "evaluator", **kwargs)

    def get_eval_task(self, name: str, **kwargs) -> Optional[EvalTaskCfg]:
        return self._dereference(name, self._eval_task, "eval task", **kwargs)

    def get_infer_task(self, name: str, **kwargs) -> Optional[InferTaskCfg]:
        return self._dereference(name, self._infer_task, "infer task", **kwargs)

    def get_dataset(self, name: str, **kwargs) -> Optional[BatchLoader]:
        return self._dereference(name, self._dataset, "dataset", **kwargs)

    def get_template(self, name: str, **kwargs) -> Optional[DataTemplate]:
        return self._dereference(name, self._template, "template", **kwargs)

    def get_summarizer(self, name: str, **kwargs) -> Optional[Summarizer]:
        return self._dereference(name, self._summarizer, "summarizer", **kwargs)

    def _load_file(self, path: Path) -> Generator[Tuple[str, Path, dict], None, None]:
        # from https://github.com/openai/evals/blob/main/evals/registry.py
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f.read())

        if d is None or not isinstance(d, dict):
            # no entries in the file
            return

        for name, spec in d.items():
            yield name, path, spec

    def _load_directory(
        self, path: Path
    ) -> Generator[Tuple[str, Path, dict], None, None]:
        # from https://github.com/openai/evals/blob/main/evals/registry.py
        files = Path(path).glob("*.yaml")
        for file in files:
            yield from self._load_file(file)

    def _load_resources(
        self, registry_path: Path, resource_type: str
    ) -> Generator[Tuple[str, Path, dict], None, None]:
        # from https://github.com/openai/evals/blob/main/evals/registry.py
        path = registry_path / resource_type
        logging.info(f"Loading registry from {path}")

        if os.path.exists(path):
            if os.path.isdir(path):
                yield from self._load_directory(path)
            else:
                yield from self._load_file(path)

    @staticmethod
    def _validate_reserved_keywords(spec: dict, name: str, path: Path) -> None:
        # from https://github.com/openai/evals/blob/main/evals/registry.py
        for reserved_keyword in SPEC_RESERVED_KEYWORDS:
            if reserved_keyword in spec:
                raise ValueError(
                    f"{reserved_keyword} is a reserved keyword, but was used in {name} from {path}"
                )

    def _load_registry(
        self, registry_paths: Sequence[Path], resource_type: str
    ) -> RawRegistry:
        # from https://github.com/openai/evals/blob/main/evals/registry.py
        """Load registry from a list of regstry paths and a specific resource type

        Each path includes yaml files which are a dictionary of name -> spec.
        """

        registry: RawRegistry = {}

        for registry_path in registry_paths:
            for name, path, spec in self._load_resources(registry_path, resource_type):
                assert name not in registry, f"duplicate entry: {name} from {path}"
                self._validate_reserved_keywords(spec, name, path)

                spec["key"] = name
                spec["group"] = str(os.path.basename(path).split(".")[0])
                spec["registry_path"] = registry_path

                if "class" in spec:
                    spec["cls"] = spec["class"]
                    del spec["class"]

                registry[name] = spec

        return registry

    @functools.cached_property
    def _model(self) -> RawRegistry:
        return self._load_registry(self._registry_paths, "model")

    @functools.cached_property
    def _dataset(self) -> RawRegistry:
        return self._load_registry(self._registry_paths, "dataset")

    @functools.cached_property
    def _evaluator(self) -> RawRegistry:
        return self._load_registry(self._registry_paths, "evaluator")

    @functools.cached_property
    def _template(self) -> RawRegistry:
        return self._load_registry(self._registry_paths, "template")

    @functools.cached_property
    def _eval_task(self) -> RawRegistry:
        return self._load_registry(self._registry_paths, "eval_task")

    @functools.cached_property
    def _infer_task(self) -> RawRegistry:
        return self._load_registry(self._registry_paths, "infer_task")

    @functools.cached_property
    def _summarizer(self) -> RawRegistry:
        return self._load_registry(self._registry_paths, "summarizer")

registry = Registry()
