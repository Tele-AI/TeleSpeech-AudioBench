import logging
import soundfile as sf
import os
import torch
from typing import Any, Union, Dict, List, Tuple
from src.prompt.system_prompt import get_system_prompts

logger = logging.getLogger(__name__)

class Model:
    def __init__(self, sample_params: Dict[str, Any] = None):
        """
        :param sample_params: from registry/model/*.yaml
        """
        if sample_params is None:
            sample_params = {}
        self.sample_params = sample_params
        self.mono_only = True
        self.inputs_sr = 16000
        self.system_prompt = None
        self.stable_system_prompt = False
        self._system_prompt_initialized = False
        self._default_system_prompt = None

    @torch.inference_mode()
    def inference(self, inputs: Union[List[Union[List[Dict], Dict]]], **kwargs) -> Union[str, Dict[int, str], List[str]]:
        """
        :param inputs: data format, after template.load
        :param kwargs: other configs, e.g. pred_audio, use_model_history
        """
        def _generate(inputs, **kwargs):
            self._init_default_system_prompt()
            if kwargs.get("pred_audio"):
                base, ext = os.path.splitext(os.path.basename(kwargs["pred_audio"]))
                dir_path = os.path.dirname(kwargs["pred_audio"])

            processed_input = self._process_inputs(inputs)
            self.system_prompt = self.resolve_task_prompt(kwargs.get("task_prompt"))
            logger.info(f"Resolved system prompt for generation: {self.system_prompt}")

            if processed_input["type"] == "single_turn":
                # processed_input: {"type": "single_turn", "audio": audio, "query": query, "assistant": assistant_text, "instruct": instruct_text}
                results = self.generate_once(**processed_input, **kwargs)
                return results
            
            elif processed_input["type"] == "multi_turn":
                # processed_input: {"type": "multi_turn", "nrounds": nrounds, "audio": [audio1, audio2, ...], "query": [text1, text2, ...], "instruct": instruct_text}
                audio_list, text_list = processed_input["audio"], processed_input["query"]

                reverse = kwargs.get("reverse_spkr", False)
                use_model_history = kwargs.get("use_model_history", True)
                save_latest_only = kwargs.get("save_latest_only", False)

                # alternate grouping, with the default parity division user/assistant
                user_audio, assistant_audio = (audio_list[1::2], audio_list[0::2]) if reverse else (audio_list[0::2], audio_list[1::2])
                user_text, assistant_text = (text_list[1::2], text_list[0::2]) if reverse else (text_list[0::2], text_list[1::2])

                nrounds = min(len(user_audio), len(user_text), len(assistant_text))
                results, history = [], []
                user_history, assistant_history, assistant_his_audio = [], [], []
                generate_kwargs = {"instruct": processed_input.get("instruct")}
                user_infos, assistant_infos = processed_input["meta_info"][0::2], processed_input["meta_info"][1::2]

                for i in range(nrounds):
                    current_history = list(history)
                    if kwargs.get("pred_audio"):
                        generate_kwargs["pred_audio"] = os.path.join(dir_path, f"{base}_round{i + 1}{ext}")
                    generate_kwargs["user_query_history"] = user_text[:i]
                    generate_kwargs["user_query"] = user_text[i]
                    generate_kwargs["assistant_his_audio"] = assistant_his_audio
                    response = self.generate_multiturn(user_audio[i], user_history, assistant_history, **generate_kwargs)
                    
                    if not save_latest_only or i == nrounds - 1:
                        # output results
                        results.append({
                            "nround": i + 1,
                            "rounds": nrounds,
                            "pred": response.get("pred"),
                            "pred_audio": response.get("pred_audio"),
                            "query": user_text[i],
                            "ref": assistant_text[i],
                            "current_user_meta": user_infos[i] if len(user_infos[i]) > 0 else None,
                            "current_bot_meta": assistant_infos[i] if len(assistant_infos[i]) > 0 else None,
                            "history": current_history
                        })

                    history.append(
                        {
                            "user": user_text[i],
                            "bot": response["pred"],
                            "pred_audio": response.get("pred_audio"),
                            "user_meta_info": user_infos[i] if len(user_infos[i]) > 0 else None,
                            "bot_meta_info": assistant_infos[i] if len(assistant_infos[i]) > 0 else None
                        }
                    )
                    user_history.append(user_audio[i])
                    assist_his = response.get("his") if response.get("his") is not None else response["pred"]
                    assistant_history.append(assist_his if use_model_history else assistant_text[i])
                    assistant_his_audio.append(response.get("pred_audio") if use_model_history else assistant_audio[i])
                    # NOTE (TTTdas): if cache saved, then use this cache instead of history
                    #                cache only save one, and update per generation
                    if "cache" in response:
                        generate_kwargs["cache"] = response["cache"]
                return results
            else:
                raise ValueError("Unsupported processed input type.")
            
        if torch.is_grad_enabled():
            raise RuntimeError(
                "inference() should not have gradients enabled! "
                "Please check if @torch.inference_mode() is missing or misused."
            )
        # HACK (TTTdas): 应该让支持batch_decode的模型去batch推理，不支持的才在这里per batch循环
        if isinstance(inputs, list):
            return [
                _generate(p, **self._split_kwargs(kwargs, i))
                for i, p in enumerate(inputs)
            ]
        else:
            return _generate(inputs, **kwargs)

    def resolve_task_prompt(self, task_prompt_name):
        # Priority: task prompt > dataset prompt(/instruct) > model default system prompt.
        if task_prompt_name is None:
            return self._default_system_prompt
        try:
            resolved_task_prompt = get_system_prompts(task_prompt_name)
        except Exception as e:
            raise KeyError(f"Error when loading {task_prompt_name} prompt, need to be defined first!!")

        return self._default_system_prompt + resolved_task_prompt if self.stable_system_prompt else resolved_task_prompt

    def generate_once(self, audio: str, **kwargs) -> str:
        raise NotImplementedError

    def generate_multiturn(self, audio: str, user_history: List[Any], assistant_history: List[Any], **kwargs) -> str:
        raise NotImplementedError

    def _init_default_system_prompt(self):
        if self._system_prompt_initialized:
            return
        self._default_system_prompt = getattr(self, "system_prompt", None)  # load from SLM
        logger.info(f"Setting default system prompt to {self._default_system_prompt}")
        self._system_prompt_initialized = True

    def _split_kwargs(self, kwargs: dict, idx: int) -> dict:
        new_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, list):
                if idx < len(v):
                    new_kwargs[k] = v[idx]
                else:
                    raise IndexError(f"List argument '{k}' is shorter than inputs.")
            else:
                new_kwargs[k] = v  # shared
        return new_kwargs

    def _process_inputs(self, input_data: Union[List[Dict], Dict]) -> Dict:
        """
        Process input data for single-turn or multi-turn dialogue formats.
        Supports two input formats: **Single-turn format**, **Multi-turn format**
        
        Args:
            input_data (Union[List[Dict], Dict]): The input data to process. Can be a single-turn (list) or multi-turn (dict) format.

        Returns:
            Dict: A standardized dictionary representing the processed input with type, queries, audio paths, and instruction text.
        """
        def _extract_from_contents(content: Dict[str, Any]):
            audio, text = content.get("audio"), content.get("text")
            meta_info = {k: v for k, v in content.items() if k not in ["audio", "text"]}
            if audio:
                try:
                    info = sf.info(audio)
                except:
                    raise FileNotFoundError(f"Can not load file {audio}...")
                
                if self.mono_only and info.channels != 1:
                    raise ValueError("Inputs audio has more than 1 channel, need convertion!!!")
                if info.samplerate != self.inputs_sr:
                    raise ValueError(f"Inputs audio sampling rate is {info.samplerate}, not equal to {self.inputs_sr}, need resample!!!")
            return audio, text, meta_info

        def _process_single_turn(data: List[Dict]) -> Dict:
            """
            - **Single-turn format** (List[Dict]):
            A list of role-based utterances, including system, instruction, user query (with optional audio), and assistant reply.
            Example:
                [
                    {"role": "system", "content": {"text": system_prompt}},
                    {"role": "instruct", "content": {"text": instruct_text}},
                    {"role": "user", "content": {"audio": audio_path, "text": query}},
                    {"role": "assistant", "content": {"text": assistant_text}}
                ]
            
            Returns:
                {
                    "type": "single_turn",
                    "system": system_prompt,
                    "instruct": instruct_text,
                    "query": query,
                    "audio": audio_path,
                    "assistant": assistant_text
                }
            """
            result = {"type": "single_turn", "audio": None, "query": None, "assistant": None, "instruct": "", "system": None}
            for info in data:
                role = info.get("role", "").lower()
                audio, text, meta_info = _extract_from_contents(info.get("content", {}))
                if role == "user":
                    result["audio"], result["query"] = audio, text

                elif role in ("assistant", "instruct", "system"):
                    result[role] = text
            
            if result["audio"] is None and result["query"] is None:
                raise ValueError("Single-turn prompt must contain 'user' role with audio or text.")
            return result

        def _process_multi_turn(data: Dict) -> Dict:
            """
            - **Multi-turn format** (Dict):
            A dict containing dialogue history and number of rounds, with each utterance annotated by role and round number.
            Example:
                {
                    "nrounds": "2",
                    "dialogue": [
                        {"role": "A", "round": "1", "content": {"audio": audio1, "text": text1, "meta": xxx}},
                        {"role": "B", "round": "1", "content": {"audio": audio2/None, "text": text2, "meta": xxx}},
                        ...
                    ],
                    "instruct": {"content": {"text": instruct_text}}
                }

            Returns:
                {
                    "type": "multi_turn",
                    "nrounds": 2,
                    "audio": [audio1, audio2, ...],
                    "query": [text1, text2, ...],
                    "instruct": instruct_text
                }
            """
            dialogue = data.get("dialogue", [])
            if not dialogue:
                raise ValueError("Multi-turn dialogue is empty.")
            audio_list, text_list, meta_info_list = zip(*[
                _extract_from_contents(turn.get("content", {})) for turn in dialogue
            ])
            
            if "instruct" in data:
                instruct_text = _extract_from_contents(data["instruct"].get("content", {}))[1]
            else:
                instruct_text = ""
            return {
                "type": "multi_turn",
                "nrounds": data.get("nrounds"),
                "audio": list(audio_list),
                "query": list(text_list),
                "instruct": instruct_text,
                "meta_info": list(meta_info_list)
            }
        if isinstance(input_data, dict) and "dialogue" in input_data:
            return _process_multi_turn(input_data)
        elif isinstance(input_data, list):
            return _process_single_turn(input_data)
        else:
            raise ValueError("Input data format is not recognized.")
