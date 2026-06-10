## Defining Your Own Configuration

Televal supports the input types required for SLMs inference, including pure audio input, audio + text instruction input (**NOT RECOMMEND**), and pure text input (for LLM model inference). For SLMs, we recommend using audio-only input to simulate end-to-end dialogue scenarios.

All configurations live under **`registry/*/*.yaml`**. Each YAML entry follows the same format:

```yaml
<name>:                          # must be unique across the same registry subdirectory
  class: <module_path>.<Class>   # fully-qualified class path
  args:                           # kwargs passed to the class constructor
    key1: value1
```

---

### Infer Task

An `infer_task` ties together a dataset, template, model, and eval task. It is the single entry point for running both inference and evaluation.

```yaml
# registry/infer_task/my_task.yaml
my_task:
  class: src.config.InferTaskCfg
  args:
    dataset:                         # str or list of dataset names
      - my_dataset_name1
      - my_dataset_name2      
    template: zeroshot-aqa           # template name from registry/template/
    model: my_model                  # default model, overridable via --model
    eval_task:                       # str, list, or dict of eval task names
      - my_eval_task
    save_pred_audio: False           # set True if you need to save the response audio
    task_prompt: chitchat_prompt     # (optional) override model's system prompt
    # --- multi-turn only ---
    # reverse_spkr: False           # swap A/B speaker order
    # use_model_history: True       # use model output (vs. ground truth) as history
    # save_latest_only: False       # only save the final turn's output
```

**Key points:**
- `eval_task` can be a list — the same prediction file will be evaluated by multiple evaluators.
- `task_prompt` replaces the model's default system prompt (unless the model sets `stable_system_prompt=True`, in which case it is appended).
- Command-line flags (`--model`, `--eval_task`, `--save_pred_audio`, `--bsz`) override the YAML values.

---

### Dataset

The `BatchLoader` supports three data sources:

| Source | `file` format | Notes |
|--------|---------------|-------|
| Local JSONL | `/path/to/data.jsonl` | Directly loaded; audio paths must be valid local paths |
| HuggingFace | `org/repo/subdir` | Auto-downloaded; audio bytes decoded to temporary WAV files |
| Local Parquet | `/path/to/parquet_dir/` | Loaded without HF download |

```yaml
# registry/dataset/my_dataset.yaml
my_dataset:
  class: src.dataset.BatchLoader
  args:
    file: path/to/data.jsonl
    key_col: key                      # unique sample identifier column
    ref_col: answer                   # reference answer column
    query_col: query                  # question column (for logging)
    meta_col: ["query", "answer"]     # extra fields saved to prediction file
    batch_size: 1                     # use 1 for multi-turn to avoid OOM
    tuple_decode: False
    # save_query_audio_dir: audios/   # decode & persist HF audio (optional)
```

**Tip:** Use `tools/parquet2jsonl.py` to convert a HuggingFace Parquet dataset to local JSONL + WAV for faster repeated access.

---

### Model

Implement a class under `src/models/` inheriting from `Model`, then register it:

```python
# src/models/my_model.py
from src.models.base import Model

class MyModel(Model):
    def __init__(self, path, sample_params=None, **kwargs):
        super().__init__(sample_params)
        # load model weights ...

    def generate_once(self, audio, query=None, instruct=None, **kwargs):
        # single-turn inference
        # return {"pred": str} or {"pred": str, "pred_audio": "/path.wav"}
        return {"pred": "response"}

    def generate_multiturn(self, audio, user_history, assistant_history,
                           user_query=None, instruct=None, **kwargs):
        # multi-turn inference
        # return {"pred": str} or {"pred": str, "pred_audio": str, "his": str, "cache": str}
        return {"pred": "response"}
```

```yaml
# registry/model/my_model.yaml
my_model:
  class: src.models.my_model.MyModel
  args:
    path: /path/to/checkpoint
    sample_params:
      gen_type: greedy            # "greedy" or "default"
```

* **`cache`** and **`his`** are optional. If both are missing, the model uses default history (usually the last output). 
  * **`cache`** holds temporary states updated each turn (e.g. kv_cache or token_ids). 
  * **`his`** is for output history. Use this if the model requires a different history format than plain output text. It will be accumulated into **`assistant_history`**.
* After implementing a custom model, just add a new model config. The parameters in `args` can be customized as needed and should match the `__init__` method of your model class.
* **Multi-GPU:** If GPU memory is limited, use `load_model_with_auto_device_map` from `model_utils.py` to split the model across devices. See `kimi_audio.py` for an example.


---

### Evaluator

Implement a class under `src/evaluator/` inheriting from `Evaluator`:

```python
# src/evaluator/my_eval.py
from src.evaluator.base import Evaluator
from src.utils import parallel_batch

class MyEvaluator(Evaluator):
    REQUIRED_FIELDS = {"prediction", "reference"}  # this can be changed

    def __init__(self, my_param=42, max_workers=None):
        self.my_param = my_param
        if max_workers is not None:
            self.max_workers = max_workers  # read by @parallel_batch

    @parallel_batch(default_workers=4)
    def evaluate(self, pred_info, fields, **kwargs):
        f = self.get_fields(fields)         # merge DEFAULT_FIELDS + override
        pred = pred_info[f["prediction"]]  # following REQUIRED_FIELDS
        ref  = pred_info[f["reference"]]
        return {"key": pred_info[f["key"]], "score": self._compute(pred, ref)}
```

```yaml
# registry/evaluator/my_eval.yaml
my_evaluator:
  class: src.evaluator.my_eval.MyEvaluator
  args:
    my_param: xxx
    max_workers: 8
```

**`@parallel_batch`** handles multithreading automatically — write your `evaluate` for a single sample and the decorator parallelizes it.

**Field mapping:** `Evaluator.DEFAULT_FIELDS` maps internal names (`prediction`, `reference`, `pred_audio`) to data column names (`pred`, `ref`, `pred_audio`). Override via `eval_task.fields` if your data uses different column names.

---

### LLM-as-Judge Builder & Parser

For LLM-based evaluation, if you don't need a new Evaluator class, just register a **builder** and **parser**:

```python
# src/evaluator/llm/handlers_impl.py

@LLMHandlerRegistry.register_builder("my_builder")
def my_builder(pred_info):
    """Preprocess pred_info into kwargs for the judge prompt template."""
    return {"pred": pred_info["pred"], "context": pred_info.get("history", "")}

@LLMHandlerRegistry.register_parser("my_parser")
def my_parser(data: dict):
    """Extract structured result from the LLM judge's JSON output."""
    return {"score": int(data["Score"]), "reason": data.get("Explanation", "")}
```

Then use `LLMAPIScorer` (online) or `QwenScorer`/`Qwen3OmniScorer` (offline) in the evaluator YAML:

```yaml
# Online (GPT-4o API)
my_llm_eval:
  class: src.evaluator.llm.llm_api.LLMAPIScorer
  args:
    llm_name: gpt4o
    judge_task: my_judge_task          # key in TASK_PROMPT_MAP (src/prompt/llm_judge.py)
    builder_name: my_builder
    parser_name: my_parser
    api_keys: {key1: "sk-xxx"}

# Offline (local Qwen3Omni, supports audio input)
my_offline_eval:
  class: src.evaluator.llm.qwen_scorer.Qwen3OmniScorer
  args:
    llm_name: qwen3_omni
    judge_task: my_judge_task
    builder_name: my_builder
    parser_name: my_parser
    generate_params:
      ngpus: 4
      max_tokens: 8192
```

**Register the prompt** in `src/prompt/llm_judge.py` by adding an entry to `TASK_PROMPT_MAP`.

---

### Template

Templates are Jinja2-powered data processors defined in YAML:

```yaml
# Single-turn
zeroshot-aqa:
  class: src.prompt.template.DataTemplate
  args:
    template:
      - role: user
        content:
          audio: "{{audio}}"
          text: "{{query}}"

# Multi-turn with emotion metadata
multiturn_emo-audio:
  class: src.prompt.template.DataTemplate
  args:
    template: |
      {
        "nrounds": {{ nrounds }},
        "dialogue": [
          {% for i in range(1, nrounds + 1) %}
          {
            "role": "A", "round": "{{ i }}",
            "content": {
              "audio": {{ getvar("user_audio" ~ i) | tojson }},
              "text": {{ getvar("user_text" ~ i) | tojson }},
              "user_emo": {{ getvar("user_emo" ~ i) | tojson }}
            }
          },
          {
            "role": "B", "round": "{{ i }}",
            "content": {
              "audio": {{ getvar("bot_audio" ~ i) | tojson }},
              "text": {{ getvar("bot_text" ~ i) | tojson }}
            }
          }{% if not loop.last %},{% endif %}
          {% endfor %}
        ]
      }
```

Variables (`{{audio}}`, `{{query}}`, etc.) come from data columns. Use `getvar("field" ~ i)` for dynamic field names in multi-turn templates.

---

### Summarizer

Summarizers aggregate per-sample scores into a final metric. Register in `registry/summarizer/base.yaml`:

| Class | Use case |
|-------|----------|
| `Avg` | Simple average of single-value scores |
| `AvgInfo` | Average of dict-type scores, per-key |
| `AvgThreshold` | Average + percentage above a threshold |
| `AvgWER` | WER/CER aggregation (100 - WER) |
| `AvgMOS` | DNSMOS score average |
| `AvgHumanDialChallenge` | Per-dimension average + macro average for multi-dim scores |

To implement a custom summarizer:

```python
# In src/summarizer/summarizer.py
class MySummarizer(Summarizer):
    def statistic(self, scores, **kwargs):
        values = [self.rescale_func(float(s)) for s in scores]
        return {"score": f"MY_METRIC: {sum(values)/len(values):.2f}"}
```

```yaml
# registry/summarizer/base.yaml
my_summarizer:
  class: src.summarizer.summarizer.MySummarizer
  args:
    rescale: power       # "base" (no rescale) or "power"
    power: 2             # exponent for power rescaling
    score_scale: 5       # max raw score, used in 100*(s/scale)^power
```

---

### Eval Task

An eval task pairs an evaluator with a summarizer:

```yaml
# registry/eval_task/my_eval_task.yaml
my_eval_task:
  class: src.config.EvalTaskCfg
  args:
    evaluator: my_evaluator
    summarizer: my_summarizer
    fields:                           # (optional) override field mapping
      prediction: pred
      reference: ref
      pred_audio: pred_audio
    batch_size: 32                    # (optional) eval-time batch size
```

---

## Example: Human-like Spoken Dialogue Challenge

This example shows how to integrate the [Human-like Spoken Dialogue Systems Challenge (HumanDial)](https://github.com/ASLP-lab/Hum-Dial) — a multi-turn emotional dialogue benchmark — into the framework. It demonstrates the full pipeline: multi-turn dataset, custom template, LLM-as-judge evaluation with Builder/Parser, and a dedicated Summarizer.

### What HumanDial evaluates

HumanDial defines three tasks:

| Task | Name | What is evaluated | Input | Output modality |
|------|------|-------------------|-------|-----------------|
| 1 | Emotion Trajectory | Can the model summarize the user's emotional journey? | Multi-turn dialogue | Text |
| 2 | Emotion Reasoning | Can the model reason about underlying causes of emotion? | Multi-turn dialogue | Text |
| 3 | Emotion Response | Does the model respond with appropriate vocal empathy? | Multi-turn dialogue | Audio + Text |

### Step 1 — Dataset

Prepare JSONL files where each line is a multi-turn dialogue. The framework expects per-turn fields like `user_audio1`, `user_text1`, `user_emo1`, `bot_audio1`, `bot_text1`, etc.

```yaml
# registry/dataset/human_dial.yaml
task1-zh:
  class: src.dataset.BatchLoader
  args:
    file: /path/to/HumDial/dev_zh/task1.jsonl
    batch_size: 1                   # use 1 for multi-turn (avoid OOM)
    tuple_decode: False

task3-zh:
  class: src.dataset.BatchLoader
  args:
    file: /path/to/HumDial/dev_zh/task3.jsonl
    batch_size: 1
    tuple_decode: False
```

### Step 2 — Template

The template renders each turn with user emotion metadata. Note the `user_emo` field carried per turn:

```yaml
# registry/template/multiturn.yaml (excerpt)
multiturn_emo-audio:
  class: src.prompt.template.DataTemplate
  args:
    template: |
      {
        "nrounds": {{ nrounds }},
        "dialogue": [
          {% for i in range(1, nrounds + 1) %}
          {
            "role": "A", "round": "{{ i }}",
            "content": {
              "audio": {{ getvar("user_audio" ~ i) | tojson }},
              "text": {{ getvar("user_text" ~ i) | tojson }},
              "user_emo": {{ getvar("user_emo" ~ i) | tojson }}
            }
          },
          {
            "role": "B", "round": "{{ i }}",
            "content": {
              "audio": {{ getvar("bot_audio" ~ i) | tojson }},
              "text": {{ getvar("bot_text" ~ i) | tojson }}
            }
          }{% if not loop.last %},{% endif %}
          {% endfor %}
        ]
      }
```

### Step 3 — Infer Task

Each HumanDial task needs its own infer task. Task 3 requires `save_pred_audio: True` because the evaluator judges the generated audio.

```yaml
# registry/infer_task/HumanDial.yaml
humandial_emo_change-zh:               # Task 1
  class: src.config.InferTaskCfg
  args:
    dataset: task1-zh
    template: multiturn_emo-audio
    model: qwen2_5_omni
    eval_task: humandial_task1
    save_pred_audio: False
    reverse_spkr: False
    use_model_history: True
    save_latest_only: True

humandial_emo_response-zh:             # Task 3
  class: src.config.InferTaskCfg
  args:
    dataset: task3-zh
    template: multiturn_emo-audio
    model: qwen2_5_omni
    eval_task: humandial_task3
    save_pred_audio: True              # Task 3 needs audio for evaluation
    reverse_spkr: False
    use_model_history: True
    save_latest_only: True
```

### Step 4 — LLM Judge Prompt

Add a prompt rendering function in `src/prompt/llm_judge.py`:

```python
# src/prompt/llm_judge.py (excerpt)

PROMPT_HUMANDIAL_TASK1 = """You are a dialogue analyst. Evaluate the AI's ability
to summarize the user's emotional trajectory across the conversation.

Conversation history:
{conversation_history}

Current user question: {user_question}
AI final response: {final_model_response}

Rate on 3 dimensions (1-5 scale):
- Accuracy_Completeness: Are all emotion tags precisely matched?
- Depth_Granularity: Are emotional dynamics vividly depicted?
- Added_Value: Are emotions linked to concrete events?

Output JSON: {{"scores": {{"Accuracy_Completeness": X, "Depth_Granularity": X, "Added_Value": X}},
"justification": {{...}}, "overall_comment": "..."}}"""

TASK_PROMPT_MAP = {
    # ...
    "humandial_emotion_trajectory": lambda **kwargs: PROMPT_HUMANDIAL_TASK1.format(
        user_question=kwargs["query"],
        final_model_response=kwargs["pred"],
        conversation_history=kwargs["history"],
    ),
}
```

### Step 5 — Builder & Parser

Register preprocessing and scoring logic in `src/evaluator/llm/handlers_impl.py`:

```python
# Shared builder for Task 1 & 2 (text-only evaluation)
@LLMHandlerRegistry.register_builder("humandial_emotion_preprocess")
def humandial_emotion_preprocess(pred_info):
    turns = pred_info["history"]
    conversation = [
        f"User(emotion:{t['user_meta_info']['user_emo']}): {t['user']}\nAI: {t['bot']}"
        for t in turns
    ]
    return {"query": pred_info["query"], "history": "\n".join(conversation)}

# Task 3 builder (locates the target turn and extracts pred_audio)
@LLMHandlerRegistry.register_builder("humandial_task3_preprocess")
def humandial_task3_preprocess(pred_info):
    # Find the 2nd turn where user emotion is not "neutral"
    # Build context up to that turn, and pass the model's pred_audio for vocal evaluation
    turns = pred_info["history"]
    # ... locate target turn and build context ...
    return {"pred_audio": target_turn.get("pred_audio"), "history": conversation}

# Task 1 parser: map scores to {1, 3, 5}
@LLMHandlerRegistry.register_parser("humandial_task1_eval")
def humandial_task1_parser(data: dict):
    scores = data.get("scores", {})
    for k in ["Accuracy_Completeness", "Depth_Granularity", "Added_Value"]:
        v = int(scores.get(k, 1))
        scores[k] = 1 if v <= 2 else (3 if v <= 4 else 5)
    return {"score": scores, "meta": {"avg_score": sum(scores.values())/len(scores)}}

# Task 3 parser: clamp scores to 1-5 integers
@LLMHandlerRegistry.register_parser("humandial_task3_eval")
def humandial_task3_parser(data: dict):
    scores = data.get("scores", {})
    for k in ["textual_empathy_insight", "vocal_empathy_congruence", "audio_quality_naturalness"]:
        scores[k] = max(1, min(5, int(scores.get(k, 1))))
    return {"score": scores, "meta": {"avg_score": sum(scores.values())/len(scores)}}
```

### Step 6 — Evaluator

**Task 3 must use `Qwen3OmniScorer`** because it needs to listen to the generated audio.

```yaml

# registry/evaluator/llm_offline.yaml
humandial_task3_llm_offline:                  # Task 3: audio+text, offline
  class: src.evaluator.llm.qwen_scorer.Qwen3OmniScorer
  args:
    llm_name: qwen3_omni
    judge_task: humandial_emotion_response
    builder_name: humandial_task3_preprocess
    parser_name: humandial_task3_eval
    generate_params:
      ngpus: 4
      max_tokens: 16384
```

### Step 7 — Summarizer

`AvgHumanDialChallenge` computes per-dimension averages plus a macro average:

```python
# src/summarizer/summarizer.py
class AvgHumanDialChallenge(Summarizer):
    def statistic(self, scores, **kwargs):
        keys = scores[0].keys()
        result = {}
        for key in keys:
            values = [float(s[key]) for s in scores if key in s]
            result[key] = f"{sum(values)/len(values):.2f}"
        avg_all = sum(float(v) for v in result.values()) / len(keys)
        result["avg_all_score"] = f"{avg_all:.2f}"
        return result
```

```yaml
# registry/summarizer/base.yaml
AvgHumanDialChallenge:
  class: src.summarizer.summarizer.AvgHumanDialChallenge
  args: {}
```

### Step 8 — Eval Task

Wire evaluator and summarizer together:

```yaml
# registry/eval_task/llm.yaml
humandial_task1:
  class: src.config.EvalTaskCfg
  args:
    evaluator: humandial_task1_llm
    summarizer: AvgHumanDialChallenge
    batch_size: 150

humandial_task3:
  class: src.config.EvalTaskCfg
  args:
    evaluator: humandial_task3_llm_offline     # must use Omni scorer for audio
    summarizer: AvgHumanDialChallenge
    batch_size: 150
```

### Run

```bash
# Inference
python main.py --mode infer --task humandial_emo_change-zh --model my_model

# Evaluation
python main.py --mode eval --task humandial_emo_change-zh

# Override the model at runtime
python main.py --mode infer --task humandial_emo_response-zh --model another_model
```

### Data flow summary

```
JSONL dataset
  → Template (multiturn_emo-audio) renders dialogue with per-turn emotion tags
    → Model.inference() runs turn-by-turn, accumulates history with meta_info
      → Predictions saved (text + optionally audio)
        → Builder preprocesses history for the judge prompt
          → LLM judge scores on multiple dimensions
            → Parser extracts structured scores
              → AvgHumanDialChallenge computes per-dim + macro averages
```
