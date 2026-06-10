<p align="left">
        <a href="README.md">中文</a> &nbsp｜ &nbsp English&nbsp&nbsp
</p>
<br>

<div align="center">
<h1>
  TELEVAL
</h1>
</div>

<p align="center">
🤗 <a href="https://huggingface.co/datasets/Tele-AI/TeleSpeech-AudioBench" target="_blank">HuggingFace Dataset</a>️ • 
🤖 <a href="https://modelscope.cn/datasets/TeleAI/TELEVAL/files" target="_blank">ModelScope</a> • 
📃 <a href="https://arxiv.org/abs/2507.18061" target="_blank">Technical Report</a>
</p>

## Updates
- [Update Jun. 10, 2026] 🔥 Update v2.0
- [Update Jul. 25, 2025] Technical report updated
- [Update Jun. 5, 2025] Evaluation code and datasets released

## Introduction

**TELEVAL** is an evaluation benchmark for spoken-language models (SLMs) that decomposes spoken interaction ability into three levels:
- **Perceptual Robustness**: the ability to reliably capture and process user speech signals;
- **Explicit Semantic Reasoning**: the ability to correctly understand user intent and generate semantically accurate and factually grounded responses;
- **Social-Pragmatic Alignment**: the ability to behave in ways consistent with human conversational norms and to adjust response strategies based on implicit interactional cues.

Beyond assessing whether a model correctly fulfills user intent (Reliable Content Fulfillment) and produces high-quality outputs, TELEVAL places strong emphasis on Interactional Appropriateness. In particular, it evaluates whether models can generate spoken, non-templated responses and implicitly leverage paralinguistic information in speech, such as emotional states, age-related cues, and other non-verbal signals, to guide interactional decisions. Rather than testing a model’s ability to perform explicit classification or label prediction of acoustic attributes under predefined system prompts, TELEVAL directly evaluates whether such paralinguistic information is implicitly perceived and appropriately reflected in the model’s natural conversational responses.

- **Multi-dimensional Evaluation 🧠**: Covers 12 tasks across 34 datasets, with more continuously added.
- **Real-world Interaction Testing 🎧**: Designed around practical spoken interaction needs, such as question answering and companion-style dialogue. The benchmark avoids artificial or information-leaking prompts (e.g., “I am a child, what should I do…” or “What emotion am I feeling?”), and focuses on natural conversational behavior.
- **Multilingual & Dialect-rich Data 🌏**: Primarily based on Mandarin Chinese, with additional coverage of English Q&A and multiple Chinese dialects (e.g., Cantonese, Henan, Northeastern, Shanghainese, Sichuanese).
- **Modular Evaluation Framework 🔧**: Provides a complete pipeline for model inference and result evaluation, with decoupled inference and scoring stages, enabling reuse of existing model outputs and easy customization of models, tasks, and datasets.

## Current SLMs and Leaderboard
| Rank | Model | Average Score (%)  |
|:--:|:-----:|:-------:|
| 🥇 | [Qwen3-Omni](https://github.com/QwenLM/Qwen3-Omni) | 53.46 |
| 🥈 | [StepAudio2-mini](https://github.com/stepfun-ai/Step-Audio2) | 46.64 |
| 🥉 | [Mimo-Audio-Instruct](https://github.com/XiaomiMiMo/MiMo-Audio) | 46.10 |
| #4 | GPT4o-Audio (2024-12-17 preview) | 45.46 |
| #5 | [Qwen2.5-Omni](https://github.com/QwenLM/Qwen2.5-Omni) | 42.51 |
| #6 | [Kimi-Audio](https://github.com/MoonshotAI/Kimi-Audio) | 38.82 |
| #7 | [MiniCPM-o-2.6](https://github.com/OpenBMB/MiniCPM-o) | 37.40 |
| #8 | [Baichuan-Omni-1.5](https://github.com/baichuan-inc/Baichuan-Omni-1.5) | 36.90 |
| #9 | [Freeze-Omni](https://github.com/VITA-MLLM/Freeze-Omni) | 33.19 |
| #10 | [GLM-4-Voice](https://github.com/THUDM/GLM-4-Voice) | 31.87 |
| #11 | [LLaMA-Omni2](https://github.com/ictnlp/LLaMA-Omni2) | 24.67 |
| #12 | [SpeechGPT-2.0-preview](https://github.com/OpenMOSS/SpeechGPT-2.0-preview) | 14.49 |

## Environment
```bash
python -m venv televal-env
source televal-env/bin/activate

# evaluation only
pip install -r requirements_eval.txt

# Install dependencies for inference & evaluation
pip install -r requirements_all.txt
```

We provide a unified environment in `requirements_all.txt` that includes dependencies for all supported models.  
However, some models like `qwen2.5-omni` and `kimi-audio` require a higher version of `transformers`. For these models, it is recommended to install the corresponding version of `transformers` separately, as specified in `requirements_all.txt`.

## Usage

### Stage 0: Dataset Preparation (Optional)

The framework supports loading datasets from HuggingFace and local (Parquet format) or local JSONL files. Due to network limitations and large dataset sizes, we strongly recommend downloading and converting datasets to `jsonl + wav` format beforehand for repeated use.  

The ```parquet2jsonl.py``` provides multiple usage combinations that can automatically handle dataset downloading and preprocessing, converting datasets into JSONL and WAV formats for easier use.
```bash
# set $save_root_dir and choose a usage mode, then running:
python tools/parquet2jsonl.py
```

To use your **own dataset**, refer to [Custom Dataset](assets/custom.md#自定义dataset) for how to add and test custom data.

### Stage 1: Inference (Optional)

Download the model you want to use for inference and set its path in `registry/model/offline.yaml`.

Tasks are configured via YAML files under `registry/infer_task`. Once the corresponding `*.yaml` file is ready, you can quickly run:
```bash
export PYTHONPATH=$PWD:$PYTHONPATH
python main.py --mode "infer" --task "aqa-llamaqa-zh"
```

(**Strongly recommended**) You can also use the ```run.sh``` script to perform automatic inference across multiple tasks and models. Simply modify the parameters in ```run.sh``` and run:
```bash
bash run.sh  # stage=1
```

### Stage 2: Evaluation
If inference results already exist, you can run the following script to get scores for a given eval_task.
**You can also run scoring directly via ```run.sh``` for a one-stop evaluation.**

The framework also supports evaluating external results (without running Stage 1). Just ensure the model predictions are saved at ```${save_dir}/prediction/${model}/${infer_task}.jsonl```. Each line in the JSONL file must include at least the fields: ```key, pred, ref``` (custom fields are also supported). Then run the same evaluation script as above.

### Directory Structure

Model predictions and evaluation results are automatically saved in the following structure:
```
$save_dir
    ├── prediction
    │   └── $model
    │     └── $infer_task
    │       └── ${dataset}.jsonl
    ├── result
    │   └── $model
    │     └── $infer_task
    │       └── ${dataset}.${eval_task}.jsonl
    ├── summary
    │   └── $model
    │     └── $infer_task
    │       └── ${dataset}.${eval_task}.jsonl
    └── results.csv
```

## Supported Tasks

Currently supports 34 main datasets (98 sub-datasets). For full details, see [assets/task.md](assets/task.md).

## Dataset Information
Dataset details and their corresponding evaluation abilities can be found in [assets/dataset.md](assets/dataset.md#Dataset_Information).

## Results of Open-source Models
Key results are shown in the table below:

| **Model**                 | **Basic Knowledge** (%) | **Dialect Comprehension** (%) | **Safety&Morality** (%) | **Humanlike Chitchat** (%) | **Livelihood Policy** (%) | **Multiturn Dialogue** (%) | **Dialect-Aware Response** (%) | **Empathetic Response** (%) | **Age-Aware Response** (%) | **NSV-Aware Response** (%) | **Scene** (%) | **Acoustic Robustness** (%) | **Speech-Text Consistency** (%) | **Response Quality (Speech)** (⬆) | **Empathetic Response (Speech)** (%) |
|---------------------------|---------------------|---------------------------|-----------------------|------------------------|-----------------------|------------------------|----------------------------|-------------------------|------------------------|------------------------|-----------|-------------------------|-----------------------------|-------------------------------|-----------------------------------|
| **GPT4o-Audio (API)**     | 52.93               | 21.15                     | 96.29                 | 34.45                  | 16.39                 | 84.00                  | 9.19                       | 35.28                   | 17.65                  | 2.52                   | 8.01      | 38.79                   | 98.06                       | 3.46                          | 24.09                             |
| **GLM-4-Voice**           | 31.55               | 13.13                     | 92.55                 | 59.50                  | 16.84                 | 80.00                  | 4.57                       | 35.55                   | 27.81                  | 1.89                   | 0.75      | 32.88                   | 94.45                       | 3.38                          | 34.32                             |
| **MiniCPM-o-2.6**         | 36.16               | 16.67                     | 87.60                 | 58.29                  | 19.78                 | 86.67                  | 10.98                      | 44.03                   | 34.56                  | 2.08                   | 8.91      | 36.18                   | 95.74                       | 3.48                          | 27.90                             |
| **Baichuan-Omni-1.5**     | 34.84               | 30.68                     | 95.00                 | 26.26                  | 19.91                 | 78.67                  | 7.38                       | 13.55                   | 12.24                  | 1.80                   | 1.48      | 42.97                   | 91.31                       | 3.40                          | 23.66                             |
| **LLaMA-Omni2**           | 24.89               | 7.79                      | 77.97                 | 20.77                  | 14.27                 | 54.00                  | 4.26                       | 21.12                   | 13.12                  | 1.77                   | 0.56      | 28.24                   | 98.22                       | 3.49                          | 26.21                             |
| **SpeechGPT-2.0-preview** | 9.88                | 4.98                      | 76.41                 | 41.22                  | 10.38                 | 20.00                  | 5.17                       | 22.59                   | 23.63                  | 1.52                   | 0.27      | 10.70                   | 83.34                       | 2.45                          | 27.78                             |
| **Freeze-Omni**           | 33.05               | 16.44                     | 87.57                 | 30.90                  | 16.64                 | 62.67                  | 5.72                       | 20.72                   | 13.68                  | 1.85                   | 9.15      | 30.48                   | 98.14                       | 3.48                          | 38.87                             |
| **Qwen2.5-Omni**          | 34.77               | 40.54                     | 82.93                 | 80.89                  | 17.89                 | 88.67                  | 18.91                      | 44.83                   | 42.51                  | 2.19                   | 18.90     | 42.79                   | 98.83                       | 3.46                          | 51.71                             |
| **Kimi-Audio**            | 37.18               | 25.71                     | 86.67                 | 47.95                  | 13.45                 | 84.87                  | 10.18                      | 53.17                   | 22.77                  | 9.19                   | 22.01     | 45.30                   | 96.73                       | 3.40                          | 46.25                             |
| **StepAudio2-mini**       | 38.96               | 45.45                     | 91.93                 | 29.25                  | 23.18                 | 82.67                  | 40.12                      | 16.43                   | 18.77                  | 1.97                   | 16.42     | 42.79                   | 94.31                       | 3.22                          | 38.60                             |
| **Qwen3-Omni**            | 50.52               | 41.52                     | 90.11                 | 73.45                  | 22.31                 | 92.67                  | 32.82                      | 44.03                   | 26.43                  | 2.52                   | 18.53     | 50.24                   | 97.86                       | 3.48                          | 48.26                             |
| **Mimo-Audio-Instruct**   | 46.11               | 36.57                     | 99.36                 | 29.27                  | 19.89                 | 88.00                  | 23.74                      | 16.43                   | 11.55                  | 1.87                   | 15.04     | 56.97                   | 31.61                       | 1.80                          | 26.69                             |

* Basic Knowledge, Dialect Comprehension, and Dialect-Aware Response are weighted averages across multiple datasets. Acoustic Robustness is the average of the worst-case performance under each acoustic condition.  
* Since most open-source models do not support "dialect audio generation without explicit instructions," this part is excluded from the table.
* Results by dimension can be found in [assets/result.md](assets/result.md#results).  
  For more experiments and analysis, see <a href="https://arxiv.org/abs/2507.18061" target="_blank">Technical Report</a>.

## Define Your Own Dataset and Model
This framework provides a complete pipeline for model inference and evaluation. It supports flexible definitions of tasks, datasets, and models by modifying the corresponding config files under `registry`.  

To add a new model, simply inherit the **`Model`** class and implement the **`generate_once`** and **`generate_multiturn`** methods. See [assets/custom.md](assets/custom.md) for more details.

## Acknowledgements

* Parts of the code in this framework are referenced and adapted from [UltraEval-Audio](https://github.com/OpenBMB/UltraEval-Audio) and [OpenCompass](https://github.com/open-compass/opencompass).
* The audio for datasets `llamaqa-en`, `triviaqa-en`, and `webq-en` comes from [https://huggingface.co/TwinkStart](https://huggingface.co/TwinkStart). We manually filtered these datasets to remove unsuitable QA samples and corrected the answers, so the total number of samples is smaller than the original datasets.
* The inference implementations for each SLM are based on demo scripts from their respective open-source projects. We restructured them to integrate seamlessly into the TELEVAL framework. To ensure all models support *greedy_search* inference, we also adjusted some model codes, for example in `src_freezeomni/audioLLM.py`.

## Citation
If you find this project useful, a star ⭐ and citation would be greatly appreciated —— thank you for your support!
```bibtex
@article{li2025televal,
  title={TELEVAL: A Dynamic Benchmark Designed for Spoken Language Models in Chinese Interactive Scenarios},
  author={Zehan Li and Hongjie Chen and Qing Wang and Yuxin Zhang and Jing Zhou and Hang Lv and Mengjie Du and Yaodong Song and Jie Lian and Jian Kang and Jie Li and Yongxiang Li and Xuelong Li},
  journal={arXiv preprint arXiv:2507.18061},
  year={2025}
}
```