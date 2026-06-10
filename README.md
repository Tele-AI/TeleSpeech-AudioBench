<p align="left">
        中文</a>&nbsp ｜ &nbsp<a href="README_EN.md">English</a>
</p>
<br>

<div align="center">
<h1>
  TELEVAL
</h1>
</div>

<p align="center">
🤗 <a href="https://huggingface.co/datasets/Tele-AI/TELEVAL" target="_blank">HuggingFace Dataset</a>️ • 
🤖 <a href="https://modelscope.cn/datasets/TeleAI/TELEVAL/files" target="_blank">ModelScope</a> • 
📃 <a href="https://arxiv.org/abs/2507.18061" target="_blank">Technical Report</a>
</p>

## 更新
- [Update Jun. 10, 2026] 🔥 框架更新
- [Update Jul. 25, 2025] 技术报告已更新
- [Update Jun. 5, 2025] 测评代码与数据均已开放

## 简介

**TELEVAL** 是一个面向语音对话大模型（Spoken-Language Models, SLMs）的评测基准，将模型的语音交互能力拆解为三个层次：
- **感知鲁棒性（Perceptual Robustness）**：准确接收用户的声音信号；
- **显示语义推断（Explicit Semantic Reasoning）**：正确理解用户意图，并生成语义正确、事实可靠的回应；
- **社交-语用一致性（Social-Pragmatic Alignment）**：在对话中表现出符合人类交互习惯的行为，并根据隐含互动线索调整回应策略。

TELEVAL不仅衡量模型是否正确完成用户意图（Reliable Content Fulfillment）与生成质量，更强调模型能否生成口语化、非模板化的回应，并能够隐式利用副语言信息（如情绪、年龄线索、非语言信号）来支撑交互决策（Interactional Appropriateness）。相比在特定 system prompt 下对声学信息进行显式分类或标签预测的评测，TELEVAL直接评估模型在对话回应中，是否隐式地感知并合理地利用了这些副语言信息。

- **多维实用性评估 🧠**：覆盖12大任务34个数据集，包含基础知识、方言理解与回应、基于副语言信息的回应等多个任务与测评能力，数据持续扩充中。
- **真实交互测试 🎧**：结合实际交互需求（如知识问答、拟人陪伴等），避免人工化或信息泄露式指令如“我是个小孩子，我应该...”、“我现在是什么心情？” ，全面考察模型对用户语音的自然对话能力。
- **多语种与多方言数据支持 🌏**：评测数据以中文普通话为主，同时涵盖英文问答与多种中文方言（如粤语、河南话、东北话、上海话、四川话等）。
- **模块化评测框架 🔧**：完整的模型推理与结果评估框架，推理与评估流程解耦，便于自定义模型、任务与数据集。

## 已有的模型与综合得分
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

## 环境准备
```bash
python -m venv televal-env
source televal-env/bin/activate

# evaluation only
pip install -r requirements_eval.txt

# Install dependencies for inference & evaluation
pip install -r requirements_all.txt
```

在```requirements_all.txt```中我们提供了一个综合的环境，满足各个模型的版本依赖。但是，一些模型如```qwen2.5-omni```和```kimi-audio```要求的```transformers```版本较高，因此在执行这些模型推理时，建议按照requirements_all.txt里的提示单独安装对应版本的transformers

## 运行示例

### Stage 0: 数据集准备 (可选)
框架支持从huggingface或本地读取parquet，以及读取本地jsonl文件两种方法。但由于网速的影响，以及部分数据集较大，强烈建议先从huggingface或modelscope下载parquet数据集，方便反复调用。

在 ```parquet2jsonl.py``` 工具中我们提供了多种组合方式，可自动执行数据集的下载以及处理，将数据集转为jsonl + wav格式方便调用
```bash
# set $save_root_dir and choose a usage mode, then running:
python tools/parquet2jsonl.py
```

如需使用自有数据集，可参考[自定义dataset](assets/custom.md#自定义dataset)中的方式添加自定义数据集进行测试。

### Stage 1: 模型推理 (可选)
下载需要推理的模型，并配置```registry/model/offline.yaml```中相应模型的路径。

任务运行依赖于 ```registry/infer_task``` 中的设置，如果相应```*.yaml```配置文件已修改完成，快速运行可执行例如
```bash
export PYTHONPATH=$PWD:$PYTHONPATH
python main.py --mode "infer" --task "aqa-llamaqa-zh"
```

（**强烈建议**）也可以使用```run.sh```脚本，执行多任务、多模型自动推理。修改```run.sh```中的参数并执行
```bash
bash run.sh  # stage=1
```

### Stage 2: 打分
已有推理结果，可以使用```run.sh```脚本获得在当前eval_task上的得分。

* 框架也支持自有结果的评测（不执行Stage 1），需确保已有的模型推理结果保存在 ```${save_dir}/prediction/${model}/${infer_task}.jsonl``` ，jsonl文件每一行的json需要至少有```key, pred, ref```字段（也可自行指定修改），之后同样执行推理脚本即可。

### 保存目录结构
模型推理、测评结果自动保存如下
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

## 支持的任务
当前支持34个主数据集（98个子数据集），支持的数据集、任务详见[assets/task.md](assets/task.md)

## 数据集信息
数据集信息与对应的测评能力见 [assets/dataset.md](assets/dataset.md#Dataset_Information)

## 具体结果
主要的结果如下表所示
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

* 其中Basic Knowledge、Dialect Comprehension、Dialect-Aware Response为多数据集的加权平均值，Acoustic Robustness为每种声学设置中最差情况的平均值。由于测试的开源模型基本不具备 "无指令条件下回应方言音频"的能力，因此不在此表中展示。
* 不同维度的结果见 [assets/result.md](assets/result.md#results)，更多实验结果及分析见 <a href="https://arxiv.org/abs/2507.18061" target="_blank">Technical Report</a>


## 自定义数据集与模型
本框架提供了完整的模型推理、结果评价的流程，支持灵活的任务、数据集、模型定义，只需要修改```registry```下对应配置文件；如需新增模型，则要继承 **```Model```** 类，并实现 **```generate_once```** 与 **```generate_multiturn```** 方法。详见[assets/custom.md](assets/custom.md)


## 致谢与声明
* 本框架中的部分代码引用、修改自 [UltraEval-Audio](https://github.com/OpenBMB/UltraEval-Audio) 和 [OpenCompass](https://github.com/open-compass/opencompass)
* 数据集中```llamaqa-en, triviaqa-en, webq-en```的音频来自[https://huggingface.co/TwinkStart](https://huggingface.co/TwinkStart)，我们对这些数据集进行了人工筛选，去除不适合作为问答测试的数据，并对答案进行了订正，因此总条数会少于源数据集的条数。
* 各SLM的推理实现基于相应开源项目的演示脚本，我们对其进行了结构上的修改，以便无缝集成到TELEVAL框架中。然而，为了确保所有模型都能执行 *greedy_search* 推理，我们调整了一些模型的代码，例如 ```src_freezeomni/audioLLM.py```

## 引用
如果TELEVAL对您的研究有帮助，期待您能给一个⭐和引用
```bibtex
@article{li2025televal,
  title={TELEVAL: A Dynamic Benchmark Designed for Spoken Language Models in Chinese Interactive Scenarios},
  author={Zehan Li and Hongjie Chen and Qing Wang and Yuxin Zhang and Jing Zhou and Hang Lv and Mengjie Du and Yaodong Song and Jie Lian and Jian Kang and Jie Li and Yongxiang Li and Xuelong Li},
  journal={arXiv preprint arXiv:2507.18061},
  year={2025}
}
```