def get_system_prompts(prompt_name):
    PROMPT_MAP = {
        "qwen2_prompt": qwen2_prompt,
        "qwen3_prompt": qwen3_prompt,
    }
    return PROMPT_MAP[prompt_name]

# 任务prompt
qwen2_prompt = """You are a helpful assistant."""
qwen3_prompt = """"""
think_instruct = """
严格遵循用户指令，按用户要求完成任务。
"""