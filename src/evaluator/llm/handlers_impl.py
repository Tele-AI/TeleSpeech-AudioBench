import json_repair
from src.evaluator.llm.llm_handlers import LLMHandlerRegistry

@LLMHandlerRegistry.register_builder("emo_response_accept_preprocess")
def build_emo_response_accept_prompt(pred_info):
    query, query_emo_zh = pred_info["query"], pred_info["query_emo_zh"]
    pred = pred_info["pred"]
    parsed_output = json_repair.repair_json(pred, return_objects=True, ensure_ascii=False)
    if not isinstance(parsed_output, dict):
        return None
    pred, pred_emo = parsed_output.get("pred"), parsed_output.get("pred_emo")
    return {"pred": pred, "pred_emo": pred_emo, "query": query, "query_emo_zh": query_emo_zh}

@LLMHandlerRegistry.register_builder("humandial_emotion_preprocess")
def build_humandial_emotion_prompt(pred_info):
    turns = pred_info["history"]
    conversation = [
        f"用户(emotion:{t['user_meta_info']['user_emo']}): {t['user']}\nAI: {t['bot']}"
        for t in turns
    ]
    return {"query": pred_info["query"], "history": "\n".join(conversation)}

@LLMHandlerRegistry.register_builder("humandial_task3_preprocess")
def build_humandial_task3_prompt(pred_info):
    def find_target_turn(turns, current_user_meta=None):
        """
        找到用户input_emotion第二次不为neutral的轮次
        返回 (target_turn, history_turns)，如果没找到返回 (None, None)
        """
        non_neutral_count = 0
        for i, turn in enumerate(turns):
            emotion = turn["user_meta_info"].get("user_emo")
            if emotion and emotion.lower() != "neutral":
                non_neutral_count += 1
                if non_neutral_count == 2:
                    return turn, turns[: i + 1]
        if current_user_meta and current_user_meta["user_emo"] and current_user_meta["user_emo"].lower() != "neutral":  # 最后一轮满足条件的情况
            non_neutral_count += 1
            if non_neutral_count == 2:
                return pred_info, turns

        return None, None
    
    def build_context(history_turns):
        """把历史轮次拼成上下文"""
        ctx_lines = []
        for t in history_turns:
            user_emotion = t['user_meta_info']['user_emo']
            user_text = t['user']
            ai_text = t['bot']
            if "assistant\n" in ai_text:
                ai_text = ai_text.split("assistant\n")[-1]
            ctx_lines.append(f"User({user_emotion}): {user_text}")
            ctx_lines.append(f"AI: {ai_text}")
        return "\n".join(ctx_lines)

    turns, current_user_meta = pred_info["history"], pred_info.get("current_user_meta", None)
    target_turn, history_turns = find_target_turn(turns, current_user_meta)
    if not target_turn:
        raise ValueError(f"Not target_turn: {pred_info}")
    
    conversation = build_context(history_turns)
    pred_audio = target_turn.get("pred_audio")

    return {"pred_audio": pred_audio, "history": conversation}

# --------------------------- parser -----------------------------------------
@LLMHandlerRegistry.register_parser("emo_response_accept")
def emo_response_accept_parser(data: dict):
    scores = data.get("Score", None)
    if not scores:
        # LLM输出格式不对，需要重新打分
        return None
    reasons = data.get("Explanation", {})
    response_emotion_label_score = int(scores["response_emotion_label"])
    response_text_score = int(scores["response_text"])
    label_consistency_score = int(scores["label_consistency"])
    if response_emotion_label_score > 0:
        avg_score = (response_text_score + label_consistency_score) / 2
    else:
        avg_score = 0
    scores = {
        "avg_score": avg_score,
        "response_emotion_label_score": response_emotion_label_score,
        "response_text_score": response_text_score,
        "label_consistency_score": label_consistency_score
    }

    return {"score": scores, "reason": reasons}

@LLMHandlerRegistry.register_parser("humandial_task1_eval")
def humandial_task1_parser(data: dict):
    def map_to_valid_score(score):
        """确保分数为 1, 3, 5"""
        try:
            score = int(score)
        except (ValueError, TypeError):
            # 如果解析失败，默认给最低分
            return 1
            
        if score <= 2:
            return 1
        elif score <= 4:
            return 3
        else:
            return 5
    
    scores = data.get("scores", {})
    scores["Accuracy_Completeness"] = map_to_valid_score(scores.get("Accuracy_Completeness"))
    scores["Depth_Granularity"] = map_to_valid_score(scores.get("Depth_Granularity"))
    scores["Added_Value"] = map_to_valid_score(scores.get("Added_Value"))

    reasons = data.get("justification", {})
    overall = data.get("overall_comment", "")

    avg_score = sum(scores.values()) / len(scores) if scores else None

    return {
        "score": scores,
        "meta": {
            "avg_score": avg_score,
            "reasons": reasons,
            "overall_comment": overall,
        },
    }

@LLMHandlerRegistry.register_parser("humandial_task2_eval")
def humandial_task2_parser(data: dict):
    def map_to_valid_score(score):
        """确保分数为 1, 3, 5"""
        try:
            score = int(score)
        except (ValueError, TypeError):
            # 如果解析失败，默认给最低分
            return 1
            
        if score <= 2:
            return 1
        elif score <= 4:
            return 3
        else:
            return 5
    
    scores = data.get("scores", {})
    scores["Information_Integration"] = map_to_valid_score(scores.get("Information_Integration"))
    scores["Insight_RootCause"] = map_to_valid_score(scores.get("Insight_RootCause"))
    scores["Clarity_Logic"] = map_to_valid_score(scores.get("Clarity_Logic"))

    reasons = data.get("justification", {})
    overall = data.get("overall_comment", "")

    avg_score = sum(scores.values()) / len(scores) if scores else None

    return {
        "score": scores,
        "meta": {
            "avg_score": avg_score,
            "reasons": reasons,
            "overall_comment": overall,
        },
    }

@LLMHandlerRegistry.register_parser("humandial_task3_eval")
def humandial_task3_parser(data: dict):
    def clamp_score(v):
        """确保分数为 1-5 之间的整数"""
        try:
            iv = int(v)
        except:
            print(f"clamp_score: invalid value {v}")
            return 1
        return max(1, min(5, iv))
    
    scores = data.get("scores", {})
    scores["textual_empathy_insight"] = clamp_score(scores.get("textual_empathy_insight"))
    scores["vocal_empathy_congruence"] = clamp_score(scores.get("vocal_empathy_congruence"))
    scores["audio_quality_naturalness"] = clamp_score(scores.get("audio_quality_naturalness"))

    reasons = data.get("justification", {})
    overall = data.get("overall_comment", "")

    avg_score = sum(scores.values()) / len(scores) if scores else None

    return {
        "score": scores,
        "meta": {
            "avg_score": avg_score,
            "reasons": reasons,
            "overall_comment": overall,
        },
    }

@LLMHandlerRegistry.register_builder("simple_builder")
def build_humandial_emotion_prompt(pred_info):
    return pred_info

@LLMHandlerRegistry.register_parser("simple_parser")
def simple_parser(data: dict):
    score = data.get("Score", None)
    return {
        "score": int(score) if score is not None else None,
        "reason": data.get("Explanation", ""),
        "meta": data
    }
