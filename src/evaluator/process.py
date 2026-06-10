import unicodedata
import regex
import re
import json
from collections import Counter

class OptionExtractor:
    RAW_OPTION = r'(?P<option>[A-Da-d])(?![A-Za-z0-9])'
    OPTION_PATTERN = (
        rf"(?:[\(\[【“‘']?({RAW_OPTION})[\)\]】”’']?)"
    )

    # universal delimiter that allows crossing punctuation and spaces
    OPTIONAL_MIDDLE = r'[\s，。、：,.:]*'

    # —— keyword list —— 
    POSITIVE_HINTS     = ["综上(?:所述)?", "总结", "综合(?:分析|来看)?", "最终", "因此", "所以",
                          "由此可见", "可见", "总的来说", "结论(?:是)?", "可以看出"]
    POSITIVE_ACTIONS = [
        "答案", "选项", "选择", "结论", "应选", "倾向(?:选择)?",
        "应当(?:选择)?", "推荐(?:选择)?", "应该",
        "正确(?:的)?(?:叙述|选项|描述|陈述)", "最(?:有)?可能"
    ]
    ANSWER_VERBS = [
        "为", "是", "选", "应该", "应是", "应为", "可能是", "应该是", "应该选", "会是", "选择", "就是", "是(?:最|更)?(?:合适|合理|正确)(?:的)?"
    ]

    CORRECT_INDICATORS = ["正确", "最佳", "合适", "唯一", "准确", "合理", "首选", "正确答案"]

    NEGATIVE_HINTS     = ["错误(?:的)?", "错误(?:说法)?", "错误(?:选项)?", "错误(?:叙述|选项|描述|陈述)?",
                          "不正确(?:的)?", "不对", "无效", "不合理", "不符合", "有问题", "不当"]
    # —— summary (the first) ——  
    NEG_INDICATOR = rf"(?:{'|'.join(NEGATIVE_HINTS)})"
    ANSWER_VERBS_RE = rf"(?:{'|'.join(ANSWER_VERBS)})"
    SUMMARY_HINT_RE = rf"(?:{'|'.join(POSITIVE_HINTS)})"

    STRONG_ANSWER_RE = re.compile(
        rf"{SUMMARY_HINT_RE}{OPTIONAL_MIDDLE}"
        rf"(?:{'|'.join(POSITIVE_ACTIONS)})?{OPTIONAL_MIDDLE}"
        rf"{ANSWER_VERBS_RE}{OPTIONAL_MIDDLE}"
        rf"({OPTION_PATTERN})"
    )
    STRONG_WRONG_RE = re.compile(
        rf"{SUMMARY_HINT_RE}{OPTIONAL_MIDDLE}"
        rf"{NEG_INDICATOR}{OPTIONAL_MIDDLE}"
        rf"{ANSWER_VERBS_RE}{OPTIONAL_MIDDLE}"
        rf"({OPTION_PATTERN})"
    )
    # —— negative ——  
    NEG_INDICATOR = rf"(?:{'|'.join(NEGATIVE_HINTS)})"
    ERROR_QUESTION_PATTERN = re.compile(
        rf"(?:"
            # basic: e.g.哪项是错误的说法？
            rf"(?:以下|下列|下面)?{OPTIONAL_MIDDLE}"
            rf"(?:哪(?:个|项|一项)?|哪些|什么)?{OPTIONAL_MIDDLE}"
            rf"(?:说法|选项|描述|陈述|观点|行为|做法)?{OPTIONAL_MIDDLE}"
            rf"(?:中)?{OPTIONAL_MIDDLE}(?:是|为)?{OPTIONAL_MIDDLE}"
            rf"{NEG_INDICATOR}{OPTIONAL_MIDDLE}(?:选项|说法)?"
        rf"|"
            # predicate first: e.g. 错误的是哪项？
            rf"{NEG_INDICATOR}(?:的)?{OPTIONAL_MIDDLE}"
            rf"(?:选项|说法)?{OPTIONAL_MIDDLE}(?:是|为)?{OPTIONAL_MIDDLE}"
            rf"(?:哪(?:个|项|一项)?|哪些|什么)?"
        rf"|"
            # negative struct: e.g. 不是正确的是哪项？
            rf"(?:以下|下列|下面)?{OPTIONAL_MIDDLE}"
            rf"(?:哪(?:个|项|一项)?|哪些|什么)?{OPTIONAL_MIDDLE}"
            rf"(?:选项|说法|描述|陈述)?(?:中)?{OPTIONAL_MIDDLE}"
            rf"(?:是|为)?{OPTIONAL_MIDDLE}"
            rf"(?:不是|不属于|不应被选为|不算是|不应该是){OPTIONAL_MIDDLE}"
            rf"(?:正确|合理|符合|应选)(?:的)?"
        rf")"
    )

    # —— regular positive/negative ——  
    # positive
    KEYWORD_RE = re.compile(
        rf"(?:{'|'.join(POSITIVE_ACTIONS)}){OPTIONAL_MIDDLE}"
        rf"{ANSWER_VERBS_RE}{OPTIONAL_MIDDLE}"
        rf"({OPTION_PATTERN})"
    )
    # positive suffix
    POST_KEYWORD_CORRECT = re.compile(
        rf"(?:{'|'.join(POSITIVE_ACTIONS)}){OPTIONAL_MIDDLE}"
        rf"({ANSWER_VERBS_RE})?{OPTIONAL_MIDDLE}"
        rf"({OPTION_PATTERN})"
        rf"({ANSWER_VERBS_RE})?"
        rf"(?=[\s\S]*?(?:{'|'.join(CORRECT_INDICATORS)}))"
    )
    # negative
    KEYWORD_NEG_RE = re.compile(
        rf"(?:{'|'.join(POSITIVE_ACTIONS)}){OPTIONAL_MIDDLE}"
        rf"{ANSWER_VERBS_RE}{OPTIONAL_MIDDLE}"
        rf"{NEG_INDICATOR}?{OPTIONAL_MIDDLE}"
        rf"({OPTION_PATTERN})"
    )
    POST_KEYWORD_WRONG = re.compile(
        rf"(?:{'|'.join(POSITIVE_ACTIONS)}){OPTIONAL_MIDDLE}"
        rf"{ANSWER_VERBS_RE}?{OPTIONAL_MIDDLE}"
        rf"({OPTION_PATTERN})(?=[\s\S]*?{NEG_INDICATOR})"
    )

    @classmethod
    def _cushion_match(cls, text: str):
        # only if cusion=True
        matches = re.findall(cls.RAW_OPTION, text)
        counter = Counter(c.upper() for c in matches)
        filtered = {k: v for k, v in counter.items() if k in 'ABCD'}

        if len(filtered) == 1:
            return next(iter(filtered.keys()))
        return None

    @classmethod
    def _extract_positive(cls, text: str, cushion=False):
        for pat in (cls.STRONG_ANSWER_RE, cls.KEYWORD_RE, cls.POST_KEYWORD_CORRECT):
            if m := pat.search(text):
                if option := m.groupdict().get("option"):
                    return option.upper()
        # fallback cushion
        if cushion:
            return cls._cushion_match(text)
        return None

    @classmethod
    def _extract_negative(cls, text: str, cushion=False):
        for pat in (cls.STRONG_WRONG_RE, cls.KEYWORD_NEG_RE, cls.POST_KEYWORD_WRONG):
            if m := pat.search(text):
                if option := m.groupdict().get("option"):
                    return option.upper()
        if cushion:
            return cls._cushion_match(text)
        return None

    @classmethod
    def extract(cls, pred: str, question: str, cushion=False):
        if cls.ERROR_QUESTION_PATTERN.search(question):
            return cls._extract_negative(pred, cushion=cushion)
        return cls._extract_positive(pred, cushion=cushion)

    @classmethod
    def has_answer(cls, ref: str, pred: str, question: str, cushion=False):
        final_answer = cls.extract(pred, question, cushion=cushion)
        if final_answer is None:
            return {"cover": 0, "correct": 0}
        return {
            "cover": 1,
            "correct": int(final_answer.upper() == ref.upper())
        }

class SimpleTokenizer(object):
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'
    PUNCTUATION = r'[\p{P}]'
    PUNC_REGEX = regex.compile(PUNCTUATION, flags=regex.UNICODE)
    _regexp = regex.compile(
            '(%s)|(%s)' % (ALPHA_NUM, NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )
    
    @staticmethod
    def is_punctuation(char):
        return bool(SimpleTokenizer.PUNC_REGEX.fullmatch(char))
        
    @staticmethod
    def is_chinese_char(char):
        """Determine the Chinese char (including extended characters)"""
        return (
            ("\u4e00" <= char <= "\u9fff") or
            ("\u3400" <= char <= "\u4DBF") or
            ("\u20000" <= char <= "\u2A6DF") or
            ("\u2A700" <= char <= "\u2B73F") or
            ("\u2B740" <= char <= "\u2B81F") or
            ("\u2B820" <= char <= "\u2CEAF") or
            ("\u2CEB0" <= char <= "\u2EBEF") or
            ("\uF900" <= char <= "\uFAFF")
        )
    
    @staticmethod
    def unicode_normalize(text):
        return unicodedata.normalize('NFD', text)

    @classmethod
    def tokenize(cls, text, uncased=False, keep_punc=False):
        text = cls.unicode_normalize(text)
        tokens = [m.group().lower() if uncased else m.group() for m in cls._regexp.finditer(text)]
        new_tokens, temp = [], ""

        def flush_temp(tmp):
            if tmp:
                new_tokens.append(tmp)
            return ""

        for t in tokens:
            for char in t:
                if cls.is_chinese_char(char):
                    temp = flush_temp(temp)
                    new_tokens.append(char)
                elif cls.is_punctuation(char):
                    if keep_punc:
                        temp = flush_temp(temp)
                        new_tokens.append(char)
                else:
                    temp += char
            temp = flush_temp(temp)

        return new_tokens
    
    @classmethod
    def has_answer(cls, refs, pred: str, uncased=True, keep_punc=False) -> bool:
        pred_tokens = cls.tokenize(pred, uncased, keep_punc)
        def match(candidate):
            if isinstance(candidate, str):
                return cls._match_tokens(pred_tokens, candidate, uncased, keep_punc)
            elif isinstance(candidate, list):
                # one in list
                return any(match(c) for c in candidate)
            elif isinstance(candidate, tuple):
                # all in tuple
                return all(match(c) for c in candidate)
            else:
                raise ValueError(f"Invalid ref type: {type(candidate)}")

        # The outer List represents an "or" relationship. Any one of the matches being successful is sufficient.
        return any(match(ref) for ref in refs)

    @classmethod
    def _match_tokens(cls, tokens, candidate, uncased=True, keep_punc=False):
        candidate_tokens = cls.tokenize(candidate, uncased, keep_punc)
        for i in range(len(tokens) - len(candidate_tokens) + 1):
            if candidate_tokens == tokens[i: i + len(candidate_tokens)]:
                return True
        return False
