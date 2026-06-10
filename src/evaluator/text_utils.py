import re
import string
import zhconv
from zhon.hanzi import punctuation as zh_punct

from tn.chinese.normalizer import Normalizer as ZhNormalizer
from tn.english.normalizer import Normalizer as EnNormalizer


class TextProcessor:
    RE_PUNCTUATION = re.compile(rf'[{re.escape(zh_punct + string.punctuation)}]')
    RE_SPACES = re.compile(r'[\s\u3000]+')

    def __init__(self, language: str = "zh"):
        self.language = language
        if self.language == "zh":
            self.normalizer = ZhNormalizer()
        elif self.language == "en":
            self.normalizer = EnNormalizer()
        else:
            raise ValueError(f"Unsupported language: {self.language}")

    @staticmethod
    def clean_text(text: str, remove_punct: bool = True, remove_space: bool = True) -> str:
        if remove_punct:
            text = TextProcessor.RE_PUNCTUATION.sub('', text)
        if remove_space:
            text = TextProcessor.RE_SPACES.sub('', text)
        return text
    
    @staticmethod
    def convert_cn(text: str) -> str:
        return zhconv.convert(text, 'zh-cn')
    
    def normalize_text(self, text: str) -> str:
        return self.normalizer.normalize(text)

    def normalize_and_clean(self, text: str,
                            do_normalize: bool = True,
                            simplified_zh: bool = True,
                            remove_punct: bool = True,
                            remove_space: bool = False) -> str:
        if simplified_zh:
            text = self.convert_cn(text)
        if do_normalize:
            text = self.normalize_text(text)
        return self.clean_text(text, remove_punct=remove_punct, remove_space=remove_space)