from .base import Tokenizer
from .jieba_tokenizer import JiebaTokenizer
from .punct_tokenizer import PunctTokenizer
from .transformers_tokenizer import TransformersTokenizer
from .protein_tokenizer import ProteinTokenizer

def get_default_tokenizer(lang):
    from ...tags import TAG_English, TAG_Chinese, TAG_Protein
    if lang == TAG_English:
        return PunctTokenizer()
    if lang == TAG_Chinese:
        return JiebaTokenizer()
    if lang == TAG_Protein:
        return ProteinTokenizer()
    return PunctTokenizer()