from .base import Tokenizer

from .protein_tokenizer import ProteinTokenizer

def get_default_tokenizer(lang):
    from ...tags import TAG_Protein
    if lang == TAG_English:
        return PunctTokenizer()
    if lang == TAG_Chinese:
        return JiebaTokenizer()
    if lang == TAG_Protein:
        return ProteinTokenizer()
    return PunctTokenizer()