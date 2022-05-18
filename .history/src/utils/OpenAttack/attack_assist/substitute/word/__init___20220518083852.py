from .base import WordSubstitute
from .chinese_cilin import ChineseCiLinSubstitute
from .chinese_hownet import ChineseHowNetSubstitute
from .chinese_wordnet import ChineseWordNetSubstitute
from .chinese_word2vec import ChineseWord2VecSubstitute

from .embed_based import EmbedBasedSubstitute

from .english_hownet import HowNetSubstitute
from .english_wordnet import WordNetSubstitute
from .english_counterfit import CounterFittedSubstitute
from .english_word2vec import Word2VecSubstitute
from .english_glove import GloveSubstitute

from .protein_blosum62 import ProteinBlosum62Substitute


def get_default_substitute(lang):
    from ....tags import TAG_Protein
    if lang == TAG_Protein:
        return ProteinBlosum62Substitute()
    return WordNetSubstitute()