
from .protein_blosum62 import ProteinBlosum62Substitute
f

def get_default_substitute(lang):
    from ....tags import TAG_Protein
    if lang == TAG_Protein:
        return ProteinBlosum62Substitute()