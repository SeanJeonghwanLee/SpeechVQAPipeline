import re
from .korean import latin_to_hangul, number_to_hangul, divide_hangul, korean_to_lazy_ipa, korean_to_ipa

def korean_cleaners(text):
    '''Pipeline for Korean text'''
    text = latin_to_hangul(text)
    text = number_to_hangul(text)
    text = divide_hangul(text)
    #text = korean_to_ipa(text)
    text = re.sub(r'([\u3131-\u3163])$', r'\1.', text)
    return text