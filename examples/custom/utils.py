import unicodedata
import re

def count_char_types(text):
    counts = {
        'english_count': [0, ""],
        'arabic_count': [0, ""],
        'other_language_count': [0, ""],
        'digit_count': [0, ""],
        'punctuation_symbol_count': [0, ""],
        'whitespace_count': [0, ""],
    }

    for char in text:
        if re.match(r'[A-Za-z]', char):
            counts['english_count'][0] += 1
            counts['english_count'][1] += char
        elif re.match(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]', char):
            counts['arabic_count'][0] += 1
            counts['arabic_count'][1] += char
        elif char.isdigit():
            counts['digit_count'][0] += 1
            counts['digit_count'][1] += char
        elif char.isspace():
            counts['whitespace_count'][0] += 1
            counts['whitespace_count'][1] += char
        elif unicodedata.category(char).startswith(('P', 'S')):  # Punctuation or Symbol
            counts['punctuation_symbol_count'][0] += 1
            counts['punctuation_symbol_count'][1] += char
        elif char.isalpha():  # Anything else that's a letter from another script

            counts['other_language_count'][0] += 1
            counts['other_language_count'][1] += char


    return counts