import re


def replace_urls(text, replacement=' url '):
    return re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+ ', replacement, text)


def add_space_around(text, elements=r'([+\-\{\}\[\]\(\)=–])'):
    return re.sub(elements, r' \1 ', text)
    # return re.sub(r'([-\)])', r' \1 ', text)
