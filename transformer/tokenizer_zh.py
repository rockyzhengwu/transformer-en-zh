# -*-coding:utf-8-*-

import jieba

class Tokenizer(object):
    def __init__(self, user_dict=None):
        if user_dict:
            jieba.load_userdict(user_dict)

    def tokenize(self, text):
        words = list(jieba.cut(text))
        return words

