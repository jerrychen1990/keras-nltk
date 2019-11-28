# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     mytokenize
   Description :
   Author :       chenhao
   date：          2019-11-21
-------------------------------------------------
   Change Activity:
                   2019-11-21:
-------------------------------------------------
"""
from keras_bert import Tokenizer
from eigen_nltk.utils import split_text_by_sep


class MyTokenizer(Tokenizer):
    def __init__(self, token_dict, special_token_list):
        super().__init__(token_dict)
        self.special_token_list = special_token_list

    def _tokenize(self, text):
        seg_list = split_text_by_sep(text, self.special_token_list)
        rs_list = []
        for idx, ele in enumerate(seg_list):
            if idx % 2 == 0:
                rs_list.extend(super()._tokenize(ele))
            else:
                rs_list.append(ele)
        return rs_list
