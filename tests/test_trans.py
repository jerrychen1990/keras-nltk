# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     test_trans
   Description :
   Author :       chenhao
   date：          2019-09-29
-------------------------------------------------
   Change Activity:
                   2019-09-29:
-------------------------------------------------
"""
import unittest
from eigen_nltk.trans import DataParser, get_token2char_mapping
from eigen_nltk.utils import jload
from eigen_nltk.core import Context

text_item = dict(
    content="They are all in on it and the same people who caused the destruction of the economy are still running "
            "the show so they can keep us down and buy up the country cheap.")


@unittest.skip("past")
class TestUtils(unittest.TestCase):
    pass
    # context = Context("../../pretrained_model/multi_cased_L-12_H-768_A-12/vocab.json")
    # data_parser = DataParser(context)
    #
    # def test_get_bert_input(self):
    #     rs = TestUtils.data_parser.get_bert_input(text_item['content'])
    #     print(rs)

    # def test_get_token_input(self):
    #     text_list = ["2284 . The Adams - Onís Treaty upset many American expansionists , who criticized Adams for not " \
    #                  "laying claim to all of Texas , which they believed had been included in the Louisiana Purchase . "]
    #
    #     for text in text_list:
    #         tmp = TestUtils.data_parser.get_token_input(text)
    #         token2char = tmp['token2char_mapping'][1:]
    #         print([text[token2char[i]:token2char[i + 1]] for i in range(len(token2char) - 1)])
    #
    # def test_get_token2char_mapping(self):
    #     text = text_item['content']
    #     token_list = self.data_parser.get_bert_input(text)['token']
    #
    #     rs = get_token2char_mapping(token_list, text)
    #     print(rs)
