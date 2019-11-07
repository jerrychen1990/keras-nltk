# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     test_utils
   Description :
   Author :       chenhao
   date：          2019-09-20
-------------------------------------------------
   Change Activity:
                   2019-09-20:
-------------------------------------------------
"""

import unittest
from eigen_nltk.utils import *


class TestUtils(unittest.TestCase):
    test_split_text = "They are, al.l in o;n it a?nd the same peo。ple! who caused the de；struction of the economy are？ still " \
                      "running the， show so ！they can keep us down and buy up the country cheap. "
    test_split_token_list = ["[CLS]", "they", "are", ",", "al", ".", "l", "in", "o", ";", "n", "it", "a", "?", "n", "##d",
                        "the", "same", "p", "##eo", "。", "p", "##le", "!", "who", "caused", "the", "de", "；", "s",
                        "##truction", "of", "the", "economy", "are", "？", "still", "running", "the", "，", "show", "so",
                        "！", "they", "can", "keep", "us", "down", "and", "buy", "up", "the", "country", "cheap", ".",
                        "[SEP]"]

    def test_find_all_char(self):
        content = "ABCb caABababab"
        text = "abab"
        self.assertEqual([(7, 11), (9, 13), (11, 15)], find_all_char(content, text, overlap=True, ignore_case=True))
        self.assertEqual([(7, 11), (11, 15)], find_all_char(content, text, overlap=False, ignore_case=True))
        self.assertEqual([(9, 13), (11, 15)], find_all_char(content, text, overlap=True, ignore_case=False))
        self.assertEqual([(9, 13)], find_all_char(content, text, overlap=False, ignore_case=False))

    def test_sample_data(self):
        data = [dict(label=1)] * 20 + [dict(label=2)] * 10 + [dict(label=3)] * 2
        sample_rs = sample_data(data, 0.1)
        print(sample_rs)
        sample_rs = sample_data(data, 0.1, label_key="label")
        print(sample_rs)
        print(data)

    def test_split_text_by_commas(self):
        rs = split_text_by_commas(self.test_split_text)
        print(rs)
        self.assertEqual(12, len(rs))

    def test_split_token_by_commas(self):
        rs = split_token_by_commas(self.test_split_token_list)
        print(rs)
        # self.assertEqual(12, len(rs))

    def test_split_token(self):
        token_list = ["[CLS]", "ant", "ant", ".", "but", "but", "but", "?", "cat", "cat", "cat", "cat", "cat", "cat",
                      "。", "dog", "dog", "dog", "dog", "dog", "dog", "dog", "##dog", "dog", "dog", "dog", "dog", "dog",
                      "dog", "dog", "!", "egg", "egg", "[SEP]"]
        short_token_list = split_token(token_list, 7)
        print(short_token_list)

