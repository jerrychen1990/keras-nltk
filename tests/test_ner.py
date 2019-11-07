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

from eigen_nltk.constants import *
from eigen_nltk.ner import *
from eigen_nltk.utils import jdumps

test_data = [
    dict(id=1, title="1",
         content="They are all in on it and the same people. who caused the destruction of the economy are still running "
                 "the show so they can keep us down and buy up the country cheap.",
         entity_list=[
             [
                 "people",
                 "ENTITY",
                 [
                     [
                         35,
                         41
                     ]
                 ]
             ],
             [
                 "destruction",
                 "ENTITY",
                 [
                     [
                         58,
                         69
                     ]
                 ]
             ]
         ])
]

item = test_data[0]


class TestNer(unittest.TestCase):
    # context = NerContext("../../pretrained_model/cased_L-12_H-768_A-12/vocab.txt",
    #                      "../schema/ner_dict_simple_bmeso.json",
    #                      BMESO)
    # data_parser = NerDataParser(context)

    # def test_enhance_data(self):
    #     ner_extractor = NerExtractor("test_ner_extractor", self.context, 6)
    #     rs = ner_extractor._get_enhanced_data(test_data)
    #     print(jdumps(rs))
    #
    # def test_get_short_item(self):
    #     ner_extractor = NerExtractor("test_ner_extractor", self.context, 18)
    #     rs = ner_extractor._get_short_data(test_data)
    #     print(rs)
    #
    # def test_ner_output(self):
    #     ner_extractor = NerExtractor("test_ner_extractor", self.context, 5)
    #     enhanced_data = ner_extractor._get_enhanced_data(test_data)
    #     ner_out = [e['ner_output'] for e in enhanced_data]
    #     print(ner_out)
    #     entity_list_pred = [
    #         ner_extractor.data_parser.get_ner_from_output(item['text'], pred, item['token2char_mapping']) for
    #         item, pred in zip(enhanced_data, ner_out)]
    #     merged_entity_list_pred = merge_entity_lists(entity_list_pred, enhanced_data)
    #     print(merged_entity_list_pred)

    def test_get_max_ner_pair(self):
        prob_matrix = [
            [.6, .9, .5, .5],
            [.5, .9, 1., .5],
            [.8, .6, .5, .8],
            [.9, .9, .5, .8],
            [.5, .4, .7, .4],
            [.5, .5, .6, .5],
            [.7, .2, .8, .7]
        ]
        prob_matrix = np.array(prob_matrix)
        loss, path = get_max_ner_pair(prob_matrix[:, :2], 0, 0, 7)
        print(loss, path)
        loss, path = get_max_ner_pair(prob_matrix[:, 2:], 0, 0, 7)
        print(loss, path)
