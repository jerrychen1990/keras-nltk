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
from eigen_nltk.nre import *
from eigen_nltk.constants import *
from eigen_nltk.eval import eval_nre
from eigen_nltk.utils import split_data2seq_list, jdumps, jload

test_data = [
    dict(id=1, title="1",
         content="They are all in on it and the same people who caused the destruction of the economy are still running the show so they can keep us down and buy up the country cheap.",
         rel_list=[
             [

                 "Message-Topic",
                 [
                     "people",
                     "ENTITY",
                     [
                         35,
                         41
                     ]
                 ],
                 [
                     "destruction",
                     "ENTITY",
                     [
                         57,
                         68
                     ]
                 ]
             ]
         ]
         )
]

item = test_data[0]


@unittest.skip("past")
class TestNre(unittest.TestCase):
    # context = NreContext("../../pretrained_model/cased_L-12_H-768_A-12/vocab.json",
    #                      "../schema/rel_dict.json")
    # data_parser = NreDataParser(context)

    # def test_enhance_data(self):
    #     nre_extractor = NreExtractor("test_nre_extractor", self.context, 50)
    #     rs = nre_extractor._get_enhanced_data(test_data)
    #     print(jdumps(rs))
    #
    # def test_model_input(self):
    #     nre_extractor = NreExtractor("test_nre_extractor", self.context, 50)
    #     data = nre_extractor._get_enhanced_data(test_data)
    #     x, y = nre_extractor._get_model_train_input(data)
    #     print(x)
    #     print(y)
    #
    # def test_predict(self):
    #     nre_extractor = NreExtractor("test_nre_extractor", self.context, 100)
    #     nre_extractor.create_model(dict(use_bert=False, lstm_dim=4))
    #     rs = nre_extractor.predict_batch(test_data, batch_size=1)
    #     print(rs)
    @unittest.skip("past")
    def test_add_entity_tag(self):
        rs = add_entity_tag("“1363622”是七星彩第17127期的开奖号码，眼尖的购彩者看出这是十堰地区的手机号。", [10, 13], [1, 8])
        print(rs)

    @unittest.skip("past")
    def test_eval_nre(self):
        nre_pred = [[("Message-Topic", ('people', 'ENTITY', [35, 41]), ('destruction', 'ENTITY', [57, 68]))]]
        rs = eval_nre(test_data, nre_pred)
        print(rs)
    #
    # def test_get_short_data(self):
    #     data = [{'id': '1492',
    #              'title': '1492',
    #              'content': 'The present bright, dry weather, although cold, is just what flock owners require and '
    #                         'judging from the large number of lambs seen skipping about nibbling the tender shoots of '
    #                         'the fresh green herbage, the flocks of Mr John Cracknell, Mr Geo Walker, and Mr Russell '
    #                         'Walker appear to have had a very satisfactory fall of good strong lambs .',
    #              'entity_list': [['lambs', 'ENTITY', [[119, 124], [329, 334]]],
    #                              ['fall', 'ENTITY', [[309, 313]]]],
    #              'rel_list': [['Member-Collection',
    #                            ['lambs', 'ENTITY', [119, 124]],
    #                            ['fall', 'ENTITY', [309, 313]]]]}]
    #     nre_extractor = NreExtractor("test_nre_extractor", self.context, 120)
    #     rs = nre_extractor._get_short_data(data)
    #     print(rs)
    #
