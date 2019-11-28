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

from eigen_nltk.entity_cliassify import *
from eigen_nltk.utils import jdumps

test_data = [
    {
        "id": "e574ac4d",
        "content": "#余盆网诈骗犯覃丽昀##igofx张雪娇武加伟王明王海坤郝龙##懒财网诈骗高手陶伟杰邹通祥##懒财陶伟杰李子拓莫晓淅孙菲廖志达##甘肃省人大法制办邹通祥参与诈骗平台懒财网##懒财网陶伟杰诈骗[超话]##余盆网#",
        "entity_list": [
            [
                "余盆网",
                "1",
                [
                    [
                        1,
                        4
                    ],
                    [
                        101,
                        104
                    ]
                ]
            ],
            [
                "懒财网",
                "1",
                [
                    [
                        32,
                        35
                    ],
                    [
                        82,
                        85
                    ],
                    [
                        87,
                        90
                    ]
                ]
            ]
        ]
    },
]

item = test_data[0]


class TestEntityClassify(unittest.TestCase):
    context = EntityClsContext("schema/vocab.txt", "schema/entity_classify_label_dict.json")
    @unittest.skip()
    def test_enhance_data(self):
        nre_extractor = EntityClsEstimator("test_entity_classify_estimator", self.context, 50)
        rs = nre_extractor._get_enhanced_data(test_data)
        print(jdumps(rs))

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
    #
    # def test_eval_nre(self):
    #     nre_pred = [[("Message-Topic", ('people', 'ENTITY', [35, 41]), ('destruction', 'ENTITY', [57, 68]))]]
    #     rs = eval_nre(test_data, nre_pred)
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
