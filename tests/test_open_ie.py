# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     test_open_ie
   Description :
   Author :       chenhao
   date：          2019-11-25
-------------------------------------------------
   Change Activity:
                   2019-11-25:
-------------------------------------------------
"""
import unittest

from eigen_nltk.open_ie import *
from eigen_nltk.utils import jload


class TestOpenIE(unittest.TestCase):
    @unittest.skip()
    def test_predict(self):
        data = jload("../data/sample.json")
        data = data[1:2]
        print(data)

        ner_extractor = NerExtractor.load_estimator("../model/ner-head-tail-model-v1")
        nre_extractor = NreExtractor.load_estimator("../model/nre-model-v1")
        predicate_ner_extractor = NerExtractor.load_estimator("../model/ner-predicate-given-model-v1")

        open_ie = SchemaFreeSPOExtractor("test_open_ie", ner_extractor, nre_extractor, predicate_ner_extractor)
        pred = open_ie.predict_batch(data)
        print(jdumps(pred))
