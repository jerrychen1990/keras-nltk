# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     test_service
   Description :
   Author :       chenhao
   date：          2019-12-03
-------------------------------------------------
   Change Activity:
                   2019-12-03:
-------------------------------------------------
"""
import unittest
from eigen_nltk.service.open_ie_service import OpenIEService
from eigen_nltk.utils import jdumps


class TestService(unittest.TestCase):
    @unittest.skip("no model file")
    def test_spo_extract(self):
        predicate_extractor_path = "../model/dev-ner-cls-p-model-v1"
        subject_extractor_path = "../model/dev-ner-s-with-p-model-v1"
        object_extractor_path = "../model/dev-ner-o-with-sp-model-v1"

        service = OpenIEService(model_name="dev_test_service", predicate_extractor_path=predicate_extractor_path,
                                subject_extractor_path=subject_extractor_path,
                                object_extractor_path=object_extractor_path)
        data = ["艾耕科技是杭州的一家人工智能创业公司"]

        rs = service.predict(data)
        print(jdumps(rs))
