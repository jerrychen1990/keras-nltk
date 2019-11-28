# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     test_pretrain
   Description :
   Author :       chenhao
   date：          2019-10-12
-------------------------------------------------
   Change Activity:
                   2019-10-12:
-------------------------------------------------
"""
import unittest
from eigen_nltk.trans import DataParser, get_token2char_mapping
from eigen_nltk.pretrain import BertPreTrainer
from eigen_nltk.core import Context

test_data = [
    'What is biology? In simple terms, biology is the study of living organisms and their interactions with '
    'one another and their environments. This is a very broad definition because the scope of biology is '
    'vast.',
    'What does the study of biology share with other scientific disciplines? Science (from the Latin '
    'scientia, meaning “knowledge”) can be defined as knowledge that covers general truths or the operation '
    'of general laws, especially when acquired and tested by the scientific method. It becomes clear from '
    'this definition that the application of the scientific method plays a major role in science.',
    'A valid hypothesis must be testable. It should also be falsifiable, meaning that it can be disproven by '
    'experimental results. Importantly, science does not claim to “prove” anything because scientific '
    'understandings are always subject to modification with further information.']


@unittest.skip()
class TestUtils(unittest.TestCase):

    @unittest.skip("reason for skipping")
    def test_enhance_data(self):
        context = Context("../../pretrained_model/multi_cased_L-12_H-768_A-12/vocab.json")
        bert_pre_trainer = BertPreTrainer("test_bert_pre_trainer", context, 50)
        data = [
            'What is biology? In simple terms, biology is the study of living organisms and their interactions with '
            'one another and their environments. This is a very broad definition because the scope of biology is '
            'vast.',
            'What does the study of biology share with other scientific disciplines? Science (from the Latin '
            'scientia, meaning “knowledge”) can be defined as knowledge that covers general truths or the operation '
            'of general laws, especially when acquired and tested by the scientific method. It becomes clear from '
            'this definition that the application of the scientific method plays a major role in science.',
            'A valid hypothesis must be testable. It should also be falsifiable, meaning that it can be disproven by '
            'experimental results. Importantly, science does not claim to “prove” anything because scientific '
            'understandings are always subject to modification with further information.']

        rs = bert_pre_trainer._get_enhanced_data(data)
        print(rs)
