# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     text_generate
   Description :
   Author :       chenhao
   date：          2019-10-23
-------------------------------------------------
   Change Activity:
                   2019-10-23:
-------------------------------------------------
"""

import numpy as np
from eigen_nltk.core import ModelEstimator
from eigen_nltk.trans import DataParser
from eigen_nltk.optimizer import get_optimizer_cls
from eigen_nltk.utils import padding_seq
from eigen_nltk.model_utils import get_seq_embedding_model
from keras.layers import *
from keras.losses import sparse_categorical_crossentropy
from keras.models import Model


class TextGenerator(ModelEstimator):
    def __init__(self, name, context, max_len, logger_level="INFO"):
        data_parser = DataParser(context)
        self.vocab_size = context.vocab_size
        self.max_len = max_len
        super().__init__(name, data_parser, logger_level)

    def _build_model(self, word_embedding_dim, lstm_dim, use_bert=False, fine_tune_bert=True, use_lstm=True,
                     bert_ckpt_path=None, bert_keras_path=None, **kwargs):
        super()._build_model(**kwargs)

        seq_embedding_model = get_seq_embedding_model(self.max_len, self.vocab_size,
                                                      0, word_embedding_dim, lstm_dim,
                                                      use_bert, fine_tune_bert, use_lstm,
                                                      bert_ckpt_path, bert_keras_path)
        word_feature = seq_embedding_model.output

        word_out = Dense(self.vocab_size, activation="softmax")(word_feature)
        model = Model(seq_embedding_model.inputs, word_out)
        return model

    def _compile_model(self, optimizer_name, optimizer_args, **kwargs):
        opt_cls = get_optimizer_cls(optimizer_name)
        optimizer = opt_cls(**optimizer_args)
        self.training_model.compile(optimizer, loss=sparse_categorical_crossentropy)

    def _get_predict_data_from_model_output(self, origin_data, enhanced_data, pred_data, show_detail=False):
        pass

    def _get_short_data(self, data):
        return self.data_parser.get_short_data(data, self.max_len)

    def _get_enhanced_data(self, data):
        rs_list = []
        short_data = self._get_short_data(data)
        for item in short_data:
            text = item['content']
            tmp_item = copy.copy(item)
            if not text:
                continue
            bert_dict = self.data_parser.get_bert_input(text)
            tmp_item.update(**bert_dict)
            tmp_item['out_x'] = bert_dict["x"][1:] + [102]
            tmp_item['out_token'] = bert_dict['token'][1:] + ["[SEP]"]
            rs_list.append(tmp_item)
        return rs_list

    def _get_model_train_input(self, train_data, **kwargs):
        x = []
        seg = []
        y = []
        for item in train_data:
            x.append(padding_seq(item['x'], self.max_len))
            seg.append(padding_seq(item['seg'], self.max_len))
            y.append(padding_seq(item['out_x'], self.max_len))

        x = np.array(x)
        seg = np.array(seg)
        y = np.array(y)[:, :, np.newaxis]

        return [x, seg], y

    def _get_model_test_input(self, test_data, **kwargs):
        pass
