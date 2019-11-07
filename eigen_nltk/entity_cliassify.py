# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     nre
   Description :
   Author :       chenhao
   date：          2019-10-08
-------------------------------------------------
   Change Activity:
                   2019-10-08:
-------------------------------------------------
"""

from collections import defaultdict

from keras import Model
from keras.layers import *
from keras.losses import sparse_categorical_crossentropy
from keras.regularizers import l1_l2

from eigen_nltk.classify import ClassifyContext
from eigen_nltk.core import ModelEstimator
from eigen_nltk.model_utils import pick_slice, get_seq_embedding_model, get_base_customer_objects
from eigen_nltk.optimizer import get_optimizer_cls, AccumOptimizer
from eigen_nltk.trans import DataParser
from eigen_nltk.utils import padding_seq, add_offset, get_major_element, get_distribution

EntityClsContext = ClassifyContext


class EntityClsDataParser(DataParser):
    def __init__(self, context):
        assert isinstance(context, EntityClsContext)
        self.context = context
        self.tokenizer = context.tokenizer


class EntityClsEstimator(ModelEstimator):
    customer_objects = get_base_customer_objects()

    def __init__(self, name, context, max_len, logger_level="INFO"):
        self.context = context
        self.data_parser = EntityClsDataParser(context)
        self.max_len = max_len
        self.vocab_size = self.context.vocab_size
        self.label_size = self.context.label_size
        super().__init__(name, self.data_parser, logger_level)

    def _build_model(self, use_bert=True, fine_tune_bert=False, use_lstm=False,
                     word_embedding_dim=16, lstm_dim=16, freeze_layer_num=0, drop_rate=0.2, l1=0, l2=0.01,
                     bert_ckpt_path=None, bert_keras_path=None, **kwargs):
        entity_input = Input(shape=(self.max_len, 1), dtype='float32', name='e1')

        seq_embedding_model = get_seq_embedding_model(self.max_len, self.vocab_size,
                                                      freeze_layer_num, word_embedding_dim, lstm_dim,
                                                      use_bert, fine_tune_bert, use_lstm,
                                                      bert_ckpt_path, bert_keras_path)

        words_input, seg_input = seq_embedding_model.inputs
        feature = seq_embedding_model.output

        entity_feature = Lambda(pick_slice)([feature, entity_input])
        entity_feature = Concatenate()([entity_feature, Lambda(lambda x: x[:, 0, :])(feature)])
        final_feature = Dropout(drop_rate)(entity_feature)
        out = Dense(self.label_size, activation="sigmoid", kernel_regularizer=l1_l2(l1, l2))(final_feature)
        model = Model([words_input, seg_input, entity_input], out)
        return model

    def _compile_model(self, optimizer_name, optimizer_args, acc_num=1, **kwargs):
        opt_cls = get_optimizer_cls(optimizer_name)
        optimizer = opt_cls(**optimizer_args)
        if acc_num > 1:
            self.logger.info("get soft batch with acc_num = {}".format(acc_num))
            optimizer = AccumOptimizer(optimizer, acc_num)
        self.training_model.compile(optimizer, loss=sparse_categorical_crossentropy, metrics=["accuracy"])
        return self.training_model

    def _get_model_train_input(self, train_data):
        x = []
        seg = []
        entity_pos = []
        y = []
        for item in train_data:
            x.append(padding_seq(item['x'], self.max_len))
            seg.append(padding_seq(item['seg'], self.max_len))
            entity, start, end = item['entity_info']
            entity_tag = [0] * self.max_len
            entity_tag[start] = 1
            entity_pos.append(entity_tag)
            if 'label' in item.keys():
                label = item['label']
                y.append(self.context.label2id[label])

        x = np.array(x)
        seg = np.array(seg)
        entity_pos = np.array(entity_pos)[:, :, np.newaxis]
        if y:
            y = np.array(y)
        return [x, seg, entity_pos], y

    def create_model(self, model_args):
        model_args["max_len"] = self.max_len
        super().create_model(model_args)

    # add more information to the origin data
    def _get_enhanced_data(self, data):
        short_data = self._get_short_data(data)
        rs_data = []
        for idx, item in enumerate(short_data):
            text = item['content']
            token_input = self.data_parser.get_token_input(text)
            char2token_mapping = token_input['char2token_mapping']
            item.update(**token_input)
            entity_list = item['entity_list']
            offset = item['offset']
            for entity, entity_type, span_list in entity_list:
                span_list = [add_offset(span, -offset) for span in span_list]
                for s, e in span_list:
                    if s < 0 or e >= len(char2token_mapping):
                        continue
                    start, end = char2token_mapping[s], char2token_mapping[e]
                    tmp_item = copy.deepcopy(item)
                    tmp_item, start, end = add_entity_tag(tmp_item, start, end)
                    tmp_item["entity_info"] = (entity, start, end)
                    if entity_type:
                        tmp_item["label"] = entity_type
                    rs_data.append(tmp_item)
        self.logger.info("get {0} enhanced data from {1} origin data".format(len(rs_data), len(data)))

        return rs_data

    def _get_short_data(self, data):
        return self.data_parser.get_short_data(data, self.max_len - 2)

    def train_model(self, train_data, dev_data, train_args, compile_args):
        """

        :param dev_data:
        :param train_data: [{"title":"test", "content":"The band performs with a high level of musicality , energy and spirit while combining sensitive group interplay with dynamic solo improvisations.",
                            "entity_list":[["余盆网","1",[[1,4],[101,104]]],["懒财网","1",[[32,35],[82,85],[87,90]]]]]
        :param train_args:
        :return: model
        """
        return super().train_model(train_data, dev_data, train_args, compile_args)

    def _get_predict_data_from_model_output(self, origin_data, enhanced_data, pred_data, show_detail=False,
                                            return_distribute=False):
        rs_dict = defaultdict(dict)
        pred_hard = np.argmax(pred_data, axis=-1)
        if show_detail:
            print("raw ner output:\n{}".format(pred_hard))
        for rel, item in zip(pred_hard, enhanced_data):
            idx = item['idx']
            rel_name = self.context.id2label[rel]
            entity_name = item['entity_info'][0]
            tmp_dict = rs_dict[idx]
            if entity_name in tmp_dict.keys():
                tmp_dict[entity_name].append(rel_name)
            else:
                tmp_dict[entity_name] = [rel_name]
        rs_list = []
        for idx in range(len(origin_data)):
            tmp_dict = rs_dict[idx]
            if return_distribute:
                tmp_list = [(k, get_distribution(v)) for k, v in tmp_dict.items()]
            else:
                tmp_list = [(k, get_major_element(v)) for k, v in tmp_dict.items()]
            rs_list.append(tmp_list)
        return rs_list


ENTITY_START = ['[s]', 10]
ENTITY_END = ['[e]', 11]


def add_entity_tag(item, start, end):
    token = item['token']
    x = item['x']
    token.insert(start, ENTITY_START[0])
    x.insert(start, ENTITY_START[1])
    end += 1
    token.insert(end, ENTITY_END[0])
    x.insert(end, ENTITY_END[1])
    end += 1
    item['seg'] = item['seg'] + [0] * 2
    return item, start, end
