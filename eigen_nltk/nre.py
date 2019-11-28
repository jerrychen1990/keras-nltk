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
from tqdm import tqdm

from keras import Model
from keras.layers import *
from keras.losses import sparse_categorical_crossentropy
from keras.regularizers import l1_l2

from eigen_nltk.core import ModelEstimator, Context
from eigen_nltk.model_utils import pick_slice, get_seq_embedding_model, get_base_customer_objects
from eigen_nltk.optimizer import get_optimizer_cls
from eigen_nltk.trans import DataParser, add_entity_tag
from eigen_nltk.utils import read_id_mapping, padding_seq, add_offset


class NreContext(Context):
    def __init__(self, vocab_path, rel_dict_path):
        super().__init__(vocab_path)
        self.id2rel, self.rel2id = read_id_mapping(rel_dict_path)
        self.rel_size = len(self.id2rel)

    def get_model_args(self):
        model_args = super().get_model_args()
        model_args.update(rel_size=self.rel_size)
        return model_args


def get_nre_customer_objects():
    nre_customer_objects = get_base_customer_objects()
    return nre_customer_objects


class NreDataParser(DataParser):
    def __init__(self, context):
        assert isinstance(context, NreContext)
        self.context = context
        self.tokenizer = context.tokenizer


class NreExtractor(ModelEstimator):
    customer_objects = get_nre_customer_objects()

    def __init__(self, name, context, max_len, logger_level="INFO"):
        self.context = context
        self.data_parser = NreDataParser(context)
        self.max_len = max_len
        self.vocab_size = self.context.vocab_size
        self.rel_size = self.context.rel_size
        super().__init__(name, self.data_parser, logger_level)

    def _build_model(self, use_bert=True, fine_tune_bert=False, use_lstm=False,
                     word_embedding_dim=16, lstm_dim=16, freeze_layer_num=0, drop_rate=0.2, l1=0, l2=0.01,
                     bert_ckpt_path=None, bert_keras_path=None, **kwargs):
        e1_input = Input(shape=(self.max_len, 1), dtype='float32', name='e1')
        e2_input = Input(shape=(self.max_len, 1), dtype='float32', name='e2')

        seq_embedding_model = get_seq_embedding_model(self.max_len, self.vocab_size,
                                                      freeze_layer_num, word_embedding_dim, lstm_dim,
                                                      use_bert, fine_tune_bert, use_lstm,
                                                      bert_ckpt_path, bert_keras_path)

        words_input, seg_input = seq_embedding_model.inputs
        feature = seq_embedding_model.output

        feature_e1 = Lambda(pick_slice)([feature, e1_input])
        feature_e2 = Lambda(pick_slice)([feature, e2_input])
        final_feature = Concatenate()([feature_e1, feature_e2])
        final_feature = Dropout(drop_rate)(final_feature)
        out = Dense(self.rel_size, activation="softmax", kernel_regularizer=l1_l2(l1, l2))(final_feature)
        model = Model([words_input, seg_input, e1_input, e2_input], out)
        return model

    def _compile_model(self, optimizer_name, optimizer_args, **kwargs):
        opt_cls = get_optimizer_cls(optimizer_name)
        optimizer = opt_cls(**optimizer_args)
        self.training_model.compile(optimizer, loss=sparse_categorical_crossentropy, metrics=["accuracy"])
        return self.training_model

    def _get_model_train_input(self, train_data):
        x = []
        seg = []
        e1_pos = []
        e2_pos = []
        y = []
        for item in train_data:
            x.append(padding_seq(item['x'], self.max_len))
            seg.append(padding_seq(item['seg'], self.max_len))
            e1_tag = item['e1'][3][0] - 1
            e2_tag = item['e2'][3][0] - 1
            e1_tag_vec = [0] * self.max_len
            e1_tag_vec[e1_tag] = 1
            e2_tag_vec = [0] * self.max_len
            e2_tag_vec[e2_tag] = 1
            e1_pos.append(e1_tag_vec)
            e2_pos.append(e2_tag_vec)
            if 'rel_output' in item.keys():
                y.append([item['rel_output']])

        x = np.array(x)
        seg = np.array(seg)
        e1_pos = np.array(e1_pos)[:, :, np.newaxis]
        e2_pos = np.array(e2_pos)[:, :, np.newaxis]
        if y:
            y = np.array(y)
        return [x, seg, e1_pos, e2_pos], y

    def _get_model_test_input(self, test_data):
        [x, seg, e1_pos, e2_pos], y = self._get_model_train_input(test_data)
        return [x, seg, e1_pos, e2_pos]

    def create_model(self, model_args):
        model_args["max_len"] = self.max_len
        super().create_model(model_args)

    # add more information to the origin data
    def _get_enhanced_data(self, data):
        short_data = self._get_short_data(data)
        self.logger.info("get {0} short data from {1} origin data".format(len(short_data), len(data)))
        rs_data = []
        for idx, item in tqdm(enumerate(short_data)):
            text = item['content']
            rel_list = item['rel_list']
            offset = item['offset']
            for rel, (e1, et1, es1), (e2, et2, es2) in rel_list:
                start1, end1 = add_offset(es1, -offset)
                start2, end2 = add_offset(es2, -offset)
                if min(start1, start2) < 0 or max(end1, end2) > len(text):
                    continue

                if 0 <= start1 and 0 <= start2 and end1 < self.max_len - 2 and end2 < self.max_len - 2:
                    tmp_item = copy.deepcopy(item)
                    span1 = (start1, end1)
                    span2 = (start2, end2)
                    tmp_item["e1"] = (e1, et1, es1, span1)
                    tmp_item["e2"] = (e2, et2, es2, span2)
                    tmp_text = add_entity_tag(text, [span1, span2])
                    token_input = self.data_parser.get_token_input(tmp_text)
                    tmp_item.update(**token_input)
                    if rel:
                        rel_output = self.context.rel2id[rel]
                        tmp_item['rel_output'] = rel_output
                    rs_data.append(tmp_item)
        self.logger.info("get {0} enhanced data from {1} origin data".format(len(rs_data), len(data)))
        return rs_data

    def _get_short_data(self, data):
        return self.data_parser.get_short_data(data, self.max_len - 4)

    def train_model(self, train_data, dev_data, train_args, compile_args):
        """

        :param dev_data:
        :param train_data: [{"title":"test", "content":"The band performs with a high level of musicality , energy and spirit while combining sensitive group interplay with dynamic solo improvisations.",
                            "rel_list":[["Other",["band","ENTITY",[4,8]],["musicality","ENTITY",[39,49]]]]}]
        :param train_args:
        :return: model
        """
        return super().train_model(train_data, dev_data, train_args, compile_args)

    def _get_predict_data_from_model_output(self, origin_data, enhanced_data, pred_data, show_detail=False):
        rs_dict = defaultdict(list)
        pred_hard = np.argmax(pred_data, axis=-1)
        if show_detail:
            print("raw ner output:\n{}".format(pred_hard))
        for rel, item in zip(pred_hard, enhanced_data):
            idx = item['idx']
            rel_name = self.context.id2rel[rel]
            rel_pair = (rel_name, item['e1'][:-1], item['e2'][:-1])
            rs_dict[idx].append(rel_pair)
        rel_list = [rs_dict.get(idx, []) for idx in range(len(origin_data))]
        return rel_list


