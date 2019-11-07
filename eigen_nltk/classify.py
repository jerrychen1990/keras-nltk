# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     classify
   Description :
   Author :       chenhao
   date：          2019-09-20
-------------------------------------------------
   Change Activity:
                   2019-09-20:
-------------------------------------------------
"""

from keras.layers import *
from keras.losses import sparse_categorical_crossentropy
from keras.models import Model
from keras.regularizers import l1_l2

from eigen_nltk.core import Context, ModelEstimator
from eigen_nltk.model_utils import get_seq_embedding_model, get_base_customer_objects
from eigen_nltk.optimizer import get_optimizer_cls
from eigen_nltk.trans import DataParser
from eigen_nltk.utils import read_id_mapping, padding_seq


class ClassifyContext(Context):
    def __init__(self, vocab_path, label_dict_path):
        super().__init__(vocab_path)
        self.id2label, self.label2id = read_id_mapping(label_dict_path)
        assert len(self.id2label) == len(self.label2id)
        self.label_size = len(self.id2label)

    def get_model_args(self):
        model_args = super().get_model_args()
        model_args.update(label_size=self.label_size)
        return model_args


class ClassifyDataParser(DataParser):
    def __init__(self, context):
        assert isinstance(context, ClassifyContext)
        super().__init__(context)


def get_classify_custom_objects():
    customer_objects = get_base_customer_objects()
    return customer_objects


class ClassifyEstimator(ModelEstimator):
    customer_objects = get_classify_custom_objects()

    def __init__(self, name, context, max_len, logger_level="INFO"):
        assert isinstance(context, ClassifyContext)
        data_parser = ClassifyDataParser(context)
        self.max_len = max_len
        self.context = context
        self.label_size = context.label_size
        self.vocab_size = context.vocab_size
        super().__init__(name, data_parser, logger_level)

    def _build_model(self, use_bert=False, fine_tune_bert=False, use_lstm=False,
                     word_embedding_dim=32, lstm_dim=32, dense_dim_list=[], drop_out_rate=0.2, l1=0.01, l2=0.01,
                     freeze_layer_num=0,
                     bert_ckpt_path=None, bert_keras_path=None, **kwargs):

        seq_embedding_model = get_seq_embedding_model(self.max_len, self.vocab_size,
                                                      freeze_layer_num, word_embedding_dim, lstm_dim,
                                                      use_bert, fine_tune_bert, use_lstm,
                                                      bert_ckpt_path, bert_keras_path)

        words_input, seg_input = seq_embedding_model.inputs
        feature = seq_embedding_model.output
        if use_lstm and not use_bert:
            feature = Lambda(lambda t: t[:, -1, :])(feature)
        else:
            feature = Lambda(lambda t: t[:, 0, :])(feature)

        for dim in dense_dim_list:
            feature = Dense(dim, activation="relu")(feature)
        feature = Dropout(rate=drop_out_rate)(feature)
        label_out = Dense(self.label_size, activation="softmax", kernel_regularizer=l1_l2(l1, l2))(feature)
        model = Model([words_input, seg_input], label_out)
        return model

    def _compile_model(self, optimizer_name, optimizer_args, **kwargs):
        opt_cls = get_optimizer_cls(optimizer_name)
        optimizer = opt_cls(**optimizer_args)
        self.training_model.compile(optimizer, loss=sparse_categorical_crossentropy, metrics=["accuracy"])
        return self.training_model

    def _get_model_train_input(self, train_data):
        x = []
        seg = []
        y = []
        for item in train_data:
            x.append(padding_seq(item['x'], self.max_len))
            seg.append(padding_seq(item['seg'], self.max_len))
            if 'label' in item.keys():
                y.append(self.context.label2id[str(item['label'])])
        x = np.array(x)
        seg = np.array(seg)
        if y:
            y = np.array(y)[:, np.newaxis]
        return [x, seg], y

    def _get_model_test_input(self, test_data):
        [x, seg], y = self._get_model_train_input(test_data)
        return [x, seg]

    # add more information to the origin data
    def _get_enhanced_data(self, data):
        rs_list = []
        for idx, item in enumerate(data):
            text = item['content']
            tmp_item = copy.copy(item)
            bert_input = self.data_parser.get_bert_input(text)
            tmp_item.update(**bert_input)
            rs_list.append(tmp_item)
        return rs_list

    def _get_predict_data_from_model_output(self, origin_data, enhanced_data, pred_data, show_detail=False, soft=False,
                                            label=True):
        pred_idx = np.argmax(pred_data, axis=-1)
        pred_label = [self.context.id2label[e] for e in pred_idx]
        if soft:
            return pred_data
        if label:
            return pred_label
        return pred_idx
