# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     ner
   Description :
   Author :       chenhao
   date：          2019-09-19
-------------------------------------------------
   Change Activity:
                   2019-09-19:
-------------------------------------------------
"""

from keras import Model
from keras.layers import *
from keras.losses import sparse_categorical_crossentropy, binary_crossentropy
from keras.regularizers import l1_l2
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy, crf_viterbi_accuracy
from tqdm import tqdm

from eigen_nltk.classify import merge_label_lists
from eigen_nltk.core import ModelEstimator
from eigen_nltk.model_utils import CRF, get_seq_embedding_model, get_base_customer_objects
from eigen_nltk.ner import NerDataParser, NerContext, merge_entity_lists
from eigen_nltk.optimizer import get_optimizer_cls
from eigen_nltk.utils import read_id_mapping, padding_seq, add_offset, array2zero_one_with_threshold


class NerClassifyContext(NerContext):
    def __init__(self, vocab_path, ner_dict_path, ner_annotation_type, label_dict_path):
        super().__init__(vocab_path, ner_dict_path, ner_annotation_type)
        self.id2label, self.label2id = read_id_mapping(label_dict_path)
        assert len(self.id2label) == len(self.label2id)
        self.label_size = len(self.id2label)

    def get_model_args(self):
        model_args = super().get_model_args()
        model_args.update(ner_size=self.ner_size, label_size=self.label_size)
        return model_args


def get_ner_classify_customer_objects():
    ner_customer_objects = get_base_customer_objects()
    ner_customer_objects.update(loss=crf_loss, accuracy=crf_accuracy, CRF=CRF,
                                crf_loss=crf_loss, crf_accuracy=crf_accuracy, crf_viterbi_accuracy=crf_viterbi_accuracy)
    return ner_customer_objects


class NerClassifyExtractor(ModelEstimator):
    customer_objects = get_ner_classify_customer_objects()
    cache_keys = ['x', 'ner_output']

    def __init__(self, name, context, max_len, logger_level="INFO"):
        assert isinstance(context, NerClassifyContext)
        self.context = context
        self.data_parser = NerDataParser(context)
        self.max_len = max_len
        self.ner_annotation_type = self.context.ner_annotation_type
        self.vocab_size = self.context.vocab_size
        self.ner_size = self.context.ner_size
        self.label_size = self.context.label_size
        self.label2id = self.context.label2id
        self.id2label = self.context.id2label
        super().__init__(name, self.data_parser, logger_level)

    def _build_model(self, use_bert=True, fine_tune_bert=False, use_lstm=False, use_crf=True,
                     word_embedding_dim=32, lstm_dim=32, freeze_layer_num=0,
                     dense_dim_list=[], drop_out_rate=0.2, l1=0.01, l2=0.01,
                     crf_constraint_trans_path=None, bert_ckpt_path=None, bert_keras_path=None, **kwargs):
        seq_embedding_model = get_seq_embedding_model(self.max_len, self.vocab_size,
                                                      freeze_layer_num, word_embedding_dim, lstm_dim,
                                                      use_bert, fine_tune_bert, use_lstm,
                                                      bert_ckpt_path, bert_keras_path)

        input_feature = seq_embedding_model.inputs
        feature = seq_embedding_model.output

        if use_lstm and not use_bert:
            cls_feature = Lambda(lambda t: t[:, -1, :])(feature)
        else:
            cls_feature = Lambda(lambda t: t[:, 0, :])(feature)

        for dim in dense_dim_list:
            cls_feature = Dense(dim, activation="relu")(cls_feature)
        cls_feature = Dropout(rate=drop_out_rate)(cls_feature)
        label_out = Dense(self.label_size, name="label_out", activation="softmax", kernel_regularizer=l1_l2(l1, l2))(
            cls_feature)

        if use_crf:
            crf_constraint_trans_matrix = np.load(crf_constraint_trans_path) if crf_constraint_trans_path else None
            crf = CRF(self.ner_size, chain_kernel_constant_matrix=crf_constraint_trans_matrix, use_bias=True,
                      use_boundary=True,
                      test_mode="viterbi", name="ner_out", sparse_target=True)
            ner_out = crf(feature)
        else:
            ner_out = Dense(self.ner_size, activation="softmax", name="ner_out")(feature)
        model = Model(input_feature, [label_out, ner_out])
        return model

    def _compile_model(self, optimizer_name, optimizer_args, **kwargs):
        use_crf = self.model_args['use_crf']
        opt_cls = get_optimizer_cls(optimizer_name)
        optimizer = opt_cls(**optimizer_args)
        ner_loss = crf_loss if use_crf else sparse_categorical_crossentropy
        classify_loss = binary_crossentropy

        self.training_model.compile(
            optimizer=optimizer,
            loss={
                'ner_out': ner_loss,
                'label_out': classify_loss
            },
            loss_weights={
                'ner_out': .5,
                'label_out': 1.,
            }
        )

        return self.training_model

    def _get_model_train_input(self, train_data):
        x = []
        seg = []
        ner_y = []
        classify_y = []
        for item in train_data:
            x.append(padding_seq(item['x'], self.max_len))
            seg.append(padding_seq(item['seg'], self.max_len))
            if 'ner_output' in item.keys():
                ner_y.append(padding_seq(item['ner_output'], self.max_len))
            if 'label_id_list' in item.keys():
                tmp_y = [0] * self.label_size
                for idx in item['label_id_list']:
                    tmp_y[idx] = 1
                classify_y.append(tmp_y)

        x = np.array(x)
        seg = np.array(seg)
        if ner_y:
            ner_y = np.array(ner_y)[:, :, np.newaxis]
        if classify_y:
            classify_y = np.array(classify_y)

        return [x, seg], [classify_y, ner_y]

    def create_model(self, model_args):
        model_args["max_len"] = self.max_len
        super().create_model(model_args)

    def _get_enhanced_data(self, data):
        short_data = self._get_short_data(data)
        self.logger.info("get {0} short data from {1} origin data".format(len(short_data), len(data)))
        enhance_data = []

        for idx, item in tqdm(iterable=enumerate(short_data), mininterval=5):
            tmp_item = copy.copy(item)
            content = item['content']
            offset = item['offset']
            prefix = item.get("prefix", None)
            token_input = self.data_parser.get_token_input(content, prefix)
            token = token_input['token']
            char2token = token_input['char2token_mapping']
            tmp_item.update(**token_input)
            if "entity_list" in item.keys():
                entity_list = item['entity_list']
                entity_list = [(e[0], e[1], [add_offset(span, -offset) for span in e[2]]) for e in entity_list]
                ner_output = self.data_parser.get_ner_output(token, entity_list, char2token, self.ner_annotation_type)
                tmp_item["ner_output"] = ner_output
            if "label_list" in item.keys():
                label_id_list = [self.label2id[e] for e in item['label_list']]
                tmp_item['label_id_list'] = label_id_list
            enhance_data.append(tmp_item)
        self.logger.info("get {0} enhanced data from {1} origin data".format(len(enhance_data), len(data)))
        return enhance_data

    def _get_short_data(self, data):
        return self.data_parser.get_short_data(data, self.max_len)

    def _get_predict_data_from_model_output(self, origin_data, enhanced_data, pred_data, show_detail=False,
                                            threshold=0.5, ignore_label_id_list=[0], **kwargs):
        label_id_pred, ner_pred = pred_data
        label_id_pred_hard = array2zero_one_with_threshold(label_id_pred, threshold=threshold)
        # print(label_id_pred_hard)
        label_pred = [[self.id2label[idx] for idx, e in enumerate(seq) if e == 1 and idx not in ignore_label_id_list]
                      for seq in label_id_pred_hard]
        # print(label_pred)
        ner_pred_hard = np.argmax(ner_pred, axis=-1)
        if show_detail:
            print("raw ner output:\n{}".format(ner_pred_hard))
            print("raw classify output:\n{}".format(label_id_pred))

        entity_list_pred = [self.data_parser.get_ner_from_output(item['content'], pred, item['token2char_mapping']) for
                            item, pred in zip(enhanced_data, ner_pred_hard)]
        merged_entity_list_pred = merge_entity_lists(entity_list_pred, enhanced_data)
        merged_label_list_pred = merge_label_lists(label_pred, enhanced_data)
        assert len(merged_entity_list_pred) == len(merged_label_list_pred)
        rs_list = [dict(label_list=label_list, entity_list=entity_list) for label_list, entity_list in
                   zip(merged_label_list_pred, merged_entity_list_pred)]

        return rs_list
