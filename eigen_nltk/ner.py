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
import math
from tqdm import tqdm
import codecs
from collections import defaultdict
from itertools import groupby

from keras import Model
from keras.layers import *
from keras.losses import sparse_categorical_crossentropy, binary_crossentropy
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy, crf_viterbi_accuracy

from eigen_nltk.constants import *
from eigen_nltk.core import ModelEstimator, Context
from eigen_nltk.model_utils import CRF, get_seq_embedding_model, get_base_customer_objects
from eigen_nltk.optimizer import get_optimizer_cls
from eigen_nltk.trans import DataParser, add_entity_tag
from eigen_nltk.utils import read_id_mapping, padding_seq, add_offset, flat, convert_span


class NerContext(Context):
    def __init__(self, vocab_path, ner_dict_path, ner_annotation_type, pos_vocab_path=None):
        super().__init__(vocab_path)
        self.id2ner, self.ner2id = read_id_mapping(ner_dict_path)
        self.ner_size = len(self.id2ner)
        self.ner_annotation_type = ner_annotation_type
        self.annotation_dim = len(self.ner_annotation_type) - 1
        self.entity_type_list = []
        for idx in range(1, self.ner_size, self.annotation_dim):
            entity_type = self.id2ner[idx].split("_")[1]
            self.entity_type_list.append(entity_type)
        self.entity_type_num = len(self.entity_type_list)
        self.pos_size = 0
        if pos_vocab_path:
            self.pos2id = {}
            with codecs.open(pos_vocab_path, "r", "utf8") as f:
                for line in f:
                    self.pos2id[line.strip()] = len(self.pos2id)
            self.pos_size = len(self.pos2id)
            self.id2pos = {v: k for k, v in self.pos2id.items()}

    def get_model_args(self):
        model_args = super().get_model_args()
        model_args.update(ner_size=self.ner_size)
        return model_args


def get_ner_customer_objects():
    ner_customer_objects = get_base_customer_objects()
    ner_customer_objects.update(CRF=CRF,
                                crf_loss=crf_loss, crf_accuracy=crf_accuracy, crf_viterbi_accuracy=crf_viterbi_accuracy)
    return ner_customer_objects


class NerExtractor(ModelEstimator):
    customer_objects = get_ner_customer_objects()
    tf_serving_input_keys = ['x', 'seg', 'pos']
    tf_serving_output_keys = ['ner0', 'ner1', 'ner2']

    def __init__(self, name, context, max_len, logger_level="INFO"):
        assert isinstance(context, NerContext)
        self.context = context
        self.data_parser = NerDataParser(context)
        self.max_len = max_len
        self.ner_annotation_type = self.context.ner_annotation_type
        self.vocab_size = self.context.vocab_size
        self.ner_size = self.context.ner_size
        self.pos_size = self.context.pos_size
        super().__init__(name, self.data_parser, logger_level)

    def _build_model(self, use_bert=True, fine_tune_bert=False, use_lstm=False, use_crf=True,
                     word_embedding_dim=32, lstm_dim=32, pos_embedding_dim=32, freeze_layer_num=0, head_num=1,
                     crf_constraint_trans_path=None, bert_ckpt_path=None, bert_keras_path=None, **kwargs):
        seq_embedding_model = get_seq_embedding_model(self.max_len, self.vocab_size, self.pos_size,
                                                      freeze_layer_num, word_embedding_dim, lstm_dim, pos_embedding_dim,
                                                      use_bert, fine_tune_bert, use_lstm,
                                                      bert_ckpt_path, bert_keras_path)

        input_feature = seq_embedding_model.inputs
        feature = seq_embedding_model.output
        ner_out = []
        for idx in range(head_num):
            if use_crf:
                crf_constraint_trans_matrix = np.load(crf_constraint_trans_path) if crf_constraint_trans_path else None
                crf = CRF(self.ner_size, chain_kernel_constant_matrix=crf_constraint_trans_matrix, use_bias=True,
                          use_boundary=True, test_mode="viterbi", name="ner{}".format(idx), sparse_target=True)
                ner_type = crf(feature)
            else:
                ner_type = Dense(self.ner_size, activation="softmax", name="ner{}".format(idx))(feature)
            ner_out.append(ner_type)
        ner_out = ner_out[0] if len(ner_out) == 1 else ner_out
        model = Model(input_feature, ner_out)
        self.head_num = head_num
        return model

    def _compile_model(self, optimizer_name, optimizer_args, **kwargs):
        use_crf = self.model_args['use_crf']
        opt_cls = get_optimizer_cls(optimizer_name)
        optimizer = opt_cls(**optimizer_args)
        mini_loss = crf_loss if use_crf else sparse_categorical_crossentropy
        loss = {"ner{}".format(idx): mini_loss for idx in range(self.head_num)}
        if "loss_weights" in kwargs:
            loss_weights = {"ner{}".format(idx): weight for idx, weight in enumerate(kwargs['loss_weights'])}
        else:
            loss_weights = {"ner{}".format(idx): 1. / self.head_num for idx in range(self.head_num)}

        self.training_model.compile(
            optimizer=optimizer,
            loss=loss,
            loss_weights=loss_weights
        )
        return self.training_model

    def _get_model_train_input(self, train_data):
        x = []
        seg = []
        pos = []
        y = []
        head_num = 1
        for item in train_data:
            x.append(padding_seq(item['x'], self.max_len))
            seg.append(padding_seq(item['seg'], self.max_len))
            if 'pos_input' in item.keys():
                pos.append(padding_seq(item['pos_input'], self.max_len))
            if 'ner_output' in item.keys():
                tmp_head_num = item.get("head_num", 1)
                if tmp_head_num > 1:
                    head_num = tmp_head_num
                    tmp_y = [padding_seq(e, self.max_len) for e in item['ner_output']]
                else:
                    tmp_y = padding_seq(item['ner_output'], self.max_len)
                y.append(tmp_y)
        input_list = [np.array(x), np.array(seg)]
        if pos:
            input_list.append(np.array(pos))
        if y:
            if head_num > 1:
                y = [[e[idx] for e in y] for idx in range(head_num)]
                y = [np.expand_dims(np.array(e), 2) for e in y]
            else:
                y = np.expand_dims(np.array(y), 2)
        return input_list, y

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
            pos = item['pos']
            prefix = item.get("prefix", None)
            token_input = self.data_parser.get_token_input(content, prefix, pos)
            token = token_input['token']
            char2token = token_input['char2token_mapping']
            tmp_item.update(**token_input)

            if "entity_list" in item.keys():
                entity_list = item['entity_list']
                if item.get("head_num", 1) > 1:
                    tmp_item["ner_output"] = []
                    for e_list in entity_list:
                        e_list = [(e[0], e[1], [add_offset(span, -offset) for span in e[2]]) for e in e_list]
                        ner_output = self.data_parser.get_ner_output(token, e_list, char2token,
                                                                     self.ner_annotation_type)
                        tmp_item["ner_output"].append(ner_output)
                else:
                    entity_list = [(e[0], e[1], [add_offset(span, -offset) for span in e[2]]) for e in entity_list]
                    ner_output = self.data_parser.get_ner_output(token, entity_list, char2token,
                                                                 self.ner_annotation_type)
                    tmp_item["ner_output"] = ner_output
            enhance_data.append(tmp_item)
        self.logger.info("get {0} enhanced data from {1} origin data".format(len(enhance_data), len(data)))
        return enhance_data

    def _get_short_data(self, data):
        return self.data_parser.get_short_data(data, self.max_len)

    def _get_predict_data_from_model_output(self, origin_data, enhanced_data, pred_data, show_detail=False, **kwargs):
        single_tag = False
        if not isinstance(pred_data, list) and not isinstance(pred_data, tuple):
            single_tag = True
            pred_data = [pred_data]
        rs_list = []
        for pred in pred_data:
            pred_hard = np.argmax(pred, axis=-1)
            if show_detail:
                print("raw ner output:\n{}".format(pred_hard))
            entity_list_pred = [self.data_parser.get_ner_from_output(item['content'], pred, item['token2char_mapping'])
                                for
                                item, pred in zip(enhanced_data, pred_hard)]
            merged_entity_list_pred = merge_entity_lists(entity_list_pred, enhanced_data)
            rs_list.append(merged_entity_list_pred)
        if single_tag:
            rs_list = rs_list[0]
        else:
            rs_list = [[rs_list[j][i] for j in range(len(rs_list))] for i in range(len(enhanced_data))]
        return rs_list


class EntityPairNerExtractor(NerExtractor):

    def _build_model(self, use_bert=True, fine_tune_bert=False, use_lstm=False,
                     word_embedding_dim=32, lstm_dim=32, freeze_layer_num=0,
                     crf_constraint_trans_path=None, bert_ckpt_path=None, bert_keras_path=None, **kwargs):
        seq_embedding_model = get_seq_embedding_model(self.max_len, self.vocab_size,
                                                      freeze_layer_num, word_embedding_dim, lstm_dim,
                                                      use_bert, fine_tune_bert, use_lstm,
                                                      bert_ckpt_path, bert_keras_path)

        input_feature = seq_embedding_model.inputs
        feature = seq_embedding_model.output
        output = Dense(4, activation="sigmoid")(feature)
        model = Model(input_feature, output)
        return model

    def _compile_model(self, optimizer_name, optimizer_args, **kwargs):
        opt_cls = get_optimizer_cls(optimizer_name)
        optimizer = opt_cls(**optimizer_args)

        self.training_model.compile(optimizer, loss=binary_crossentropy, metrics=["accuracy"])
        return self.training_model

    def _get_enhanced_data(self, data):
        enhanced_data = super()._get_enhanced_data(data)
        rs_list = []
        for idx, item in enumerate(enhanced_data):
            tmp_item = copy.copy(item)
            if "entity_list" in item.keys():
                offset = item['offset']
                token = item['token']
                entity_list = item['entity_list']
                char2token = item['char2token_mapping']
                entity_list = [(e[0], e[1], [add_offset(span, -offset) for span in e[2]]) for e in entity_list]
                pair_ner_output = []
                for entity, entity_type, span_list in entity_list:
                    span_list = [add_offset(span, -offset) for span in span_list]
                    span_list = [(s, e) for (s, e) in span_list if s >= 0 and e < len(char2token)]
                    if span_list:
                        s = char2token[span_list[0][0]]
                        e = char2token[span_list[0][1]]
                        start_list = [0] * len(token)
                        start_list[s] = 1
                        end_list = [0] * len(token)
                        end_list[e] = 1
                        pair_ner_output.append(start_list)
                        pair_ner_output.append(end_list)
                if len(pair_ner_output) != 4:
                    continue
                tmp_item["pair_ner_output"] = pair_ner_output
            rs_list.append(tmp_item)
        self.logger.info("get {0} enhanced data from {1} super enhanced data".format(len(rs_list), len(enhanced_data)))
        return rs_list

    def _get_model_train_input(self, train_data):
        x = []
        seg = []
        y = []
        for item in train_data:
            x.append(padding_seq(item['x'], self.max_len))
            seg.append(padding_seq(item['seg'], self.max_len))
            if 'pair_ner_output' in item.keys():
                y.append([padding_seq(e, self.max_len) for e in item['pair_ner_output']])

        x = np.array(x)
        seg = np.array(seg)
        if y:
            y = np.array(y).swapaxes(1, 2)
        return [x, seg], y

    def _get_predict_data_from_model_output(self, origin_data, enhanced_data, pred_data, show_detail=False):
        entity_list_pred = []
        for item, pred in zip(enhanced_data, pred_data):
            token_len = len(item['mapping_token'])
            e1_path_list = get_max_ner_pair(pred[:, :2], 0, 0, token_len)[1]
            e2_path_list = get_max_ner_pair(pred[:, 2:], 0, 0, token_len)[1]
            entity_idx_path_list = e1_path_list + e2_path_list
            if show_detail:
                print("raw prob output:\n{}".format(pred))
                print("entity token idx:{}".format(entity_idx_path_list))
            entity_pred = self.data_parser.get_ner_from_pair_output(item['content'], entity_idx_path_list,
                                                                    item['token2char_mapping'], item['token_offset'])
            entity_list_pred.append(entity_pred)
        # merged_entity_list_pred = merge_entity_lists(entity_list_pred, enhanced_data)
        return entity_list_pred


def get_max_ner_pair(prob_matrix, offset, idx, token_len):
    if idx == 2:
        return 0., []
    if offset + (2 - idx) > token_len:
        return math.inf, []
    loss_a, path_a = get_max_ner_pair(prob_matrix, offset + 1, idx, token_len)
    cur_value = - math.log(prob_matrix[offset][idx])
    loss_b, path_b = get_max_ner_pair(prob_matrix, offset + 1, idx + 1, token_len)
    loss_b += cur_value
    path_b = [offset] + path_b
    if loss_a < loss_b:
        return loss_a, path_a
    return loss_b, path_b


class NerDataParser(DataParser):
    def __init__(self, context):
        assert isinstance(context, NerContext)
        self.context = context
        self.tokenizer = context.tokenizer

    # get ner model output from token and entity_list
    def get_ner_output(self, token, entity_list, char2token, annotation_type):
        assert annotation_type in NER_ANNOTATION_LIST
        ner_out = [0] * len(token)
        for ner, ner_type, span_list in entity_list:
            ner_key = "B_" + ner_type
            if ner_key not in self.context.ner2id.keys():
                continue
            ner_value = self.context.ner2id[ner_key]
            for start, end in span_list:
                if start < 0 or end >= len(char2token):
                    continue
                start, end = char2token[start], char2token[end]
                if start < 0 or end > len(token):
                    continue
                if len([e for e in ner_out[start: end] if e > 0]) == end - start:
                    continue

                if annotation_type == BIO:
                    ner_out[start] = ner_value
                    ner_out[start + 1: end] = [ner_value + 1] * (end - start - 1)
                elif annotation_type == BMESO:
                    if end - start == 1:
                        ner_out[start] = ner_value + 3
                    else:
                        ner_out[start] = ner_value
                        ner_out[end - 1] = ner_value + 2
                        ner_out[start + 1: end - 1] = [ner_value + 1] * (end - start - 2)
        return ner_out

    def get_ner_from_token_span(self, text, ner_token_span_list, token2char):
        ner_char_span_list = [(ner_type, convert_span(span, token2char)) for ner_type, span
                              in ner_token_span_list]
        ner_char_span_list = [e for e in ner_char_span_list if e[1]]

        entity_list = []
        for ner_type, (start, end) in ner_char_span_list:
            if start < 0 or end < 0:
                continue
            while start < len(text) and text[start] == " ":
                start += 1
            while end > 0 and text[end - 1] == " ":
                end -= 1
            entity_list.append((text[start:end], ner_type, (start, end)))
        return entity_list

    def get_ner_from_pair_output(self, text, pair_idx_path, token2char):
        assert self.context.entity_type_num == 1
        entity_type = self.context.entity_type_list[0]
        ner_token_span_list = [(entity_type, (pair_idx_path[0], pair_idx_path[1])),
                               (entity_type, (pair_idx_path[2], pair_idx_path[3]))]
        entity_list = self.get_ner_from_token_span(text, ner_token_span_list, token2char)
        # print(entity_list)
        entity_list = [(e, et, [span]) for e, et, span in entity_list]
        return entity_list

    # get ner list from ner model output
    def get_ner_from_output(self, text, ner_idx_list, token2char):
        ner_token_span_list = self.get_ner_span(ner_idx_list, self.context.ner_annotation_type)
        entity_list = self.get_ner_from_token_span(text, ner_token_span_list, token2char)
        entity_list = group_entity_list(entity_list)
        return entity_list

    # get the span of token from model's ner index list
    def get_ner_span_bmeso(self, ner_idx_list):
        idx = 1
        rs_list = []
        while idx < len(ner_idx_list):
            ner_code = ner_idx_list[idx]
            if ner_code == 0:
                idx += 1
                continue
            ner_type = self.context.id2ner[ner_code].split("_")[1]
            if ner_code % 4 == 0:
                if ner_code != 0:
                    rs_list.append((ner_type, (idx, idx + 1)))
                idx += 1
                continue
            if ner_code % 4 == 1:
                j = idx + 1
                while j < len(ner_idx_list) and ner_idx_list[j] == ner_code + 1:
                    j += 1
                if j < len(ner_idx_list) and ner_idx_list[j] == ner_code + 2:
                    j += 1
                    rs_list.append((ner_type, (idx, j)))
                idx = j
                continue
            idx += 1
        return rs_list

    # get the span of token from model's ner index list
    def get_ner_span_bio(self, ner_idx_list):
        idx = 1
        rs_list = []
        while idx < len(ner_idx_list):
            ner_code = ner_idx_list[idx]
            if ner_code == 0:
                idx += 1
                continue
            ner_type = self.context.id2ner[ner_code].split("_")[1]
            if ner_code % 2 == 1:
                j = idx + 1
                while j < len(ner_idx_list) and ner_idx_list[j] == ner_code + 1:
                    j += 1
                rs_list.append((ner_type, (idx, j)))
                idx = j
                continue
            idx += 1
        return rs_list

    def get_ner_span(self, ner_idx_list, ner_annotation_type):
        assert ner_annotation_type in NER_ANNOTATION_LIST
        if ner_annotation_type == BIO:
            return self.get_ner_span_bio(ner_idx_list)
        if ner_annotation_type == BMESO:
            return self.get_ner_span_bmeso(ner_idx_list)


# group ner list by the ner_name and ner_type
def group_entity_list(entity_list):
    rs_dict = defaultdict(list)
    for ner, ner_type, ner_span in entity_list:
        rs_dict[(ner, ner_type)].append(ner_span)
    return [(*k, v) for k, v in rs_dict.items()]


# merge ner outputs to item level
def merge_entity_lists(entity_lists, item_lists):
    def merge_group(ner_offset_list):
        rs = []
        for ner, offset in ner_offset_list:
            rs.extend(
                [(entity, ner_type, [add_offset(span, offset) for span in span_list]) for entity, ner_type, span_list in
                 ner])
        grouped = groupby(rs, key=lambda x: (x[0], x[1]))
        rs = [(k[0], k[1], sorted(flat([e[2] for e in v]))) for k, v in grouped]
        return rs

    idx_ner_dict = defaultdict(list)
    for entity_list, item in zip(entity_lists, item_lists):
        idx = item['idx']
        idx_ner_dict[idx].append((entity_list, item['offset']))
    return [merge_group(idx_ner_dict[k]) for k in sorted(idx_ner_dict.keys())]


cls_dict = {
    "EntityPairNerExtractor": EntityPairNerExtractor,
    "NerExtractor": NerExtractor
}


def get_ner_cls(cls_name):
    return cls_dict.get(cls_name, NerExtractor)
