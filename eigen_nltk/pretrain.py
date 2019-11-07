# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     pretrain
   Description :
   Author :       chenhao
   date：          2019-09-28
-------------------------------------------------
   Change Activity:
                   2019-09-28:
-------------------------------------------------
"""
import codecs
import os

from keras.losses import sparse_categorical_crossentropy
from keras.models import load_model
from keras_bert import load_trained_model_from_checkpoint, get_model, gen_batch_inputs

from eigen_nltk.core import ModelEstimator
from eigen_nltk.decorator import ensure_dir_path
from eigen_nltk.model_utils import get_base_customer_objects
from eigen_nltk.optimizer import get_optimizer_cls
from eigen_nltk.trans import DataParser
from eigen_nltk.utils import flat, split_by_len


def get_pre_train_customer_objects():
    base_customer_objects = get_base_customer_objects()
    return base_customer_objects


class BertPreTrainer(ModelEstimator):

    def __init__(self, name, context, max_len, logger_level="INFO"):
        self.name = name
        self.context = context
        self.max_len = max_len
        self.data_parser = DataParser(context)
        self.token_dict = context.token2id
        self.token_list = list(self.token_dict.keys())
        self.max_rate = 0.15
        self.swap_sentence_rate = 0.5
        super().__init__(name, self.data_parser, logger_level)

    def _build_model(self, bert_ckpt_path, bert_keras_path, **kwargs):
        if bert_ckpt_path:
            bert_model = load_trained_model_from_checkpoint(os.path.join(bert_ckpt_path, "bert_config.json"),
                                                            os.path.join(bert_ckpt_path, "bert_model.ckpt"),
                                                            training=True,
                                                            trainable=True,
                                                            seq_len=self.max_len)
        elif bert_keras_path:
            bert_model = load_model(bert_ckpt_path)
        else:
            bert_model = get_model(
                token_num=self.context.vocab_size,
                head_num=2,
                transformer_num=1,
                embed_dim=4,
                feed_forward_dim=4,
                seq_len=self.max_len,
                pos_num=self.max_len,
                dropout_rate=0.05,
            )
        return bert_model

    def _compile_model(self, optimizer_name, optimizer_args, **kwargs):
        super()._compile_model(**kwargs)
        opt_cls = get_optimizer_cls(optimizer_name)
        optimizer = opt_cls(**optimizer_args)
        self.training_model.compile(optimizer, loss=sparse_categorical_crossentropy)

    def _get_sentence_pair(self, document):
        token_list = self.context.tokenizer.tokenize(document)[1:-1]
        sent_len = (self.max_len - 2) // 2
        sentence_list = split_by_len(token_list, sent_len)
        sentence_pair_list = []
        for idx in range(len(sentence_list) - 1):
            sentence_pair_list.append((sentence_list[idx], sentence_list[idx + 1]))
        return sentence_pair_list

    def _get_model_train_input(self, train_data):
        return gen_batch_inputs(
            train_data,
            self.token_dict,
            self.token_list,
            seq_len=self.max_len,
            mask_rate=self.max_rate,
            swap_sentence_rate=self.swap_sentence_rate,
        )

    def train_model_generator(self, train_data, dev_data, train_args, compile_args):
        self.max_rate = train_args.get("mask_rate", 0.15)
        self.swap_sentence_rate = train_args.get("swap_sentence_rate", 0.5)
        super().train_model_generator(train_data, dev_data, train_args, compile_args)

    @ensure_dir_path
    def save_model(self, path):
        self.logger.info("saving model to dir:{}".format(path))
        model_path = os.path.join(path, "model.h5")
        self.model.save(model_path)
        vocab_path = os.path.join(path, "vocab.txt")
        with codecs.open(vocab_path, "w", "utf8") as f:
            f.write("\n".join(self.context.vocab_list))

    def _get_model_test_input(self, test_data):
        pass

    def _get_predict_data_from_model_output(self, origin_data, enhanced_data, pred_data, show_detail=False):
        pass

    def _get_enhanced_data(self, data):
        return flat([self._get_sentence_pair(e) for e in data])
