# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     language_model
   Description :
   Author :       chenhao
   date：          2019-11-13
-------------------------------------------------
   Change Activity:
                   2019-11-13:
-------------------------------------------------
"""
from eigen_nltk.core import Context, ModelEstimator
from eigen_nltk.trans import DataParser, remove_token_char
from eigen_nltk.model_utils import get_lm_decoder_model
from eigen_nltk.utils import padding_seq, list_find
from eigen_nltk.optimizer import get_optimizer_cls
from keras.losses import sparse_categorical_crossentropy
from keras.layers import *


class LMContext(Context):
    def __init__(self, vocab_path):
        super().__init__(vocab_path)


class TransformerLM(ModelEstimator):

    def __init__(self, name, context, logger_level="INFO"):
        assert isinstance(context, LMContext)
        self.context = context
        self.vocab_size = self.context.vocab_size
        data_parser = DataParser(context)
        super().__init__(name, data_parser, logger_level)

    def _build_model(self, embedding_dim, decoder_block_num, head_num, hidden_dim, embed_trainable=True, **kwargs):
        model = get_lm_decoder_model(vocab_size=self.vocab_size, embedding_dim=embedding_dim,
                                     decoder_num=decoder_block_num, head_num=head_num,
                                     hidden_dim=hidden_dim, embed_trainable=embed_trainable)

        return model

    def _compile_model(self, optimizer_name, optimizer_args, **kwargs):
        opt_cls = get_optimizer_cls(optimizer_name)
        optimizer = opt_cls(**optimizer_args)
        self.training_model.compile(optimizer, loss=sparse_categorical_crossentropy)
        return self.training_model

    def _get_predict_data_from_model_output(self, origin_data, enhanced_data, pred_data, show_detail=False):
        hard_pred = pred_data.argmax(axis=2).tolist()
        text_pred = [self._id_list2text(id_list) for id_list in hard_pred]
        return text_pred

    def _get_enhanced_data(self, data):
        enhance_data = []
        for idx, line in enumerate(data):
            tmp_item = dict(content=line)
            bert_input = self.data_parser.get_bert_input(line)
            input_token = bert_input['token'][:-1]
            output_token = bert_input["token"][1:]
            x = bert_input['x'][:-1]
            y = bert_input['x'][1:]
            tmp_item.update(input_token=input_token, output_token=output_token, x=x, y=y)
            enhance_data.append(tmp_item)
        self.logger.info("get {0} enhanced data from {1} origin data".format(len(enhance_data), len(data)))
        return enhance_data

    def _get_model_train_input(self, train_data, **kwargs):
        x = []
        y = []
        max_len = max(len(e['x']) for e in train_data)
        for item in train_data:
            x.append(padding_seq(item['x'], max_len))
            if 'y' in item.keys():
                y.append(padding_seq(item['y'], max_len))

        x = np.array(x)
        if y:
            y = np.array(y)[:, :, np.newaxis]
        return x, y

    def _id_list2text(self, id_list):
        token_list = self._id_list2token_list(id_list)
        text = "".join([remove_token_char(t) for t in token_list])
        return text

    def _id_list2token_list(self, id_list):
        token_list = [self.context.id2token[i] for i in id_list]
        return token_list

    def predict_next_token(self, data, batch_size=64, verbose=1):
        raw_pred = self._get_raw_predict(data, batch_size=batch_size, verbose=verbose)
        idx_pred = raw_pred.argmax(axis=2)[:, -1]
        token_pred = [self.context.id2token[i] for i in idx_pred]
        return token_pred

    def generate_sequence(self, data, max_len, batch_size=64, verbose=1, end_token='[SEP]'):
        enhance_data = self._get_enhanced_data(data)
        x = self._get_model_test_input(enhance_data)
        cur_len = x.shape[1]
        for i in range(max_len - cur_len):
            # print(x.shape)
            raw_pred = self.model.predict(x, batch_size=batch_size, verbose=verbose)
            last_pred = raw_pred.argmax(axis=2)[:, -1][:, np.newaxis]
            # print(last_pred.shape)
            x = np.concatenate([x, last_pred], axis=1)
        token_list = [self._id_list2token_list(id_list) for id_list in x]
        text_list = []
        print(token_list)
        for tokens in token_list:
            end_idx = list_find(tokens, end_token)
            tokens = tokens[1:] if end_idx == -1 else tokens[1:end_idx]
            text = "".join([remove_token_char(t) for t in tokens])
            text_list.append(text)

        return text_list
