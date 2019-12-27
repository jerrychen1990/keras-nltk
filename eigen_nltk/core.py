# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     api
   Description :
   Author :       chenhao
   date：          2019-09-19
-------------------------------------------------
   Change Activity:
                   2019-09-19:
-------------------------------------------------
"""

import codecs
import copy
import math
import pickle
import os
from abc import abstractmethod

import numpy as np
from keras.models import load_model as load_keras_model
from keras.utils.training_utils import multi_gpu_model

from eigen_nltk.constants import SPECIAL_TOKEN_LIST
from eigen_nltk.decorator import ensure_file_path
from eigen_nltk.constants import TF_DEFAULT_SIGNATURE_NAME
from eigen_nltk.model_utils import VALID_FIT_GENERATOR_KWARGS, get_base_customer_objects, export_keras_as_tf_file
from eigen_nltk.tokenizer import MyTokenizer
from eigen_nltk.utils import get_logger, jload, call_tf_service, compress_file, cut_list


class Context(object):
    def __init__(self, vocab_path):
        self.token2id = dict()
        self.vocab_list = []
        with codecs.open(vocab_path, "r", "utf8") as f:
            for line in f:
                line = line.strip()
                if line and line not in self.token2id.keys():
                    self.token2id[line] = len(self.token2id)
                    self.vocab_list.append(line)
        self.id2token = {int(v): k for k, v in self.token2id.items()}
        self.tokenizer = MyTokenizer(self.token2id, SPECIAL_TOKEN_LIST)
        self.vocab_size = len(self.id2token)

    def get_model_args(self):
        return dict(vocab_size=self.vocab_size)


class BaseEstimator(object):
    def __init__(self, name, logger_level="INFO"):
        self.name = name
        self.logger_level = logger_level
        self.logger = get_logger(name, self.logger_level)

    @abstractmethod
    def predict_batch(self, batch, **kwargs):
        self.logger.info("predicting {0} data with args:{1}".format(len(batch), kwargs))

        pass

    def predict_item(self, item, **kwargs):
        batch_rs = self.predict_batch([item], **kwargs)
        return batch_rs[0]


class ModelEstimator(BaseEstimator):
    customer_objects = get_base_customer_objects()
    tf_serving_input_keys = ['x', 'seg']
    tf_serving_output_keys = ['y']

    def __init__(self, name, data_parser, logger_level="INFO"):
        super().__init__(name, logger_level)
        self.model_name = name
        self.model = None
        self.model_args = None
        self.training_model = None
        self.data_parser = data_parser
        self.context = data_parser.context

    def __getstate__(self):
        odict = copy.copy(self.__dict__)
        for key in ['model', 'logger', 'training_model']:
            if key in odict.keys():
                del odict[key]
        return odict

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.logger = get_logger(self.name, self.logger_level)

    def create_model(self, model_args):
        all_args = dict(**model_args, **(self.context.get_model_args()))
        self.model_args = all_args
        self.model = self._build_model(**self.model_args)
        self.model.summary(print_fn=self.logger.info)
        return self.model

    @ensure_file_path
    def save_model(self, path):
        path = path + ".h5"
        self.logger.info("saving model to path:{}".format(path))
        self.model.save(path, include_optimizer=False)

    @ensure_file_path
    def save_tf_serving_model(self, path, version=0, do_compress=True):
        export_path = os.path.join(path, str(version))
        self.logger.info("saving tensorflow serving model to {}".format(export_path))
        export_keras_as_tf_file(model=self.model, input_keys=self.tf_serving_input_keys,
                                output_keys=self.tf_serving_output_keys, export_path=export_path)
        if do_compress:
            compress_path = path + ".tgz"
            self.logger.info("compress model file to path:{}".format(compress_path))
            compress_file(export_path, compress_path)
            return compress_path
        return path

    def save_estimator(self, path):
        self.logger.info("saving extractor to {}".format(path))
        self.save_model(path=path)
        pickle.dump(self, open(path + ".pkl", 'wb'))

    def load_model(self, path):
        self.logger.info("loading model from path:{}".format(path))
        self.model = load_keras_model(path, custom_objects=self.customer_objects)
        self.model.summary(print_fn=self.logger.info)

    @classmethod
    def load_estimator(cls, path, load_model=True):
        extractor = pickle.load(open(path + ".pkl", "rb"))
        extractor.logger.info("loading extractor from path:{}".format(path))
        if load_model:
            extractor.load_model(path + ".h5")
        return extractor

    @abstractmethod
    def _build_model(self, **kwargs):
        self.logger.info("creating model args:{}".format(kwargs))
        pass

    @abstractmethod
    def _compile_model(self, **kwargs):
        self.logger.info("compiling model args:{}".format(kwargs))

    def _pre_train(self, train_args, compile_args):
        gpu_num = train_args.get("gpu_num", 1)
        if gpu_num > 1:
            self.training_model = multi_gpu_model(self.model, gpus=gpu_num)
        else:
            self.training_model = self.model
        self._compile_model(**compile_args)

    def train_model(self, train_data, dev_data, train_args, compile_args):
        self.logger.info(
            "train model with {0} train_data and {1} dev_data, training args:{2}, compile_args:{3}".format(
                len(train_data), len(dev_data), train_args, compile_args))
        self._pre_train(train_args, compile_args)
        enhanced_train_data = self._get_enhanced_data(train_data)
        enhanced_dev_data = self._get_enhanced_data(dev_data)

        x, y = self._get_model_train_input(enhanced_train_data)
        x_dev, y_dev = self._get_model_train_input(enhanced_dev_data)
        if x[0].shape[0] == 0:
            self.logger.warn("no valid training sample, model will not train!")
            return self.model
        self.training_model.fit(x, y, validation_data=[x_dev, y_dev], **train_args)
        return self.model

    def _generate_train_input(self, data, batch_size, **kwargs):
        np.random.shuffle(data)
        for idx in range(0, len(data), batch_size):
            batch_data = data[idx: idx + batch_size]
            yield self._get_model_train_input(batch_data, **kwargs)

    def _get_generator(self, train_data, batch_size, **kwargs):
        while True:
            generator = self._generate_train_input(train_data, batch_size, **kwargs)
            for i in generator:
                yield i

    def _get_generator_from_path(self, data_path_list, batch_size, **kwargs):
        while True:
            for data_path in data_path_list:
                self.logger.info("reading train data from:{}".format(data_path))
                data = jload(data_path)
                enhanced_data = self._get_enhanced_data(data)
                gen = self._generate_train_input(enhanced_data, batch_size, **kwargs)
                for batch in gen:
                    yield batch

    def train_model_generator(self, train_data, dev_data, train_args, compile_args, is_raw=True):
        self.logger.info("training model with generator")
        self._pre_train(train_args, compile_args)
        batch_size = train_args['batch_size']
        if is_raw:
            self.logger.info("enhancing train data")
            train_data = self._get_enhanced_data(train_data)

            self.logger.info("enhancing dev data")
            dev_data = self._get_enhanced_data(dev_data)
        kwargs = dict((k, v) for k, v in train_args.items() if k in VALID_FIT_GENERATOR_KWARGS)
        validation_steps = int(math.ceil(len(dev_data) / batch_size))

        self.training_model.fit_generator(
            generator=self._get_generator(train_data=train_data, batch_size=batch_size),
            validation_data=self._get_generator(train_data=dev_data, batch_size=batch_size),
            validation_steps=validation_steps,
            **kwargs)

    def _get_raw_predict(self, data, batch_size=64, verbose=1):
        enhanced_data = self._get_enhanced_data(data)
        x = self._get_model_test_input(enhanced_data)
        pred_data = self.model.predict(x, batch_size=batch_size, verbose=verbose)
        return pred_data

    def predict_batch(self, data, batch_size=64, verbose=1, show_detail=False, **kwargs):
        super().predict_batch(data, **kwargs)
        enhanced_data = self._get_enhanced_data(data)
        rs_data = [[]] * len(data)
        if enhanced_data:
            x = self._get_model_test_input(enhanced_data)
            pred_data = self.model.predict(x, batch_size=batch_size, verbose=verbose)
            rs_data = self._get_predict_data_from_model_output(data, enhanced_data, pred_data, show_detail=show_detail,
                                                               **kwargs)
        return rs_data

    @classmethod
    def _convert_tf_serving_result(cls, tf_response):
        pred_data = tf_response.json()['outputs']

        def convert_data(data):
            if isinstance(data, list):
                return np.array(data)
            if isinstance(data, dict):
                return {k: convert_data(v) for k, v in data.items()}
            return data

        pred_data = convert_data(pred_data)
        if isinstance(pred_data, dict):
            pred_data = tuple([pred_data[k] for k in cls.tf_serving_output_keys])
        # print(pred_data)
        return pred_data

    def predict_batch_tf_serving(self, data, tf_server_host, batch_size=16, action="predict", timeout=60, max_retry=3,
                                 show_detail=False, **kwargs):
        rs_list = []
        self.logger.info("predicting with tf server:{}".format(tf_server_host))
        for idx, batch in enumerate(cut_list(data, batch_size)):
            self.logger.info("predicting batch:{}".format(idx))
            enhanced_data = self._get_enhanced_data(batch)
            if enhanced_data:
                tf_request = self._get_tf_serving_request(enhanced_data)
                tf_response = call_tf_service(tf_request, tf_server_host, action, timeout, max_retry)
                pred_data = self._convert_tf_serving_result(tf_response)
                batch_rs = self._get_predict_data_from_model_output(data, enhanced_data, pred_data,
                                                                    show_detail=show_detail,
                                                                    **kwargs)
            else:
                batch_rs = [[]] * len(batch)
            rs_list.extend(batch_rs)
        return rs_list

    def _get_tf_serving_request(self, train_data):
        input_list = self._get_model_test_input(train_data)
        inputs_dict = {k: v.tolist() for k, v in zip(self.tf_serving_input_keys, input_list)}
        rs_dict = dict(inputs=inputs_dict, signature_name=TF_DEFAULT_SIGNATURE_NAME)
        return rs_dict

    @abstractmethod
    def _get_predict_data_from_model_output(self, origin_data, enhanced_data, pred_data, show_detail=False, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _get_enhanced_data(self, data):
        raise NotImplementedError

    @abstractmethod
    def _get_model_train_input(self, train_data, **kwargs):
        raise NotImplementedError

    def _get_model_test_input(self, test_data, **kwargs):
        x, y = self._get_model_train_input(test_data)
        return x

    def _get_cache_data(self, data):
        return [{k: e[k] for k in self.cache_keys} for e in data]


class RuleEstimator(BaseEstimator):
    @abstractmethod
    def predict_batch(self, batch, **kwargs):
        raise NotImplementedError


class EnsembleEstimator(BaseEstimator):
    @abstractmethod
    def predict_batch(self, batch, **kwargs):
        raise NotImplementedError


class PipelineEstimator(BaseEstimator):
    @abstractmethod
    def predict_batch(self, batch, **kwargs):
        raise NotImplementedError
