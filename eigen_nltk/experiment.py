# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     experiment
   Description :
   Author :       chenhao
   date：          2019-10-12
-------------------------------------------------
   Change Activity:
                   2019-10-12:
-------------------------------------------------
"""
from abc import abstractmethod

from keras.callbacks import TensorBoard, EarlyStopping
import os

from eigen_nltk.callback import ModelSaver
from eigen_nltk.classify import ClassifyEstimator, ClassifyContext
from eigen_nltk.core import Context
from eigen_nltk.entity_cliassify import EntityClsEstimator, EntityClsContext
from eigen_nltk.eval import eval_ner, add_ner_pred, eval_classify, eval_entity_classify, add_classify_pred, eval_nre, \
    add_nre_pred, add_entity_classify_pred, eval_language_model, add_language_model_pred, eval_ner_classify, \
    add_ner_classify_pred
from eigen_nltk.language_model import LMContext, TransformerLM
from eigen_nltk.ner import NerExtractor, NerContext
from eigen_nltk.ner_classify import NerClassifyContext, NerClassifyExtractor
from eigen_nltk.nre import NreExtractor, NreContext
from eigen_nltk.pretrain import BertPreTrainer
from eigen_nltk.utils import print_info, jdumps, read_json_data, get_now_str, jdump, create_dir, sample_data_by_num, \
    get_logger, upload_file2oss

logger = get_logger(__name__)


def star_print(info):
    return print_info(info, logger=logger)


class DataManager:
    def __init__(self, train_path_list, dev_path_list, test_path_list, enhance_func):
        self.train_path_list = train_path_list
        self.dev_path_list = dev_path_list
        self.test_path_list = test_path_list
        self.enhance_func = enhance_func
        self.data_path_dict = dict(train=self.train_path_list, dev=self.dev_path_list,
                                   test=self.test_path_list)
        self.data_cache = dict()

    @classmethod
    def _read_data(cls, data_path_list):
        logger.info("reading data from :{}".format(data_path_list))
        data = read_json_data(data_path_list)
        logger.info("get {} items".format(len(data)))
        return data

    def get_data(self, data_tag, is_enhance):
        key = (data_tag, is_enhance)
        if key in self.data_cache.keys():
            return self.data_cache[key]
        if is_enhance:
            raw_data = self.get_data(data_tag, False)
            rs_data = self.enhance_func(raw_data)
        else:
            data_path_list = self.data_path_dict[data_tag]
            rs_data = self._read_data(data_path_list)
        self.data_cache[key] = rs_data
        return rs_data


class BaseExperiment:
    def __init__(self, params):
        star_print("job param:")
        logger.info(jdumps(params))

        self.dev_data_path_list = params['data']['dev_data_path'].split(",")
        self.test_data_path_list = params['data']['test_data_path'].split(",")
        self.train_data_path_list = params['data']['train_data_path'].split(",")

        self.submit_data_path_list = params['data'].get('submit_data_path', "")

        self.vocab_path = params['schema']['vocab_path']
        self.model_args = params['model_args']
        self.train_args = params['train_args']
        self.compile_args = params['compile_args']
        self.callback_args = params['callback_args']

        self.ckp_path = params['model']['ckp_path']
        self.model_name = params['model']['model_name']
        self.max_len = params['model'].get("max_len", None)

        self.log_level = params['common'].get("log_level", "INFO")
        self.is_train = params['common'].get("is_train", True)
        self.eval_phase_list = params['common']['eval_phase_list'].split(",")
        self.is_eval = len(self.eval_phase_list) > 0
        self.output_phase_list = params['common']['output_phase_list'].split(",")
        self.is_output = len(self.output_phase_list) > 0
        self.is_save = params['common'].get("is_save", True)
        self.is_save_tf = params['common'].get("is_save_tf", False)
        self.oss_dir = params['common'].get("oss_dir", None)
        self.base_dir = params['common'].get("base_dir", ".")
        self.project_dir = params['common'].get("project_dir", ".")

        self.model_dir = os.path.join(self.base_dir, "model", self.project_dir)
        self.tensorboard_dir = os.path.join(self.base_dir, "tensorboard", self.project_dir)
        self.output_dir = os.path.join(self.base_dir, "output", self.project_dir)
        self.eval_dir = os.path.join(self.base_dir, "eval", self.project_dir)
        self.model_path = os.path.join(self.model_dir, self.model_name)

        self.estimator = None
        self.pred_data_dict = dict()
        self.data_manager = None

    @abstractmethod
    def _eval_func(self, data, pred, **kwargs):
        pass

    @abstractmethod
    def _output_func(self, data, pred, **kwargs):
        pass

    @abstractmethod
    def _load_estimator_func(self):
        pass

    @abstractmethod
    def _create_estimator_func(self):
        pass

    def initialize_estimator(self):
        star_print("estimator initialize phrase start")
        if self.ckp_path:
            star_print("initialize estimator from checkpoint")
            self.estimator = self._load_estimator_func()
        else:
            star_print("initialize estimator from model args")
            self.estimator = self._create_estimator_func()
        self.data_manager = DataManager(self.train_data_path_list, self.dev_data_path_list, self.test_data_path_list,
                                        self.estimator._get_enhanced_data)
        star_print("estimator initialize phrase end")

    def train_model(self):
        star_print("training phrase start")
        if not self.estimator:
            logger.error("please crete estimator before training!")
            return
        create_dir(self.tensorboard_dir)
        tensorboard_callback = TensorBoard(
            log_dir='{0}/{1}-{2}'.format(self.tensorboard_dir, self.model_name, get_now_str()))
        early_stop = EarlyStopping(**self.callback_args, restore_best_weights=True)
        callbacks = [tensorboard_callback, early_stop]

        save_epoch_interval = self.train_args.get("save_epoch_ckpt", -1)

        if save_epoch_interval > 0:
            model_saver = ModelSaver(self.estimator, self.model_dir, save_epoch_interval=save_epoch_interval,
                                     overwrite=False)
            callbacks.append(model_saver)

        self.train_args.update(callbacks=callbacks)
        logger.info("getting train data for training phrase")
        train_data = self.data_manager.get_data("train", True)
        logger.info("getting dev data for training phrase")
        dev_data = self.data_manager.get_data("dev", True)

        self.estimator.train_model_generator(train_data, dev_data, self.train_args,
                                             self.compile_args, is_raw=False)

    def test_model(self, show_detail=False, verbose=1, max_predict_num=10000):
        star_print("testing phrase start")
        create_dir(self.eval_dir)
        create_dir(self.output_dir)

        for tag in ['train', 'dev', 'test']:
            if tag not in self.eval_phase_list and tag not in self.output_phase_list:
                continue
            logger.info("predict result on {} set:".format(tag))
            raw_data = self.data_manager.get_data(tag, False)
            raw_data = sample_data_by_num(raw_data, max_predict_num)
            pred_data = self.estimator.predict_batch(raw_data, show_detail=show_detail, verbose=verbose)
            if tag in self.eval_phase_list:
                logger.info("evaluating {} set".format(tag))
                eval_rs = self._eval_func(raw_data, pred_data)
                logger.info(jdumps(eval_rs))
                path = "{0}/{1}_{2}.json".format(self.eval_dir, self.model_name, tag)
                logger.info("writing eval result to :{}".format(path))
                jdump(eval_rs, path)
            if tag in self.output_phase_list:
                logger.info("output detail of {} set:".format(tag))
                output_data = self._output_func(raw_data, pred_data)
                path = '{0}/{1}_{2}.json'.format(self.output_dir, self.model_name, tag)
                logger.info("writing output result to :{}".format(path))
                jdump(output_data, path)

        star_print("testing phrase end")

    def do_submit(self):
        star_print("submit phrase start")
        submit_data = read_json_data(self.submit_data_path_list)
        submit_pred = self.estimator.predict_batch(submit_data, show_detail=False, verbose=1)
        submit_output = self._output_func(submit_data, submit_pred)
        jdump(submit_output, "output/{}_submit.json".format(self.model_name))
        star_print("submit phrase end")

    def do_experiment(self):
        star_print("job start")
        self.initialize_estimator()
        if self.is_train:
            self.train_model()
            if self.is_save:
                self.estimator.save_estimator(self.model_path)
            if self.is_save_tf:
                compress_path = self.estimator.save_tf_serving_model(path=self.model_path)
                if self.oss_dir:
                    upload_file2oss(compress_path, self.oss_dir)

        self.test_model()
        if self.submit_data_path_list:
            self.do_submit()
        star_print("job finish")


class NerExperiment(BaseExperiment):

    def __init__(self, params):
        super().__init__(params)
        self.ner_dict_path = params['schema']['ner_dict_path']
        self.annotation_type = params['schema']['annotation_type']

    def _load_estimator_func(self):
        return NerExtractor.load_estimator(self.ckp_path)

    def _create_estimator_func(self):
        ner_context = NerContext(self.vocab_path, self.ner_dict_path, self.annotation_type)
        extractor = NerExtractor(self.model_name, ner_context, self.max_len, logger_level=self.log_level)
        extractor.create_model(self.model_args)
        return extractor

    def _eval_func(self, data, pred, **kwargs):
        return eval_ner(data, pred, contain_span=False)

    def _output_func(self, data, pred, **kwargs):
        return add_ner_pred(data, pred, contain_span=False)


class ClassifyExperiment(BaseExperiment):

    def __init__(self, params):
        super().__init__(params)
        self.label_dict_path = params['schema']['label_dict_path']
        self.multi_label = params['model']['multi_label']

    def _load_estimator_func(self):
        return ClassifyEstimator.load_estimator(self.ckp_path)

    def _create_estimator_func(self):
        classify_context = ClassifyContext(self.vocab_path, self.label_dict_path)
        extractor = ClassifyEstimator(self.model_name, classify_context, self.max_len, self.multi_label,
                                      logger_level=self.log_level)
        extractor.create_model(self.model_args)
        return extractor

    def _eval_func(self, data, pred, **kwargs):
        return eval_classify(data, pred)

    def _output_func(self, data, pred, **kwargs):
        return add_classify_pred(data, pred)


class NreExperiment(BaseExperiment):

    def __init__(self, params):
        super().__init__(params)
        self.rel_dict_path = params['schema']['rel_dict_path']

    def _load_estimator_func(self):
        return NreExtractor.load_estimator(self.ckp_path)

    def _create_estimator_func(self):
        context = NreContext(self.vocab_path, self.rel_dict_path)
        extractor = NreExtractor(self.model_name, context, self.max_len, logger_level=self.log_level)
        extractor.create_model(self.model_args)
        return extractor

    def _eval_func(self, data, pred, **kwargs):
        return eval_nre(data, pred)

    def _output_func(self, data, pred, **kwargs):
        return add_nre_pred(data, pred)


class EntityClsExperiment(BaseExperiment):

    def __init__(self, params):
        super().__init__(params, fit_generator=True)
        self.label_dict_path = params['schema']['label_dict_path']

    def _load_estimator_func(self):
        return EntityClsEstimator.load_estimator(self.ckp_path)

    def _create_estimator_func(self):
        context = EntityClsContext(self.vocab_path, self.label_dict_path)
        extractor = EntityClsEstimator(self.model_name, context, self.max_len, logger_level=self.log_level)
        extractor.create_model(self.model_args)
        return extractor

    def _eval_func(self, data, pred, **kwargs):
        return eval_entity_classify(data, pred)

    def _output_func(self, data, pred, **kwargs):
        return add_entity_classify_pred(data, pred)


class PreTrainExperiment(BaseExperiment):

    def __init__(self, params):
        super().__init__(params)

    def _load_estimator_func(self):
        return BertPreTrainer.load_estimator(self.ckp_path)

    def _create_estimator_func(self):
        context = Context(self.vocab_path)
        pretrainer = BertPreTrainer(self.model_name, context, self.max_len, logger_level=self.log_level)
        pretrainer.create_model(self.model_args)
        return pretrainer

    def _eval_func(self, data, pred, **kwargs):
        pass

    def _output_func(self, data, pred, **kwargs):
        pass


class LanguageModelExperiment(BaseExperiment):

    def __init__(self, params):
        super().__init__(params)

    def _load_estimator_func(self):
        return TransformerLM.load_estimator(self.ckp_path)

    def _create_estimator_func(self):
        context = LMContext(self.vocab_path)
        lm = TransformerLM(self.model_name, context, logger_level=self.log_level)
        lm.create_model(self.model_args)
        return lm

    def _eval_func(self, data, pred, **kwargs):
        return eval_language_model(data, pred)

    def _output_func(self, data, pred, **kwargs):
        return add_language_model_pred(data, pred)


class NerClassifyExperiment(BaseExperiment):

    def __init__(self, params):
        super().__init__(params)
        self.ner_dict_path = params['schema']['ner_dict_path']
        self.annotation_type = params['schema']['annotation_type']
        self.label_dict_path = params['schema']['label_dict_path']

    def _load_estimator_func(self):
        return NerClassifyExtractor.load_estimator(self.ckp_path)

    def _create_estimator_func(self):
        context = NerClassifyContext(self.vocab_path, self.ner_dict_path, self.annotation_type,
                                     self.label_dict_path)
        extractor = NerClassifyExtractor(self.model_name, context, self.max_len, logger_level=self.log_level)
        extractor.create_model(self.model_args)
        return extractor

    def _eval_func(self, data, pred, **kwargs):
        return eval_ner_classify(data, pred)

    def _output_func(self, data, pred, **kwargs):
        return add_ner_classify_pred(data, pred)
