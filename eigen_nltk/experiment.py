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
from keras.callbacks import TensorBoard, EarlyStopping
from abc import abstractmethod
from eigen_nltk.callback import ModelSaver
from eigen_nltk.core import Context
from eigen_nltk.utils import print_info, jdumps, read_json_data, get_now_str, jdump, create_dir, get_path_list, \
    jdump_lines, sample_data_by_num
from eigen_nltk.entity_cliassify import EntityClsEstimator, EntityClsContext
from eigen_nltk.language_model import LMContext, TransformerLM
from eigen_nltk.ner import NerExtractor, NerContext
from eigen_nltk.nre import NreExtractor, NreContext
from eigen_nltk.classify import ClassifyEstimator, ClassifyContext
from eigen_nltk.pretrain import BertPreTrainer
from eigen_nltk.eval import eval_ner, add_ner_pred, eval_classify, eval_entity_classify, add_classify_pred, eval_nre, \
    add_nre_pred, add_entity_classify_pred, eval_language_model, add_language_model_pred


class BaseExperiment:
    def __init__(self, params, fit_generator=True):
        print_info("job param:")
        print(jdumps(params))

        self.dev_data_path_list = params['data']['dev_data_path'].split(",")
        self.test_data_path_list = params['data']['test_data_path'].split(",")
        self.submit_data_path_list = params['data'].get('submit_data_path', "")
        self.train_path_pattern = params['data']['train_path_pattern']

        self.vocab_path = params['schema']['vocab_path']

        self.model_args = params['model_args']
        self.train_args = params['train_args']
        self.compile_args = params['compile_args']
        self.callback_args = params['callback_args']

        self.ckp_path = params['model']['ckp_path']
        self.model_name = params['model']['model_name']

        self.log_level = params['common'].get("log_level", "INFO")
        self.is_train = params['common'].get("is_train", True)
        self.is_eval = params['common'].get("is_eval", True)
        self.is_output = params['common'].get("is_output", True)
        self.is_saving = params['common'].get("is_saving", True)

        self.model_path = "model/{}".format(self.model_name)

        self.estimator = None
        self.pred_train = None
        self.pred_dev = None
        self.pred_test = None
        self.true_train = None
        self.true_dev = None
        self.true_test = None

        self.fit_generator = fit_generator

        self.max_len = params['model'].get("max_len", None)

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
        print_info("estimator initialize phrase start")
        if self.ckp_path:
            print_info("initialize estimator from checkpoint")
            self.estimator = self._load_estimator_func()
        else:
            print_info("initialize estimator from model args")
            self.estimator = self._create_estimator_func()
        print_info("estimator initialize phrase end")

    def _read_data(self):
        print_info("reading data phrase start")
        self.train_data_path_list = get_path_list(self.train_path_pattern)
        print_info("reading train data from :{}".format(self.train_data_path_list))
        self.train_data = read_json_data(self.train_data_path_list)
        print_info("got {} train data".format(len(self.train_data)))

        print_info("reading dev data from :{}".format(self.dev_data_path_list))
        self.dev_data = read_json_data(self.dev_data_path_list)
        print_info("got {} dev data".format(len(self.dev_data)))

        print_info("reading test data from :{}".format(self.test_data_path_list))
        self.test_data = read_json_data(self.test_data_path_list)
        print_info("got {} test data".format(len(self.test_data)))
        print_info("reading data phrase end")

    def preprocess_data(self):
        print_info("preprocess train data")
        enhanced_train_data = self.estimator._get_enhanced_data(self.train_data)
        cache_train_data = self.estimator._get_cache_data(enhanced_train_data)
        print_info("got {} cache train data".format(len(cache_train_data)))
        jdump_lines(cache_train_data, "cache/{0}-train.txt".format(self.model_name))
        print_info("preprocess dev data")
        enhanced_dev_data = self.estimator._get_enhanced_data(self.dev_data)
        cache_dev_data = self.estimator._get_cache_data(enhanced_dev_data)
        print_info("got {} cache dev data".format(len(cache_dev_data)))
        jdump_lines(cache_train_data, "cache/{0}-dev.txt".format(self.model_name))

    def train_model(self):
        print_info("training phrase start")
        if not self.estimator:
            print("please crete estimator before training!")
            return
        create_dir("./tensorboard/")
        tensorboard_callback = TensorBoard(log_dir='./tensorboard/{0}-{1}'.format(self.model_name, get_now_str()))
        early_stop = EarlyStopping(**self.callback_args, restore_best_weights=True)
        callbacks = [tensorboard_callback, early_stop]

        save_epoch_interval = self.train_args.get("save_epoch_ckpt", -1)

        if save_epoch_interval > 0:
            model_saver = ModelSaver(self.estimator, save_epoch_interval=save_epoch_interval, overwrite=False)
            callbacks.append(model_saver)

        self.train_args.update(callbacks=callbacks)
        if self.fit_generator:
            self.estimator.train_model_generator(self.train_data, self.dev_data, self.train_args,
                                                 self.compile_args)
        else:
            self.estimator.train_model(self.train_data, self.dev_data, self.train_args, self.compile_args)
        print_info("training phrase finish")

    def do_predict(self, show_detail=False, verbose=1, max_predict_num=10000):
        print_info("predicting phrase start")
        print_info("predict result on train set:")
        self.true_train = sample_data_by_num(self.train_data, max_predict_num)
        self.pred_train = self.estimator.predict_batch(self.true_train, show_detail=show_detail, verbose=verbose)
        print_info("predict result on dev set:")
        self.true_dev = sample_data_by_num(self.dev_data, max_predict_num)
        self.pred_dev = self.estimator.predict_batch(self.true_dev, show_detail=show_detail, verbose=verbose)
        # print_info("predict result on test set:")
        # self.true_test = sample_data_by_num(self.test_data, max_predict_num)
        # self.pred_test = self.estimator.predict_batch(self.true_test, show_detail=show_detail, verbose=verbose)
        print_info("predicting phrase end")

    def do_eval(self):
        print_info("eval phrase start")
        create_dir("eval/")

        print_info("eval result on train set:")
        eval_rs = self._eval_func(self.true_train, self.pred_train)
        print(jdumps(eval_rs))
        jdump(eval_rs, "eval/{}_train.json".format(self.model_name))

        print_info("eval result on dev set:")
        eval_rs = self._eval_func(self.true_dev, self.pred_dev)
        print(jdumps(eval_rs))
        jdump(eval_rs, "eval/{}_dev.json".format(self.model_name))

        # print_info("eval result on test set:")
        # eval_rs = self._eval_func(self.true_test, self.pred_test)
        # print(jdumps(eval_rs))
        # jdump(eval_rs, "eval/{}_test.json".format(self.model_name))
        # print_info("eval phrase end")

    def do_output(self):
        print_info("output phrase start")
        create_dir("output/")

        print_info("output result on train set:")
        output_data = self._output_func(self.true_train, self.pred_train)
        jdump(output_data, 'output/{}_train.json'.format(self.model_name))

        print_info("output result on dev set:")
        output_data = self._output_func(self.true_dev, self.pred_dev)
        jdump(output_data, 'output/{}_dev.json'.format(self.model_name))

        # print_info("output result on test set:")
        # output_data = self._output_func(self.true_test, self.pred_test)
        # jdump(output_data, 'output/{}_test.json'.format(self.model_name))

        print_info("output phrase start")

    def do_submit(self):
        print_info("submit phrase start")
        submit_data = read_json_data(self.submit_data_path_list)
        submit_pred = self.estimator.predict_batch(submit_data, show_detail=False, verbose=1)
        submit_output = self._output_func(submit_data, submit_pred)
        jdump(submit_output, "output/{}_submit.json".format(self.model_name))
        print_info("submit phrase end")

    def do_experiment(self):
        print_info("job start")

        self.initialize_estimator()
        self._read_data()
        # self.preprocess_data()
        if self.is_train:
            self.train_model()
        if self.is_saving:
            self.estimator.save_estimator(self.model_path)
        if self.is_eval or self.is_output:
            self.do_predict()
            if self.is_eval:
                self.do_eval()
            if self.is_output:
                self.do_output()
        if self.submit_data_path_list:
            self.do_submit()
        print_info("job finish")


class NerExperiment(BaseExperiment):

    def __init__(self, params, fit_generator=True):
        super().__init__(params, fit_generator)
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

    def _load_estimator_func(self):
        return ClassifyEstimator.load_estimator(self.ckp_path)

    def _create_estimator_func(self):
        classify_context = ClassifyContext(self.vocab_path, self.label_dict_path)
        extractor = ClassifyEstimator(self.model_name, classify_context, self.max_len, logger_level=self.log_level)
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
