# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     callback
   Description :
   Author :       chenhao
   date：          2019-09-19
-------------------------------------------------
   Change Activity:
                   2019-09-19:
-------------------------------------------------
"""
from keras.callbacks import Callback
from eigen_nltk.utils import jdump, jdumps, get_logger


class MetricEvaluator(Callback):
    def __init__(self, extractor, dev_data, eval_func, eval_args={}, key_metric_path=['micro_avg', 'f1'],
                 is_store=True, model_path="model",eval_path="eval"):
        self.eval_list = []
        self.best_metric = 0.
        self.best_eval_rs = None
        self.key_metric_path = key_metric_path
        self.key_metric = self.key_metric_path[-1]
        self.extractor = extractor
        self.dev_data = dev_data
        self.eval_func = eval_func
        self.eval_args = eval_args
        self.eval_path = eval_path
        self.model_path = model_path
        self.model_name = self.extractor.model_name
        self.is_store = is_store
        self.logger = get_logger(self.model_name)

    def on_epoch_end(self, epoch, show=True, logs=None):
        self.logger.info("current epoch:{}".format(epoch))
        cur_pred = self.extractor.predict_batch(self.dev_data)
        cur_eval = self.eval_func(self.dev_data, cur_pred, **self.eval_args)
        self.logger.info("current eval rs:")
        self.logger.info(jdumps(cur_eval))

        cur_metric = cur_eval
        for p in self.key_metric_path:
            cur_metric = cur_metric[p]

        self.logger.info("current {0}:{1}|best {0}:{2}".format(self.key_metric, cur_metric, self.best_metric))
        self.eval_list.append(cur_eval)
        if cur_metric >= self.best_metric:
            self.logger.info("get best {0}:{1}".format(self.key_metric, cur_metric))
            self.best_metric = cur_metric
            if self.is_store:
                best_model_path = "{0}/{1}".format(self.model_path, self.model_name)
                self.extractor.save_estimator(best_model_path)
                jdump(cur_eval, open("{0}/{1}_dev.json".format(self.eval_path, self.model_name), 'w'))
