# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     open_ie_service
   Description :
   Author :       chenhao
   date：          2019-12-03
-------------------------------------------------
   Change Activity:
                   2019-12-03:
-------------------------------------------------
"""
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
from prometheus_client import push_to_gateway, CollectorRegistry, Counter
from simplex_base_model import SimplexBaseModel

from eigen_nltk.ner import NerExtractor
from eigen_nltk.ner_classify import NerClassifyExtractor
from eigen_nltk.open_ie import SchemaFreeSPOExtractorPSO
from eigen_nltk.utils import get_logger


class OpenIEService(SimplexBaseModel):
    @staticmethod
    def __init_gpu(fraction):
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = fraction
        session = tf.Session(config=config)
        KTF.set_session(session)

    def __init__(self, *args, **kwargs):
        super(OpenIEService, self).__init__(*args, **kwargs)
        OpenIEService.__init_gpu(0.5)

        predicate_extractor_path = kwargs['predicate_extractor_path']
        subject_extractor_path = kwargs['subject_extractor_path']
        object_extractor_path = kwargs['object_extractor_path']

        self.model_name = kwargs["model_name"]

        self.logger = get_logger(self.model_name, "INFO")

        self.predicate_extractor_path = self._download_ckp_path(predicate_extractor_path)
        self.subject_extractor_path = self._download_ckp_path(subject_extractor_path)
        self.object_extractor_path = self._download_ckp_path(object_extractor_path)

        self.predicate_extractor = NerClassifyExtractor.load_estimator(self.predicate_extractor_path)
        self.subject_extractor = NerExtractor.load_estimator(self.subject_extractor_path)
        self.object_extractor = NerExtractor.load_estimator(self.object_extractor_path)
        self.open_ie_extractor = SchemaFreeSPOExtractorPSO(self.model_name, self.predicate_extractor,
                                                           self.subject_extractor, self.object_extractor)

        self._register_monitor()
        self.graph = tf.get_default_graph()

    def _register_monitor(self):
        self.registry = CollectorRegistry()
        self.person_schema_request_counter = Counter('person_schema_request_num', '模型接受到的请求次数', registry=self.registry,
                                                     labelnames=['request_num'])

    def _download_ckp_path(self, path):
        model_path = path + ".h5"
        extractor_path = path + ".pkl"
        model_path = self.download(model_path)
        self.download(extractor_path)
        rs_path = model_path[:-3]
        return rs_path

    @staticmethod
    def __init_gpu(fraction):
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = fraction
        session = tf.Session(config=config)
        KTF.set_session(session)

    def report_metric(self):
        self.person_schema_request_counter.labels(request_num='request_num').inc(1)
        push_to_gateway('pgw.monitor.aipp.io', job="person_schema_extract", registry=self.registry)

    def predict(self, data, **kwargs):
        with self.graph.as_default():
            self.logger.info("predicting with kwargs:{}".format(kwargs))
            predicate_threshold = kwargs.get("predicate_threshold", .5)
            batch_size = kwargs.get("batch_size", 64)
            data = [dict(id=idx, content=e) for idx, e in enumerate(data)]
            rs = self.open_ie_extractor.predict_batch(data, batch_size=batch_size, verbose=0,
                                                      predicate_threshold=predicate_threshold)

            return rs
