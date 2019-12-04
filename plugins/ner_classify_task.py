# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     ner_classify_task
   Description :
   Author :       chenhao
   date：          2019-12-04
-------------------------------------------------
   Change Activity:
                   2019-12-04:
-------------------------------------------------
"""
from eigen_nltk.experiment import NerClassifyExperiment
from eigen_nltk.utils import read_config
from plugin_sdk import ConfigParser

if __name__ == '__main__':
    parser = ConfigParser()
    job_configs = parser.parse()
    model_config_path = job_configs['args']['config_path']

    params = read_config(model_config_path)
    experiment = NerClassifyExperiment(params)
    experiment.do_experiment()
