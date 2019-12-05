# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     ner_task
   Description :
   Author :       chenhao
   date：          2019-11-28
-------------------------------------------------
   Change Activity:
                   2019-11-28:
-------------------------------------------------
"""

from eigen_nltk.experiment import NerExperiment
from eigen_nltk.utils import read_config
from plugin_sdk import ConfigParser

if __name__ == '__main__':
    parser = ConfigParser()
    job_configs = parser.parse()
    model_config_path = job_configs['args']['config_path']
    params = read_config(model_config_path)

    gpu_num = job_configs['args'].get("executor_gpu_num", 1)
    params = read_config(model_config_path)
    params['compile_args']['gpu_num'] = gpu_num
    experiment = NerExperiment(params)
    experiment.do_experiment()
