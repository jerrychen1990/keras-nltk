# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     language_model_task
   Description :
   Author :       chenhao
   date：          2019-12-10
-------------------------------------------------
   Change Activity:
                   2019-12-10:
-------------------------------------------------
"""
from plugin_sdk import ConfigParser

from eigen_nltk.experiment import LanguageModelExperiment
from eigen_nltk.utils import read_config

if __name__ == '__main__':
    parser = ConfigParser()
    job_configs = parser.parse()
    model_config_path = job_configs['args']['config_path']
    gpu_num = job_configs['args'].get("executor_gpu_num", 1)

    params = read_config(model_config_path)
    params['compile_args']['gpu_num'] = int(gpu_num)
    experiment = LanguageModelExperiment(params)
    experiment.do_experiment()
