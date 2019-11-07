# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     train_ner
   Description :
   Author :       chenhao
   date：          2019-09-19
-------------------------------------------------
   Change Activity:
                   2019-09-19:
-------------------------------------------------
"""

from eigen_nltk.experiment import NerExperiment
from eigen_nltk.utils import read_config

if __name__ == '__main__':
    config_path = "config/ner.ini"
    params = read_config(config_path)
    experiment = NerExperiment(params)
    experiment.do_experiment()

