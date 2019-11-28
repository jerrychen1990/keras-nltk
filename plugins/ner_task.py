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

import sys

from eigen_nltk.experiment import NerExperiment
from eigen_nltk.utils import read_config

if __name__ == '__main__':
    config_path = sys.argv[1]
    params = read_config(config_path)
    experiment = NerExperiment(params)
    experiment.do_experiment()
