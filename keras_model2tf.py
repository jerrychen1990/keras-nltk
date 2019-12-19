# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     keras_model2tf
   Description :
   Author :       chenhao
   date：          2019-12-17
-------------------------------------------------
   Change Activity:
                   2019-12-17:
-------------------------------------------------
"""

import sys

# ============================================================================== #
from keras.models import load_model

from eigen_nltk.model_utils import export_keras_as_tf_file, get_full_customer_objects

if __name__ == '__main__':
    keras_path, export_path = sys.argv[1:]
    keras_model = load_model(keras_path, custom_objects=get_full_customer_objects())
    export_keras_as_tf_file(keras_model, export_path)
