# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     decorator
   Description :
   Author :       chenhao
   date：          2019-09-28
-------------------------------------------------
   Change Activity:
                   2019-09-28:
-------------------------------------------------
"""
import os
import inspect


def ensure_file_path(func):
    def wrapper(*args, **kwargs):
        path = kwargs['path']
        dir_path = os.path.dirname(path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        return func(*args, **kwargs)

    return wrapper


def ensure_dir_path(func):
    def wrapper(*args, **kwargs):
        dir_path = kwargs['path']
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        return func(*args, **kwargs)

    return wrapper


def safe_args(func):
    valid_args = inspect.getfullargspec(func).args

    def wrapper(*args, **kwargs):
        kwargs = dict([(k, v) for k, v in kwargs.items() if k in valid_args])
        return func(*args, **kwargs)

    return wrapper


def adapt_single(func):
    def wrapper(instance, data, **kwargs):
        single_tag = not isinstance(data, list)
        if single_tag:
            data = [data]
        rs = func(instance, data, **kwargs)
        if single_tag:
            rs = rs[0] if rs else None
        return rs

    return wrapper
