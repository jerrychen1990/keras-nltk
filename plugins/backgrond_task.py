# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     backgrond
   Description :
   Author :       chenhao
   date：          2020-01-09
-------------------------------------------------
   Change Activity:
                   2020-01-09:
-------------------------------------------------
"""
import os
from plugin_sdk import ConfigParser
import time

if __name__ == '__main__':
    parser = ConfigParser()
    job_configs = parser.parse()
    sleep_time = job_configs['args']['sleep_time']
    sleep_time = int(sleep_time)
    print("sleep for {} seconds".format(sleep_time))
    time.sleep(sleep_time)
    print("job ends")
