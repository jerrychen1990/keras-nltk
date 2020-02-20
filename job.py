# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     tmp
   Description :
   Author :       chenhao
   date：          2019-11-08
-------------------------------------------------
   Change Activity:
                   2019-11-08:
-------------------------------------------------
"""
from eigen_nltk.utils import *
from threading import Thread
import threading
from tqdm import tqdm
import thulac
import re


def parse_magi_item(item, thu):
    key, v = item
    content, subject = key
    ts = "[ES1]"
    te = "[EE1]"
    sub_rep = ts + subject + te
    content = content.replace(subject, sub_rep)
    content = re.sub("\s+", " ", content)
    #     content
    pos_list = thu.cut(content)
    #     pos_list
    pos = flat([[t] * len(s) for s, t in pos_list])
    if len(pos) != len(content):
        #         print(pos)
        #         print(content)
        return None
    entity_list = [[(e, "OBJECT", find_all_char(content, e)) for e in s] for s in v]

    rs_item = dict(content=content, subject=subject, pos=pos, entity_list=entity_list)
    return rs_item


class MyThread(Thread):
    def __init__(self, idx, batch_size, thread_num, data):
        threading.Thread.__init__(self)
        self.idx = idx
        self.batch_size = batch_size
        self.count = 0
        self.thread_num = thread_num
        self.data = data

    def run(self):
        thu = thulac.thulac()  # 默认模式
        while True:
            start = self.count * self.thread_num * self.batch_size + self.idx * self.batch_size
            end = start + self.batch_size
            batch = self.data[start: end]
            if batch:
                print("thread{0} fetch data from {1} to {2}".format(self.idx, start, end))
                rs_list = []
                for item in batch:
                    rs_item = parse_magi_item(item, thu)
                    if rs_item:
                        rs_list.append(rs_item)
                path = "/data/jh/notebooks/chenhao/eigen-nlp-toolkit/data/openie/magi_general/magi_{}.json".format(start)
                print("dump to {}".format(path))
                jdump(rs_list, path)
                self.count += 1
            else:
                print("thread{}'s job done".format(self.idx))
                break


print("reading data")
data_path = "/data/jh/notebooks/chenhao/eigen-nlp-toolkit/todo_magi.json"
# data_path = "sample_magi.json"
data = jload(data_path)
print(len(data))

num = 40
thread_list = []
for idx in range(num):
    thread = MyThread(idx, 1000, num, data)
    thread_list.append(thread)
    thread.start()

for thread in thread_list:
    thread.join()

print("job finish")
