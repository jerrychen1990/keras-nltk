# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     utils
   Description :
   Author :       chenhao
   date：          2019-09-19
-------------------------------------------------
   Change Activity:
                   2019-09-19:
-------------------------------------------------
"""
import logging
import codecs
import json
import os
import re
import pickle
import numpy as np
import random
import copy
import tarfile
from datetime import datetime
from configparser import ConfigParser
from collections import defaultdict, OrderedDict
from fast_requests import FastSession
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]:%(message)s", datefmt='%Y-%m-%d %H:%M:%S')


# get logger which output ot standard output
def get_logger(logger_name, level="INFO"):
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    return logger


def call_tf_service(data, tf_server_host, action, timeout, max_retry):
    host = '{host}:{action}'.format(host=tf_server_host, action=action)
    resp = FastSession().post(host, json=data, timeout=timeout, max_retry=max_retry)
    resp.raise_for_status()
    return resp


# read id2enum and enum2id dict from json file
def read_id_mapping(mapping_path):
    name2id = jload(mapping_path)
    id2name = {int(v): k for k, v in name2id.items()}
    assert len(id2name) == len(name2id)
    return id2name, name2id


# padding sequence to max_len with pad_value
def padding_seq(seq, max_len, pad_value=0):
    return seq[:min(len(seq), max_len)] + [pad_value] * (max_len - len(seq))


# convert list of list to a flat list
def flat(seq):
    return [e for item in seq for e in item]


# execute a shell command
def execute_cmd(cmd):
    print("execute cmd:{}".format(cmd))
    return os.system(cmd)


def upload_file2oss(file_path, oss_dir_path):
    file_name = os.path.basename(file_path)
    cmd = "osscmd upload {file_path} {oss_dir_path}/{file_name}".format(file_path=file_path, oss_dir_path=oss_dir_path,
                                                                        file_name=file_name)
    return execute_cmd(cmd)


def split_text_by_commas(seq, pattern=r"\.|,|\?|\;|!|。|，|？|；|！"):
    commas = re.findall(pattern, seq)
    texts = re.split(pattern, seq)
    assert len(commas) + 1 == len(texts)
    rs = [a + b for a, b in zip(texts, commas)]
    if texts[-1]:
        rs.append(texts[-1])
    return rs


def split_token_by_commas(token_list, pattern=".,?;!。，？；！"):
    pattern_list = list(pattern)
    rs_list = []
    count = 0
    for idx, token in enumerate(token_list):
        if token in pattern_list:
            rs_list.append(token_list[count: idx + 1])
            count = idx + 1
    if count != len(token_list):
        rs_list.append(token_list[count: len(token_list)])
    return rs_list


def split_by_len(seq, max_len):
    return [seq[i: i + max_len] for i in range(0, len(seq), max_len)]


# split content into sequences shorter than max_len
def split_content(content, max_len, pattern=r"，|。|；|\.|,|;|\?|!"):
    def merge_piece(piece_list, max_len):
        rs_list = []
        tmp_str = ""
        for piece in piece_list:
            if len(tmp_str) + len(piece) > max_len:
                rs_list.append(tmp_str)
                tmp_str = piece
            else:
                tmp_str += piece
        if tmp_str:
            rs_list.append(tmp_str)
        return rs_list

    piece_list = flat([split_by_len(e, max_len) for e in split_text_by_commas(content, pattern)])
    rs_list = merge_piece(piece_list, max_len)
    return rs_list


# split token_list into sequences shorter than max_len
def split_token(token_list, max_len, pattern=r"，|。|；|\.|,|;|\?|!"):
    nature_split_list = split_token_by_commas(token_list, pattern)

    def merge_token_piece(token_piece_list, max_len):
        rs_list = []
        tmp_list = []
        for piece in token_piece_list:
            if len(piece) > max_len:
                rs_list.append(piece)
                continue
            if len(tmp_list) + len(piece) > max_len:
                rs_list.append(tmp_list)
                tmp_list = piece
            else:
                tmp_list.extend(piece)
        if tmp_list:
            rs_list.append(tmp_list)
        return rs_list

    def split_token_by_len(token_list, max_len):
        rs_list = []
        beg = 0
        while beg < len(token_list):
            end = beg + max_len
            if end >= len(token_list):
                rs_list.append(token_list[beg:end])
            else:
                while token_list[end].startswith("##"):
                    end -= 1
                if end == beg:
                    end = beg + max_len
                rs_list.append(token_list[beg:end])
            beg = end
        return rs_list

    merged_list = merge_token_piece(nature_split_list, max_len)
    rs_list = flat(split_token_by_len(e, max_len) for e in merged_list)
    return rs_list


# split data to sequences shorter than max_len
def split_data2seq_list(data, max_len, key="token"):
    rs_list = []
    for idx, item in enumerate(data):
        if key == "token":
            token_list = split_token(item['token'], max_len)
            offset = 0
            for token in token_list:
                tmp_dict = dict(**item)
                tmp_dict.update(token=token, idx=idx, offset=offset)
                rs_list.append(tmp_dict)
                offset += len(token)
    return rs_list


# add offset to a span
def add_offset(span, offset):
    return tuple([e + offset for e in list(span)])


class PythonObjectEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, bytes):
            return str(obj, encoding="utf8")
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, set):
            return list(obj)

        if isinstance(obj, (list, dict, str, int, set, float, bool, type(None))):
            return json.JSONEncoder.default(self, obj)
        return {'_python_object': pickle.dumps(obj)}


def as_python_object(dct):
    if '_python_object' in dct:
        return pickle.loads(str(dct['_python_object']))
    return dct


# my json dumps
def jdumps(obj):
    return json.dumps(obj, ensure_ascii=False, indent=4, cls=PythonObjectEncoder)


# my json dump
def jdump(obj, fp):
    if isinstance(fp, str):
        fp = codecs.open(fp, 'w', 'utf8')
    return json.dump(obj, fp, ensure_ascii=False, indent=4, cls=PythonObjectEncoder)


def jdump_lines(obj, fp):
    content = "\n".join([json.dumps(e, ensure_ascii=False) for e in obj])
    if isinstance(fp, str):
        fp = codecs.open(fp, 'w', 'utf8')
    with fp as fp:
        fp.write(content)


# my json load
def jload(fp):
    if isinstance(fp, str):
        fp = codecs.open(fp, 'r', 'utf8')
    return json.load(fp, object_hook=as_python_object)


# my json loads
def jloads(s):
    return json.loads(s, object_hook=as_python_object)


# convert cfg data to dict
def cfg2dict(cfg):
    sections = cfg.sections()
    rs = {k: dict(cfg[k]) for k in sections}
    return rs


def read_config(config_path):
    def eval_param(param):
        if isinstance(param, str):
            try:
                rs = eval(param)
                return rs
            except:
                return param
        if isinstance(param, dict):
            return {k: eval_param(v) for k, v in param.items()}

    cfg = ConfigParser()
    cfg.read(config_path)
    cfg_dict = cfg2dict(cfg)
    cfg_dict = eval_param(cfg_dict)
    return cfg_dict


# find all file path which path match pattern path_pattern
def get_path_list(path_pattern):
    path_list = []
    dir_path = os.path.dirname(path_pattern)
    file_pattern = re.compile(os.path.basename(path_pattern))
    for root, dirs, files in os.walk(dir_path):
        for f in files:
            if re.match(file_pattern, f):
                path_list.append(os.path.join(root, f))
    return path_list


# surround info with stars
def star_surround_info(info, fix_length=128):
    star_num = max(fix_length - len(info), 2)
    left_star_num = star_num // 2
    right_star_num = star_num - left_star_num
    rs = "*" * left_star_num + info + "*" * right_star_num
    return rs


# print info with stars surround
def print_info(info, logger=None, fix_length=128):
    star_info = star_surround_info(info, fix_length)
    if logger:
        logger.info(star_info)
    else:
        print(star_info)


def read_json_data(data_path):
    if isinstance(data_path, list):
        rs = []
        for dp in data_path:
            rs += jload(codecs.open(dp, 'r'))
        return rs
    else:
        return jload(codecs.open(data_path, 'r'))


# convert token_span to text_span
def convert_span(span, idx_mapping):
    start, end = span
    if start >= len(idx_mapping) or end >= len(idx_mapping):
        return None
    return idx_mapping[start], idx_mapping[end]


# split data to two parts with fraction
def split_data(data, fraction):
    random.shuffle(data)
    tgt_len = int(len(data) * fraction)
    return data[:tgt_len], data[tgt_len:]


def count_by_label(data, label):
    rs_dict = defaultdict(int)
    for item in data:
        key = item[label]
        rs_dict[key] += 1
    return dict(rs_dict)


# return shuffled copy of data without change origin data
def get_shuffle_data(data):
    rs_list = copy.copy(data)
    random.shuffle(rs_list)
    return rs_list


def sample_data_by_num(data, num):
    if num >= len(data):
        return data
    else:
        rs_list = get_shuffle_data(data)
        return rs_list[:num]


# sample data with fraction
def sample_data(data, fraction=None, label_key=None):
    def sample_by_fraction(data, fraction):
        if not data or fraction > 1:
            return data
        num = int(len(data) * fraction)
        return sample_data_by_num(data, num)

    if label_key:
        # average sample on different label
        label_distribute = count_by_label(data, label_key)
        rs_list = [sample_data([e for e in data if e[label_key] == l], fraction) for l in label_distribute.keys()]
        rs_list = flat(rs_list)
        return rs_list

    else:
        return sample_by_fraction(data, fraction)


# find all the start index and end index of text in seq with char level
def find_all_char(content, to_find, overlap=False, ignore_case=False):
    rs_list = []
    if ignore_case:
        content = content.lower()
        to_find = to_find.lower()
    if not to_find:
        return rs_list
    text_len = len(to_find)

    beg = 0
    while True:
        b = content.find(to_find, beg)
        if b == -1:
            return rs_list
        e = b + text_len
        rs_list.append((b, e))
        beg = b + 1 if overlap else e
    return rs_list


def read_line_data(path):
    rs_list = []
    with codecs.open(path, "r", "utf8") as f:
        for line in f:
            rs_list.append(line.strip())
    return rs_list


REMOVE_PATTERN = re.compile(
    r"[^\u3400-\u9FFF\w\d\n\r\·\~\！\@\#\￥\%\……\&\*\（\）\——\-\+\=\【\】\{\}\、\|\；\‘\’\：\“\”\《\》\？\，\。\、\`\~\!\#\$\%\^\&\*\(\)\_\[\]{\}\\\|\;\'\'\:\"\"\,\.\/\<\>\?]")


# remove useless chars in text
def pure_text(text):
    return re.sub(REMOVE_PATTERN, "", text)


def get_now_str(fmt="%Y-%m-%d-%H:%M:%S"):
    return datetime.strftime(datetime.now(), fmt)


def get_distribution(l, key=lambda x: x):
    group_by_dict = group_by(l, key)
    rs_dict = {k: len(v) for k, v in group_by_dict.items()}
    return rs_dict


def group_by(l, key=lambda x: x):
    rs_dict = defaultdict(list)
    for i in l:
        rs_dict[key(i)].append(i)
    return OrderedDict(sorted(rs_dict.items(), key=lambda x: len(x[1]), reverse=True))


def get_major_element(l):
    return list(get_distribution(l).items())[0][0]


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def list_find(l, ele):
    try:
        return l.index(ele)
    except Exception as e:
        return -1


def split_text_by_sep(text, sep_list):
    idx_dict = {k: find_all_char(text, k) for k in sep_list}
    idx_list = flat([(k, s) for s in span_list] for k, span_list in idx_dict.items())
    idx_list = sorted(idx_list, key=lambda x: x[1][0])
    beg = 0
    rs_list = []
    for k, (s, e) in idx_list:
        rs_list.append(text[beg:s])
        rs_list.append(k)
        beg = e
    rs_list.append(text[beg:])
    return rs_list


def array2zero_one_with_threshold(arr, threshold=0.5):
    rs_arr = np.zeros_like(arr)
    rs_arr[arr >= threshold] = 1
    rs_arr = rs_arr.astype(np.int32)
    return rs_arr


# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     utils
   Description :
   Author :       chenhao
   date：          2019-09-19
-------------------------------------------------
   Change Activity:
                   2019-09-19:
-------------------------------------------------
"""
import logging
import codecs
import json
import os
import re
import pickle
import numpy as np
import random
import copy
import tarfile
from datetime import datetime
from configparser import ConfigParser
from collections import defaultdict, OrderedDict
from fast_requests import FastSession

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]:%(message)s", datefmt='%Y-%m-%d %H:%M:%S')


# get logger which output ot standard output
def get_logger(logger_name, level="INFO"):
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    return logger


def call_tf_service(data, tf_server_host, action="predict", timeout=60, max_retry=3):
    host = '{host}:{action}'.format(host=tf_server_host, action=action)
    resp = FastSession().post(host, json=data, timeout=timeout, max_retry=max_retry)
    resp.raise_for_status()
    return resp


# read id2enum and enum2id dict from json file
def read_id_mapping(mapping_path):
    name2id = jload(mapping_path)
    id2name = {int(v): k for k, v in name2id.items()}
    assert len(id2name) == len(name2id)
    return id2name, name2id


# padding sequence to max_len with pad_value
def padding_seq(seq, max_len, pad_value=0):
    return seq[:min(len(seq), max_len)] + [pad_value] * (max_len - len(seq))


# convert list of list to a flat list
def flat(seq):
    return [e for item in seq for e in item]


def split_text_by_commas(seq, pattern=r"\.|,|\?|\;|!|。|，|？|；|！"):
    commas = re.findall(pattern, seq)
    texts = re.split(pattern, seq)
    assert len(commas) + 1 == len(texts)
    rs = [a + b for a, b in zip(texts, commas)]
    if texts[-1]:
        rs.append(texts[-1])
    return rs


def split_token_by_commas(token_list, pattern=".,?;!。，？；！"):
    pattern_list = list(pattern)
    rs_list = []
    count = 0
    for idx, token in enumerate(token_list):
        if token in pattern_list:
            rs_list.append(token_list[count: idx + 1])
            count = idx + 1
    if count != len(token_list):
        rs_list.append(token_list[count: len(token_list)])
    return rs_list


def split_by_len(seq, max_len):
    return [seq[i: i + max_len] for i in range(0, len(seq), max_len)]


# split content into sequences shorter than max_len
def split_content(content, max_len, pattern=r"，|。|；|\.|,|;|\?|!"):
    def merge_piece(piece_list, max_len):
        rs_list = []
        tmp_str = ""
        for piece in piece_list:
            if len(tmp_str) + len(piece) > max_len:
                rs_list.append(tmp_str)
                tmp_str = piece
            else:
                tmp_str += piece
        if tmp_str:
            rs_list.append(tmp_str)
        return rs_list

    piece_list = flat([split_by_len(e, max_len) for e in split_text_by_commas(content, pattern)])
    rs_list = merge_piece(piece_list, max_len)
    return rs_list


# split token_list into sequences shorter than max_len
def split_token(token_list, max_len, pattern=r"，|。|；|\.|,|;|\?|!"):
    nature_split_list = split_token_by_commas(token_list, pattern)

    def merge_token_piece(token_piece_list, max_len):
        rs_list = []
        tmp_list = []
        for piece in token_piece_list:
            if len(piece) > max_len:
                rs_list.append(piece)
                continue
            if len(tmp_list) + len(piece) > max_len:
                rs_list.append(tmp_list)
                tmp_list = piece
            else:
                tmp_list.extend(piece)
        if tmp_list:
            rs_list.append(tmp_list)
        return rs_list

    def split_token_by_len(token_list, max_len):
        rs_list = []
        beg = 0
        while beg < len(token_list):
            end = beg + max_len
            if end >= len(token_list):
                rs_list.append(token_list[beg:end])
            else:
                while token_list[end].startswith("##"):
                    end -= 1
                if end == beg:
                    end = beg + max_len
                rs_list.append(token_list[beg:end])
            beg = end
        return rs_list

    merged_list = merge_token_piece(nature_split_list, max_len)
    rs_list = flat(split_token_by_len(e, max_len) for e in merged_list)
    return rs_list


# split data to sequences shorter than max_len
def split_data2seq_list(data, max_len, key="token"):
    rs_list = []
    for idx, item in enumerate(data):
        if key == "token":
            token_list = split_token(item['token'], max_len)
            offset = 0
            for token in token_list:
                tmp_dict = dict(**item)
                tmp_dict.update(token=token, idx=idx, offset=offset)
                rs_list.append(tmp_dict)
                offset += len(token)
    return rs_list


# add offset to a span
def add_offset(span, offset):
    return tuple([e + offset for e in list(span)])


class PythonObjectEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, bytes):
            return str(obj, encoding="utf8")
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, set):
            return list(obj)

        if isinstance(obj, (list, dict, str, int, set, float, bool, type(None))):
            return json.JSONEncoder.default(self, obj)
        return {'_python_object': pickle.dumps(obj)}


def as_python_object(dct):
    if '_python_object' in dct:
        return pickle.loads(str(dct['_python_object']))
    return dct


# my json dumps
def jdumps(obj):
    return json.dumps(obj, ensure_ascii=False, indent=4, cls=PythonObjectEncoder)


# my json dump
def jdump(obj, fp):
    if isinstance(fp, str):
        fp = codecs.open(fp, 'w', 'utf8')
    return json.dump(obj, fp, ensure_ascii=False, indent=4, cls=PythonObjectEncoder)


def jdump_lines(obj, fp):
    content = "\n".join([json.dumps(e, ensure_ascii=False) for e in obj])
    if isinstance(fp, str):
        fp = codecs.open(fp, 'w', 'utf8')
    with fp as fp:
        fp.write(content)


# my json load
def jload(fp):
    if isinstance(fp, str):
        fp = codecs.open(fp, 'r', 'utf8')
    return json.load(fp, object_hook=as_python_object)


# my json loads
def jloads(s):
    return json.loads(s, object_hook=as_python_object)


# convert cfg data to dict
def cfg2dict(cfg):
    sections = cfg.sections()
    rs = {k: dict(cfg[k]) for k in sections}
    return rs


def read_config(config_path):
    def eval_param(param):
        if isinstance(param, str):
            try:
                rs = eval(param)
                return rs
            except:
                return param
        if isinstance(param, dict):
            return {k: eval_param(v) for k, v in param.items()}

    cfg = ConfigParser()
    cfg.read(config_path)
    cfg_dict = cfg2dict(cfg)
    cfg_dict = eval_param(cfg_dict)
    return cfg_dict


# find all file path which path match pattern path_pattern
def get_path_list(path_pattern):
    path_list = []
    dir_path = os.path.dirname(path_pattern)
    file_pattern = re.compile(os.path.basename(path_pattern))
    for root, dirs, files in os.walk(dir_path):
        for f in files:
            if re.match(file_pattern, f):
                path_list.append(os.path.join(root, f))
    return path_list


# surround info with stars
def star_surround_info(info, fix_length=128):
    star_num = max(fix_length - len(info), 2)
    left_star_num = star_num // 2
    right_star_num = star_num - left_star_num
    rs = "*" * left_star_num + info + "*" * right_star_num
    return rs


# print info with stars surround
def print_info(info, logger=None, fix_length=128):
    star_info = star_surround_info(info, fix_length)
    if logger:
        logger.info(star_info)
    else:
        print(star_info)


def read_json_data(data_path):
    if isinstance(data_path, list):
        rs = []
        for dp in data_path:
            rs += jload(codecs.open(dp, 'r'))
        return rs
    else:
        return jload(codecs.open(data_path, 'r'))


# convert token_span to text_span
def convert_span(span, idx_mapping):
    start, end = span
    if start >= len(idx_mapping) or end >= len(idx_mapping):
        return None
    return idx_mapping[start], idx_mapping[end]


# split data to two parts with fraction
def split_data(data, fraction):
    random.shuffle(data)
    tgt_len = int(len(data) * fraction)
    return data[:tgt_len], data[tgt_len:]


def count_by_label(data, label):
    rs_dict = defaultdict(int)
    for item in data:
        key = item[label]
        rs_dict[key] += 1
    return dict(rs_dict)


# return shuffled copy of data without change origin data
def get_shuffle_data(data):
    rs_list = copy.copy(data)
    random.shuffle(rs_list)
    return rs_list


def sample_data_by_num(data, num):
    if num >= len(data):
        return data
    else:
        rs_list = get_shuffle_data(data)
        return rs_list[:num]


# sample data with fraction
def sample_data(data, fraction=None, label_key=None):
    def sample_by_fraction(data, fraction):
        if not data or fraction > 1:
            return data
        num = int(len(data) * fraction)
        return sample_data_by_num(data, num)

    if label_key:
        # average sample on different label
        label_distribute = count_by_label(data, label_key)
        rs_list = [sample_data([e for e in data if e[label_key] == l], fraction) for l in label_distribute.keys()]
        rs_list = flat(rs_list)
        return rs_list

    else:
        return sample_by_fraction(data, fraction)


# find all the start index and end index of text in seq with char level
def find_all_char(content, to_find, overlap=False, ignore_case=False):
    rs_list = []
    if ignore_case:
        content = content.lower()
        to_find = to_find.lower()
    if not to_find:
        return rs_list
    text_len = len(to_find)

    beg = 0
    while True:
        b = content.find(to_find, beg)
        if b == -1:
            return rs_list
        e = b + text_len
        rs_list.append((b, e))
        beg = b + 1 if overlap else e
    return rs_list


def read_line_data(path):
    rs_list = []
    with codecs.open(path, "r", "utf8") as f:
        for line in f:
            rs_list.append(line.strip())
    return rs_list


REMOVE_PATTERN = re.compile(
    r"[^\u3400-\u9FFF\w\d\n\r\·\~\！\@\#\￥\%\……\&\*\（\）\——\-\+\=\【\】\{\}\、\|\；\‘\’\：\“\”\《\》\？\，\。\、\`\~\!\#\$\%\^\&\*\(\)\_\[\]{\}\\\|\;\'\'\:\"\"\,\.\/\<\>\?]")


# remove useless chars in text
def pure_text(text):
    return re.sub(REMOVE_PATTERN, "", text)


def get_now_str(fmt="%Y-%m-%d-%H:%M:%S"):
    return datetime.strftime(datetime.now(), fmt)


def get_distribution(l, key=lambda x: x):
    group_by_dict = group_by(l, key)
    rs_dict = {k: len(v) for k, v in group_by_dict.items()}
    return rs_dict


def group_by(l, key=lambda x: x):
    rs_dict = defaultdict(list)
    for i in l:
        rs_dict[key(i)].append(i)
    return OrderedDict(sorted(rs_dict.items(), key=lambda x: len(x[1]), reverse=True))


def get_major_element(l):
    return list(get_distribution(l).items())[0][0]


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def list_find(l, ele):
    try:
        return l.index(ele)
    except Exception as e:
        return -1


def get_multihot_list(max_len, idx_list):
    rs = [0] * max_len
    for idx in idx_list:
        rs[idx] = 1
    return rs


def split_text_by_sep(text, sep_list):
    idx_dict = {k: find_all_char(text, k) for k in sep_list}
    idx_list = flat([(k, s) for s in span_list] for k, span_list in idx_dict.items())
    idx_list = sorted(idx_list, key=lambda x: x[1][0])
    beg = 0
    rs_list = []
    for k, (s, e) in idx_list:
        rs_list.append(text[beg:s])
        rs_list.append(k)
        beg = e
    rs_list.append(text[beg:])
    return rs_list


def array2zero_one_with_threshold(arr, threshold=0.5):
    rs_arr = np.zeros_like(arr)
    rs_arr[arr >= threshold] = 1
    rs_arr = rs_arr.astype(np.int32)
    return rs_arr


def compress_file(src_path, dst_path):
    file_name = os.path.basename(src_path)
    dir_path = os.path.dirname(src_path)
    cmd = "cd {dir_path};tar -cvf {dst_path} {file_name}".format(dir_path=dir_path, dst_path=dst_path,
                                                                 file_name=file_name)
    return execute_cmd(cmd)


def upload_file2oss(file_path, oss_dir="oss://modelzoo/dev/chenhao/openie/"):
    file_name = os.path.basename(file_path)
    dst_path = os.path.join(oss_dir, file_name)
    cmd = "osscmd put {file_path} {dst_path}".format(file_path=file_path, dst_path=dst_path)
    return execute_cmd(cmd)
