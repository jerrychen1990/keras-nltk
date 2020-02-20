# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     trans
   Description :
   Author :       chenhao
   date：          2019-09-19
-------------------------------------------------
   Change Activity:
                   2019-09-19:
-------------------------------------------------
"""
from tqdm import tqdm
from eigen_nltk.utils import split_token, flat
from eigen_nltk.constants import *


# remove the '#' in token
def remove_token_char(token):
    return token.replace("##", "")


# get text index from token index
def get_token2char_mapping(token_list, text, token_offset):
    rs = [-1] * token_offset
    text = text.lower()
    idx = 0
    for t in token_list[token_offset:-1]:
        rs.append(idx)
        while idx < len(text) and text[idx] == " ":
            idx += 1
        t = remove_token_char(t)
        idx += len(t)
    rs.append(len(text))
    return rs


# get token index from char index
def get_char2token_mapping(text, token2char_mapping):
    rs = [0] * len(text)
    end = len(rs)
    token2char_mapping = token2char_mapping[1:-1]
    for idx, value in list(enumerate(token2char_mapping))[::-1]:
        if value == -1:
            break
        rs[value: end] = [idx + 1] * (end - value)
        end = value
    rs.append(len(token2char_mapping) + 1)
    return rs


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):].strip()
    else:
        print("error!", text)


def add_prefix(text, prefix):
    return " ".join([prefix, text])


def add_entity_tag(text, span_list):
    assert len(span_list) <= 2
    idx_list = [e[0] for e in span_list] + [e[1] for e in span_list]
    offset = 0
    char_list = list(text)
    for idx, tag in sorted(zip(idx_list, SPECIAL_TOKEN_LIST), key=lambda x: x[0]):
        true_idx = idx + offset
        char_list.insert(true_idx, tag)
        offset += 1
    return "".join(char_list)


class DataParser(object):
    def __init__(self, context):
        self.context = context
        self.tokenizer = context.tokenizer

    # get bert input from text
    def get_bert_input(self, text, second_text=None):
        x, seg = self.tokenizer.encode(text, second_text)
        token = self.tokenizer.tokenize(text, second_text)
        full_text = text + " " + second_text if second_text else text
        return dict(x=x, seg=seg, token=token, text=full_text)

    # get ner model input from text
    def get_token_input(self, content, prefix=None, pos=None):
        text, second_text = (prefix, content) if prefix else (content, None)
        bert_input = self.get_bert_input(text, second_text)
        token_offset = bert_input['seg'].index(1) if prefix else 1

        token2char_mapping = get_token2char_mapping(bert_input['token'], content, token_offset)
        char2token_mapping = get_char2token_mapping(content, token2char_mapping)
        rs = dict(**bert_input, token_offset=token_offset,
                  token2char_mapping=token2char_mapping, char2token_mapping=char2token_mapping)
        if pos and self.context.pos_size:
            pos_idx = [token2char_mapping[idx] for idx in range(len(bert_input['token']))]
            pos_token = [pos[idx] if 0 <= idx < len(pos) else '[PAD]' for idx in pos_idx]
            pos_input = [self.context.pos2id.get(p, 0) for p in pos_token]
            rs.update(pos_token=pos_token, pos_input=pos_input)
        return rs

    def item2short_item(self, idx, item, max_len):
        assert max_len > 2
        rs_list = []
        content = item["content"]
        pos = item['pos']
        prefix = item.get("prefix", None)
        detail_info = self.get_token_input(content, prefix)
        token_offset = detail_info['token_offset']
        token_list = detail_info["token"][token_offset:-1]
        short_token_list = split_token(token_list, max_len - 1 - token_offset)
        token2char_mapping = detail_info["token2char_mapping"]
        beg = token_offset
        # print(idx, len(short_token_list))
        for short_token in short_token_list:
            tmp_item = dict(**item)
            end = beg + len(short_token)
            s, e = token2char_mapping[beg], token2char_mapping[end]
            tmp_content = content[s: e]
            tmp_pos = pos[s:e]
            beg = end
            tmp_item.update(content=tmp_content, idx=idx, offset=s, pos=tmp_pos)
            rs_list.append(tmp_item)
        return rs_list

    def get_short_data(self, data, max_len):
        rs_list = []
        for idx, item in tqdm(iterable=enumerate(data), mininterval=5):
            tmp_list = self.item2short_item(idx, item, max_len)
            rs_list.extend(tmp_list)
        return rs_list
