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
from eigen_nltk.utils import split_token, flat


# remove the '#' in token
def remove_token_char(token):
    return token.replace("##", "")


# get text index from token index
def get_token2char_mapping(token_list, text):
    rs = [-1]
    text = text.lower()
    idx = 0
    for t in token_list[1:-1]:
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
        rs[value: end] = [idx + 1] * (end - value)
        end = value
    rs.append(len(token2char_mapping) + 1)
    return rs


class DataParser(object):
    def __init__(self, context):
        self.context = context
        self.tokenizer = context.tokenizer

    # get bert input from text
    def get_bert_input(self, text, second_text=None):
        # tmp_text = re.sub(r"\s", BERT_SPACE_TOKEN, text)
        x, seg = self.tokenizer.encode(text, second_text)
        token = self.tokenizer.tokenize(text, second_text)
        return dict(x=x, seg=seg, token=token, text=text)

    # get ner model input from text
    def get_token_input(self, text):
        bert_input = self.get_bert_input(text)
        token2char_mapping = get_token2char_mapping(bert_input['token'], text)
        char2token_mapping = get_char2token_mapping(text, token2char_mapping)
        return dict(**bert_input, token2char_mapping=token2char_mapping, char2token_mapping=char2token_mapping)

    def item2short_item(self, idx, item, max_len):
        assert max_len > 2
        rs_list = []
        content = item["content"]
        detail_info = self.get_token_input(content)
        token_list = detail_info["token"][1:-1]
        short_token_list = split_token(token_list, max_len - 2)
        token2char_mapping = detail_info["token2char_mapping"]
        beg = 1
        for short_token in short_token_list:
            tmp_item = dict(**item)
            end = beg + len(short_token)
            s, e = token2char_mapping[beg], token2char_mapping[end]
            tmp_content = content[s: e]
            beg = end
            tmp_item.update(content=tmp_content, idx=idx, offset=s)
            rs_list.append(tmp_item)
        return rs_list

    def get_short_data(self, data, max_len):
        return flat([self.item2short_item(idx, item, max_len) for idx, item in enumerate(data)])
