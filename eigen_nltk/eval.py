# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     eval
   Description :
   Author :       chenhao
   date：          2019-09-19
-------------------------------------------------
   Change Activity:
                   2019-09-19:
-------------------------------------------------
"""
import copy
from collections import defaultdict
from sklearn.metrics import accuracy_score
from eigen_nltk.utils import flat


# convert ner output list to spo list with span
def ner2spo_detail(entity_list, id):
    tmp = [[(id, e[1], (e[0], tuple(span))) for span in e[2]] for e in entity_list]
    rs = flat(tmp)
    return rs


def ner2spo(entity_list, id):
    return [(id, e[1], e[0]) for e in entity_list]


def nre2spo(rel_list, id):
    return [(id, e[0], (e[1][0], e[2][0])) for e in rel_list]


def entity_cls2spo(entity_list, id):
    return [(id, e[1], e[0]) for e in entity_list]


def label2spo(label, id):
    return [(id, label, id)]


# group spo tuple to dict with prediction as key
def group_spo_by_pred(spo_list):
    rs_dict = defaultdict(set)
    spo_list = flat(spo_list)
    for sub, pred, obj in spo_list:
        rs_dict[pred].add((sub, pred, obj))
    return dict(rs_dict)


# eval precision, recall, f1 value
def eval_spo_set(true_set, pred_set):
    precise_base = len(pred_set)
    recall_base = len(true_set)
    intersection_set = pred_set & true_set
    intersection_num = len(intersection_set)
    precise = 0. if precise_base == 0 else intersection_num * 1. / precise_base
    recall = 0. if recall_base == 0 else intersection_num * 1. / recall_base
    f1 = 0. if precise == 0 or recall == 0 else 2 * recall * precise / (recall + precise)
    return dict(true_num=recall_base, pred_num=precise_base, true_pred_num=intersection_num, precise=precise,
                recall=recall, f1=f1)


def get_macro_avg(eval_list):
    precise = sum([e['precise'] for e in eval_list]) / len(eval_list)
    recall = sum([e['recall'] for e in eval_list]) / len(eval_list)
    f1 = 2 * precise * recall / (precise + recall) if precise + recall > 0 else 0.
    return dict(precise=precise, recall=recall, f1=f1)


# evaluate prediction result
def eval_spo_list(true_data, pred_data):
    assert len(true_data) == len(pred_data)
    item_num = len(true_data)
    true_dict = group_spo_by_pred(true_data)
    pred_dict = group_spo_by_pred(pred_data)
    eval_dict = dict([(key, eval_spo_set(true_dict[key], pred_dict.get(key, set()))) for key in true_dict.keys()])
    eval_list = sorted(eval_dict.items(), key=lambda x: (x[1]['f1'], x[1]['pred_num']), reverse=True)
    macro_avg = get_macro_avg(eval_dict.values())
    true_total_set = set.union(*true_dict.values()) if true_dict else set()
    pred_total_set = set.union(*pred_dict.values()) if pred_dict else set()
    micro_avg = eval_spo_set(true_total_set, pred_total_set)
    return dict(item_num=item_num, detail_eval=eval_list, micro_avg=micro_avg, macro_avg=macro_avg)


# add ner pred field to data
def add_ner_pred(data, pred, contain_span=True):
    rs_list = []
    spo_func = ner2spo_detail if contain_span else ner2spo
    for p, e in zip(pred, data):
        p_set = set(spo_func(p, e['id']))
        t_set = set(spo_func(e['entity_list'], e['id']))
        fp = [list(x) for x in p_set - t_set]
        fn = [list(x) for x in t_set - p_set]
        tp = [list(x) for x in p_set & t_set]
        statistic = "tp_num:{0}, fp_num:{1}, fn_num:{2}".format(len(tp), len(fp), len(fn))

        rs = dict(id=e['id'], content=e['content'], statistic=statistic,
                  entity_list=e['entity_list'], ner_pred=p, ner_fp=fp, ner_fn=fn, ner_tp=tp)
        if "prefix" in e.keys():
            rs.update(prefix=e['prefix'])
        rs_list.append(rs)
    return rs_list


# add ner classify pred field to data
def add_ner_classify_pred(data, pred, contain_span=True):
    rs_list = []
    for p, e in zip(pred, data):
        rs = dict(id=e['id'], content=e['content'], entity_list=e['entity_list'], entity_pred=p['entity_list'],
                  label_list=e['label_list'], label_pred=p['label_list'])
        rs_list.append(rs)
    return rs_list


# add ner pred field to data
def add_nre_pred(data, pred, contain_span=True):
    rs_list = []
    spo_func = nre2spo if contain_span else ner2spo
    for p, e in zip(pred, data):
        p_set = set(spo_func(p, e['id']))
        t_set = set(spo_func(e['rel_list'], e['id']))
        fp = [list(x) for x in p_set - t_set]
        fn = [list(x) for x in t_set - p_set]
        tp = [list(x) for x in p_set & t_set]
        statistic = "tp_num:{0}, fp_num:{1}, fn_num:{2}".format(len(tp), len(fp), len(fn))
        rs = dict(id=e['id'], content=e['content'], statistic=statistic,
                  rel_list=e['rel_list'], nre_pred=p, nre_fp=fp, nre_fn=fn, nre_tp=tp)
        rs_list.append(rs)
    return rs_list


# add ner pred field to data
def add_entity_classify_pred(data, pred):
    rs_list = []
    for p, e in zip(pred, data):
        p_set = set(entity_cls2spo(p, e['id']))
        t_set = set(entity_cls2spo(e['entity_list'], e['id']))
        fp = [list(x) for x in p_set - t_set]
        fn = [list(x) for x in t_set - p_set]
        tp = [list(x) for x in p_set & t_set]
        statistic = "tp_num:{0}, fp_num:{1}, fn_num:{2}".format(len(tp), len(fp), len(fn))

        rs = dict(id=e['id'], content=e['content'], statistic=statistic,
                  entity_list=e['entity_list'], entity_pred=p, entity_cls_fp=fp, entity_cls_fn=fn, entity_cls_tp=tp)
        rs_list.append(rs)
    return rs_list


# add classify pred field to data
def add_classify_pred(data, pred):
    rs_list = []
    for p, e in zip(pred, data):
        rs = dict(idx=e['idx'], content=e['content'], label=e.get('label', None), label_pred=p)
        rs_list.append(rs)
    return rs_list


def add_language_model_pred(data, pred):
    rs_list = []
    for p, e in zip(pred, data):
        rs = dict(content=e, generate_content=p)
        rs_list.append(rs)
    return rs_list


# eval ner_result with test_data
def eval_ner(test_data, ner_pred_list, contain_span=True):
    spo_func = ner2spo_detail if contain_span else ner2spo
    spo_detail_pred = [spo_func(e, item['id']) for e, item in zip(ner_pred_list, test_data)]
    spo_detail_true = [spo_func(item['entity_list'], item['id']) for item in test_data]
    return eval_spo_list(spo_detail_true, spo_detail_pred)


# eval nre_result with test_data
def eval_nre(test_data, nre_pred_list, contain_span=True):
    spo_func = nre2spo if contain_span else nre2spo
    spo_detail_pred = [spo_func(e, item['id']) for e, item in zip(nre_pred_list, test_data)]
    spo_detail_true = [spo_func(item['rel_list'], item['id']) for item in test_data]

    return eval_spo_list(spo_detail_true, spo_detail_pred)


def eval_entity_classify(test_data, nre_pred_list):
    spo_detail_pred = [entity_cls2spo(e, item['id']) for e, item in zip(nre_pred_list, test_data)]
    spo_detail_true = [entity_cls2spo(item['entity_list'], item['id']) for item in test_data]

    return eval_spo_list(spo_detail_true, spo_detail_pred)


def eval_classify(test_data, label_pred):
    label_true = [str(e['label']) for e in test_data]
    label_pred = [str(e) for e in label_pred]
    assert len(label_true) == len(label_pred)
    assert len(label_true) > 0
    accuracy = accuracy_score(label_true, label_pred)
    spo_true = [label2spo(label, idx) for idx, label in enumerate(label_true)]
    spo_pred = [label2spo(label, idx) for idx, label in enumerate(label_pred)]
    spo_eval = eval_spo_list(spo_true, spo_pred)
    spo_eval.update(accuracy=accuracy)
    return spo_eval


def eval_ner_classify(test_data, pred):
    def convert_item(item):
        e_list = copy.copy(item['entity_list'])
        for t in item['label_list']:
            e_list.append((t, 'LABEL_ENTITY', None))
        rs_item = dict(**item)
        rs_item['entity_list'] = e_list
        return rs_item

    test_data = [convert_item(item) for item in test_data]

    def convert_pred(pred):
        rs_list = copy.copy(pred['entity_list'])
        for t in pred['label_list']:
            rs_list.append((t, 'LABEL_ENTITY', None))
        return rs_list

    pred_data = [convert_pred(p) for p in pred]
    return eval_ner(test_data, pred_data, False)


def eval_language_model(test_data, lm_pred):
    return {}
