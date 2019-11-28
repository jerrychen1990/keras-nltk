# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     open_ie
   Description :
   Author :       chenhao
   date：          2019-11-21
-------------------------------------------------
   Change Activity:
                   2019-11-21:
-------------------------------------------------
"""
from eigen_nltk.core import PipelineEstimator
from eigen_nltk.ner import NerExtractor
from eigen_nltk.nre import NreExtractor
from eigen_nltk.utils import jdumps, flat
from collections import defaultdict
import copy


def find_span(span_list, span):
    for s, e in span_list:
        if s >= span[1]:
            return s, e
    return span_list[-1]


def nre2condition_ner(nre_item):
    rs_list = []
    for idx, (rel, (e1, et1, s1), (e2, et2, s2)) in enumerate(nre_item['rel_list']):
        given_entity_list = [(e1, et1, s1), (e2, et2, s2)]
        tmp_item = dict(id=str(nre_item['id']) + "_" + str(idx), father_id=nre_item['id'], content=nre_item.content,
                        given_entity_list=given_entity_list)
        rs_list.append(tmp_item)
    return rs_list


def group_predicate_item(predicate_batch, ner_batch):
    rs_dict = defaultdict(list)
    for predicate, item in zip(predicate_batch, ner_batch):
        if predicate:
            key = item['father_id']
            spo_list = [(p, item['given_entity_list'][0], item['given_entity_list'][0]) for p in predicate]
            rs_dict[key].append(spo_list)
    return rs_dict


def get_simple_spo(rel_list, span_spo_list):
    rs_list = rel_list + span_spo_list
    rs_list = [(e[1][0], e[0], e[2][0]) for e in rs_list]

    rs_list = list(set(rs_list))
    return rs_list


class SchemaFreeSPOExtractor(PipelineEstimator):
    def __init__(self, name, ner_extractor, nre_extractor, predication_ner_extractor, logger_level="INFO"):
        super().__init__(name, logger_level)
        assert isinstance(ner_extractor, NerExtractor)
        assert isinstance(nre_extractor, NreExtractor)
        self.ner_extractor = ner_extractor
        self.nre_extractor = nre_extractor
        self.predication_ner_extractor = predication_ner_extractor

    def predict_batch(self, batch, batch_size=64, verbose=1, ner_show_detail=False, nre_show_detail=False,
                      remove_rel_list=['None'], **kwargs):
        entity_list = self.ner_extractor.predict_batch(batch, batch_size, verbose, ner_show_detail, **kwargs)
        # print("entity_list", entity_list)
        nre_batch = self._add_rel_list(batch, entity_list)

        rel_list = [[]] * len(batch)

        if nre_batch:
            # print(jdumps(nre_batch[:4]))
            condition_ner_batch = flat([nre2condition_ner(item) for item in nre_batch])
            rel_list = self.nre_extractor.predict_batch(nre_batch, batch_size, verbose, nre_show_detail, **kwargs)
            predicate_batch = self.predication_ner_extractor.predict_batch(condition_ner_batch, batch_size, verbose,
                                                                           nre_show_detail, **kwargs)

        rel_list = [[r for r in rel if r[0] not in remove_rel_list] for rel in rel_list]
        predicate_dict = group_predicate_item(predicate_batch, condition_ner_batch)

        span_predicate_list = [predicate_dict[item['id']] for item in batch]

        rs_list = [dict(entity_list=e, rel_list=s, span_spo_list=sp, spo_list=get_simple_spo(s, sp)) for e, s, sp in
                   zip(entity_list, rel_list, span_predicate_list)]

        return rs_list

    def _add_rel_list(self, batch, entity_list, ordered=True):
        assert len(batch) == len(entity_list)
        rs_list = []
        for entities, item in zip(entity_list, batch):
            tmp_item = copy.copy(item)
            rel_list = []
            for i, (e1, et1, span1) in enumerate(entities):
                second_entities = entities[i + 1:] if ordered else entities
                for j, (e2, et2, span2) in enumerate(second_entities):
                    if i == j and not ordered:
                        continue
                    s1 = span1[0]
                    s2 = find_span(span2, s1)
                    rel = (None, (e1, et1, s1), (e2, et2, s2))
                    rel_list.append(rel)
            tmp_item['rel_list'] = rel_list
            rs_list.append(tmp_item)

        total_rel_num = sum(len(e['rel_list']) for e in rs_list)
        total_entity_num = sum(len(e) for e in entity_list)

        self.logger.info("get {0} entity pairs from {1} entity".format(total_rel_num, total_entity_num))
        return rs_list
