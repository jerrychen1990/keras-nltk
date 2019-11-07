# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     pipeline
   Description :
   Author :       chenhao
   date：          2019-10-10
-------------------------------------------------
   Change Activity:
                   2019-10-10:
-------------------------------------------------
"""
import copy

from eigen_nltk.core import PipelineEstimator
from eigen_nltk.ner import NerExtractor
from eigen_nltk.nre import NreExtractor


class EntityRelationExtractor(PipelineEstimator):

    def __init__(self, name, ner_extractor, nre_extractor, logger_level="INFO"):
        super().__init__(name, logger_level)
        assert isinstance(ner_extractor, NerExtractor)
        assert isinstance(nre_extractor, NreExtractor)
        self.ner_extractor = ner_extractor
        self.nre_extractor = nre_extractor

    def predict_batch(self, batch, batch_size=64, verbose=1, ner_show_detail=False, nre_show_detail=False, **kwargs):
        super().predict_batch(batch, **kwargs)
        entity_list = self.ner_extractor.predict_batch(batch, batch_size, verbose, ner_show_detail, **kwargs)
        # print(entity_list)
        nre_batch = self._add_rel_list(batch, entity_list)
        # print(jdumps(nre_batch[:4]))
        rel_list = self.nre_extractor.predict_batch(nre_batch, batch_size, verbose, nre_show_detail, **kwargs)
        return rel_list

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
                    rel = (None, (e1, et1, span1[0]), (e2, et2, span2[0]))
                    rel_list.append(rel)
            tmp_item['rel_list'] = rel_list
            rs_list.append(tmp_item)

        total_rel_num = sum(len(e['rel_list']) for e in rs_list)
        total_entity_num = sum(len(e) for e in entity_list)

        self.logger.info("get {0} entity pairs from {1} entity".format(total_rel_num, total_entity_num))
        return rs_list
