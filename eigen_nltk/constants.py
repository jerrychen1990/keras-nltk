# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     constants
   Description :
   Author :       chenhao
   date：          2019-09-19
-------------------------------------------------
   Change Activity:
                   2019-09-19:
-------------------------------------------------
"""
BIO = 'BIO'
BMESO = 'BMESO'
ENTITY_START_1 = "[ES1]"
ENTITY_START_2 = "[ES2]"
ENTITY_END_1 = "[EE1]"
ENTITY_END_2 = "[EE2]"
SPECIAL_TOKEN_LIST = [ENTITY_START_1, ENTITY_START_2, ENTITY_END_1, ENTITY_END_2]

NER_ANNOTATION_LIST = [BIO, BMESO]
SEP = '[SEP]'
CLS = '[CLS]'
UNK = '[UNK]'
PAD = '[PAD]'
MASK = '[MASK]'

TAG_TOKEN_LIST = [CLS, SEP, UNK, PAD, MASK]
IGNORE_TOKEN_LIST = SPECIAL_TOKEN_LIST + TAG_TOKEN_LIST


