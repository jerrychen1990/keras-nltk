# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     toolkit
   Description :
   Author :       chenhao
   date：          2019-12-12
-------------------------------------------------
   Change Activity:
                   2019-12-12:
-------------------------------------------------
"""
from eigen_nltk.utils import read_json_data, flat, jdumps, jdump
from eigen_nltk.constants import BMESO, BIO


def k(data_path, schema_path, ner_tag_type):
    data = read_json_data(data_path)
    print("{} data read", len(data))
    entity_type_list = flat([[e[1] for e in item['entity_list']] for item in data])
    entity_type_list = list(set(entity_type_list))
    print("entity_list:{}".format(entity_type_list))
    schema_list = ['O']
    assert ner_tag_type in [BMESO, BIO]

    for entity_type in entity_type_list:
        for prefix in ner_tag_type[:-1]:
            tag = prefix + "_" + entity_type
            schema_list.append(tag)
    schema_dict = {t: idx for idx, t in enumerate(schema_list)}
    print("schema_dict")
    print(jdumps(schema_dict))
    print("writing result to :{}".format(schema_path))
    jdump(schema_dict, schema_path)


if __name__ == '__main__':
    generate_ner_schema_file(
        ["/nfs/pony/chenhao/data/ontonotes_chinese/train.json", "/nfs/pony/chenhao/data/ontonotes_chinese/dev.json",
         "/nfs/pony/chenhao/data/ontonotes_chinese/test.json"], "/nfs/pony/chenhao/schema/ontonotes_chinese_bio.json",
        "BIO")
