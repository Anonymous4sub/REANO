import sys
sys.path.append("/nfs/reano/common")

import os 
import re
import spacy 
import nltk
from nltk import sent_tokenize
import pickle
import random
import logging
import numpy as np 
import pandas as pd 
from tqdm import tqdm 
from collections import defaultdict

import torch
from torch.utils.data import Dataset

from my_utils import load_json, save_json, load_id2json, convert_to_unicode, contain_answers, remove_bracket
from my_evaluation import ems 

logger = logging.getLogger(__name__)


class TripleFiDDataset(Dataset):

    def __init__(self, data_path, n_context=None, question_prefix='question:', title_prefix='title:', passage_prefix='context:', max_num_entities=2000):

        self.data = self.load_data(data_path)
        self.n_context = n_context
        self.question_prefix = question_prefix
        self.title_prefix = title_prefix
        self.passage_prefix = passage_prefix
        self.max_num_entities = max_num_entities
        self.sort_data()
        self.use_entityname = "entityquestion" in data_path

    def load_data(self, data_path):
        
        print("Loading data from {} ... ".format(data_path))
        data = pickle.load(open(data_path, "rb"))
        examples = [] 
        for k, example in enumerate(data):
            if not 'id' in example:
                example['id'] = k 
            for c in example['ctxs']:
                if not 'score' in c:
                    c['score'] = 1.0 
            examples.append(example)

        return examples 

    def __len__(self):
        return len(self.data)

    def get_target(self, example):
        if 'target' in example:
            target = example['target']
            return target + ' </s>'
        elif 'answers' in example:
            return random.choice(example['answers']) + ' </s>'
        else:
            return None

    def get_context_text(self, text_with_entity, entity_list, batch_ent_map, batch_entity_name, batch_ent_is_answer, answers, entityid2name):

        entity_type = []
        sorted_entity_list = sorted(entity_list, key=lambda x: x[0])
        for start_idx, end_idx, mention, entity_id in sorted_entity_list:

            if entity_id not in batch_ent_map:
                batch_ent_map[entity_id] = len(batch_ent_map)
                if not self.use_entityname:
                    entity_name = remove_bracket(mention)
                else:
                    entity_name = remove_bracket(entityid2name[entity_id][0])
                batch_entity_name.append(entity_name)
                batch_ent_is_answer[batch_ent_map[entity_id]] = self.is_answer(entity_name, answers)

            entity_type.append(batch_ent_map[entity_id])

        return text_with_entity, entity_type
    
    def is_answer(self, text, answers):
        return ems(text, answers)

    def __getitem__(self, index):

        example = self.data[index]
        question = self.question_prefix + " " + example['question']
        gold_answers = example["answers"] 
        target = self.get_target(example)
        entityid2name = example["entityid2name"]

        batch_ent_map = {}
        batch_entity_name = []
        batch_ent_is_answer = {}

        question_entity_list = [] 
        for i, question_entity in enumerate(example["question_entity"]):
            batch_ent_map[i] = len(batch_ent_map)
            entity_name = remove_bracket(question_entity)
            batch_entity_name.append(entity_name)
            batch_ent_is_answer[batch_ent_map[i]] = self.is_answer(entity_name, answers=gold_answers)
            question_entity_list.append(batch_ent_map[i])
        
        if 'ctxs' in example and self.n_context is not None:

            f = self.title_prefix + " {} " + self.passage_prefix + " {}"
            contexts = example['ctxs'][:self.n_context] 

            passages, entity_type_list = [], []
            for c in contexts:
                title, title_entity = self.get_context_text(c["title"], c["title_entity"], batch_ent_map, batch_entity_name, batch_ent_is_answer, gold_answers, entityid2name)
                text, text_entity = self.get_context_text(c["text"], c["text_entity"], batch_ent_map, batch_entity_name, batch_ent_is_answer, gold_answers, entityid2name)
                passages.append(f.format(title, text))
                entity_type_list.append(title_entity + text_entity)

            scores = [float(c['score']) for c in contexts]
            scores = torch.tensor(scores)

            if len(passages) < self.n_context:
                while len(passages) < self.n_context:
                    passages = passages + passages[:(self.n_context-len(passages))]
                    entity_type_list = entity_type_list + entity_type_list[:(self.n_context-len(entity_type_list))]

        else:
            passages, scores, entity_type_list = None, None, None

        num_entity = len(batch_ent_map)
        
        triples = []
        for heid, rel, teid in example.get("triples", []): 
            if heid in batch_ent_map and teid in batch_ent_map:
                triple = (batch_ent_map[heid], rel, batch_ent_map[teid])
                if triple not in triples:
                    triples.append(triple)

        entity_is_answer_list = [batch_ent_is_answer[i] for i in range(len(batch_entity_name))] 
        relevant_triples = []
        for heid, rel, teid in example.get("relevant_triples", []):
            if heid in batch_ent_map and teid in batch_ent_map:
                triple = (batch_ent_map[heid], rel, batch_ent_map[teid])
                entity_is_answer_list[batch_ent_map[heid]] = True
                entity_is_answer_list[batch_ent_map[teid]] = True
                if triple not in relevant_triples:
                    relevant_triples.append(triple)

        return {
            'index' : index,
            'question' : question,
            'target' : target,
            'passages' : passages,
            'scores' : scores,
            "num_entity": num_entity, 
            'entity': batch_entity_name,
            'entity_type_list': entity_type_list,
            "entity_is_answer_list": entity_is_answer_list,
            "triples": triples, 
            "relevant_triples": relevant_triples,
            "evidences": example.get("evidences", None), 
            "question_entity": question_entity_list, 
        }

    def sort_data(self):
        if self.n_context is None or not 'score' in self.data[0]['ctxs'][0]:
            return
        for ex in self.data:
            ex['ctxs'].sort(key=lambda x: float(x['score']), reverse=True)

    def get_example(self, index):
        return self.data[index]
