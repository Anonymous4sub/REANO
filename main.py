
import os
import glob
import pickle
import random
import logging 
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from src.options import Options

import src.slurm
import src.util
import src.evaluation
import src.data
import src.model
from src.options import Options

import transformers
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule

from common.data.my_datasets import TripleFiDDataset
from common.models.fid import TripleKGFiDT5
from common.my_trainer import BaseTrainer
from common.my_utils import gpu_setup, cleanup, get_dataloader, setup_logger, load_json


logger = logging.getLogger(__name__)

def encode_passages(batch_text_passages, tokenizer, max_length):

    passage_ids, passage_masks = [], []
    for k, text_passages in enumerate(batch_text_passages):
        p = tokenizer.batch_encode_plus(
            text_passages,
            max_length=max_length,
            pad_to_max_length=True,
            return_tensors='pt',
            truncation=True
        )
        passage_ids.append(p['input_ids'][None])
        passage_masks.append(p['attention_mask'][None])

    passage_ids = torch.cat(passage_ids, dim=0)
    passage_masks = torch.cat(passage_masks, dim=0)

    return passage_ids, passage_masks.bool()


def encode_entities(batch_entities, tokenizer, batch_max_num_entities):

    all_entities = [] 

    for entities in batch_entities:
        # padding 
        pad_entities = entities + [""] * (batch_max_num_entities - len(entities))
        all_entities.extend(pad_entities)

    outputs = tokenizer.batch_encode_plus(
        all_entities, 
        padding=True, 
        truncation=True,
        max_length=16,
        return_tensors='pt',
        add_special_tokens=False, # NOTE: entity的id不添加eos_token 
    )

    batch_size = len(batch_entities)
    entity_input_ids = outputs["input_ids"].reshape(batch_size, batch_max_num_entities, -1)
    entity_attention_mask = outputs["attention_mask"].reshape(batch_size, batch_max_num_entities, -1)

    return entity_input_ids, entity_attention_mask.bool()


class FiDCollator(object):

    def __init__(self, text_maxlength, tokenizer, relation2id, answer_maxlength=20, max_num_entities=2000, max_num_mention_per_entity=50, max_num_edge=25):

        self.tokenizer = tokenizer
        self.relation2id = relation2id
        self.text_maxlength = text_maxlength
        self.answer_maxlength = answer_maxlength
        self.max_num_entities = max_num_entities
        self.max_num_mention_per_entity = max_num_mention_per_entity
        self.max_num_edge = max_num_edge

        self.sep_token_id = self.tokenizer.encode("[SEP]")[0]
        self.cls_token_id = self.tokenizer.encode("[CLS]")[0]
        self.ent_start_id = self.tokenizer.encode("<e>")[0]
        self.ent_end_id = self.tokenizer.encode("</e>")[0]

    def __call__(self, batch):

        assert(batch[0]['target'] != None)
        index = torch.tensor([ex['index'] for ex in batch])
        target = [ex['target'] for ex in batch]
        target = self.tokenizer.batch_encode_plus(
            target,
            max_length=self.answer_maxlength if self.answer_maxlength > 0 else None,
            pad_to_max_length=True,
            return_tensors='pt',
            truncation=True if self.answer_maxlength > 0 else False,
        )
        target_ids = target["input_ids"]
        target_mask = target["attention_mask"].bool()
        target_ids = target_ids.masked_fill(~target_mask, -100)
        
        def append_question(example):
            if example['passages'] is None:
                return [self.maybe_truncate_question(example['question'])]
            return [self.maybe_truncate_question(example['question']) + " [SEP] " + t for t in example['passages']]
        
        text_passages = [append_question(example) for example in batch]

        passage_ids, passage_masks = encode_passages(text_passages, self.tokenizer, self.text_maxlength) 
        
        batch_size, num_passages = passage_ids.shape[0], passage_ids.shape[1]
        flatten_passage_ids = passage_ids.reshape(-1, self.text_maxlength)
        B, maxlen = flatten_passage_ids.shape

        batch_question_text = [self.maybe_truncate_question(example["question"]) for example in batch]
        indices = torch.arange(maxlen)[None, :].expand(B, -1)
        question_length = (flatten_passage_ids == self.sep_token_id).nonzero()[:, -1:]
        question_mask = indices < question_length # B x maxlen 

        ent_start_row_indices, ent_start_col_indices = (flatten_passage_ids == self.ent_start_id).nonzero(as_tuple=True)
        ent_end_row_indices, ent_end_col_indices = (flatten_passage_ids == self.ent_end_id).nonzero(as_tuple=True)

        batch_max_num_entities = min(max([example["num_entity"] for example in batch]), self.max_num_entities)
        batch_entity_type_list = [example["entity_type_list"] for example in batch]
        batch_entity_num_mention = torch.zeros((batch_size, batch_max_num_entities), dtype=torch.long)
        batch_entity_mention_indices = torch.zeros((batch_size, batch_max_num_entities, self.max_num_mention_per_entity), dtype=torch.long)
        batch_entity_mention_passage_indices = torch.zeros((batch_size, batch_max_num_entities, self.max_num_mention_per_entity), dtype=torch.long)
        batch_entity_mention_mask = torch.zeros((batch_size, batch_max_num_entities, self.max_num_mention_per_entity), dtype=torch.bool)

        max_num_entity_per_passage = 25 
        batch_passage_entity_length = torch.zeros((batch_size, num_passages), dtype=torch.long)
        batch_passage_entity_ids = torch.zeros((batch_size, num_passages, max_num_entity_per_passage), dtype=torch.long)
        batch_passage_entity_mask = torch.zeros((batch_size, num_passages, max_num_entity_per_passage), dtype=torch.bool)

        max_num_edge_per_entity = self.max_num_edge
        batch_entity_adj = torch.zeros((batch_size, batch_max_num_entities, max_num_edge_per_entity), dtype=torch.long)
        batch_entity_num_edge = torch.zeros((batch_size, batch_max_num_entities), dtype=torch.long)
        batch_entity_adj_mask = torch.zeros((batch_size, batch_max_num_entities, max_num_edge_per_entity), dtype=torch.bool)
        batch_entity_adj_relation = torch.zeros((batch_size, batch_max_num_entities, max_num_edge_per_entity), dtype=torch.long) 
        batch_entity_adj_relevant_relation_label = torch.zeros((batch_size, batch_max_num_entities, max_num_edge_per_entity), dtype=torch.long) 
        batch_triples = [example["triples"] for example in batch]
        batch_relevant_triples = [example["relevant_triples"] for example in batch] 

        batch_has_mention_entities = [set() for i in range(batch_size)] 

        for i in range(B):

            batch_idx, passage_idx = i // num_passages, i % num_passages

            for ent_start_idx, ent_end_idx, ent_type in zip(
                ent_start_col_indices[ent_start_row_indices==i],
                ent_end_col_indices[ent_end_row_indices==i], 
                batch_entity_type_list[batch_idx][passage_idx]
            ):
                
                if ent_type >= batch_max_num_entities:
                    continue
                num_existing_mention = batch_entity_num_mention[batch_idx, ent_type]
                if num_existing_mention >= self.max_num_mention_per_entity:
                    continue

                batch_entity_mention_indices[batch_idx, ent_type, num_existing_mention] = ent_start_idx
                batch_entity_mention_passage_indices[batch_idx, ent_type, num_existing_mention] = passage_idx 
                batch_entity_mention_mask[batch_idx, ent_type, num_existing_mention] = True 
                batch_entity_num_mention[batch_idx, ent_type] = num_existing_mention + 1 

                num_existing_entity = batch_passage_entity_length[batch_idx, passage_idx]
                if num_existing_entity < max_num_entity_per_passage:
                    batch_passage_entity_ids[batch_idx, passage_idx, num_existing_entity] = ent_type
                    batch_passage_entity_mask[batch_idx, passage_idx, num_existing_entity] = True
                    batch_passage_entity_length[batch_idx, passage_idx] = num_existing_entity + 1 

                batch_has_mention_entities[batch_idx].add(ent_type)

        for batch_idx, triples in enumerate(batch_triples):
            for head, rel, tail in triples:
                if not self.is_valid_triple(head, rel, tail, batch_max_num_entities, batch_has_mention_entities[batch_idx]):
                    continue
                existing_num_neighbor = batch_entity_num_edge[batch_idx, head]
                if existing_num_neighbor >= max_num_edge_per_entity:
                    continue
                existing_neighbors = set(batch_entity_adj[batch_idx, head, :existing_num_neighbor].tolist())
                if tail in existing_neighbors:
                    continue
                batch_entity_adj[batch_idx, head, existing_num_neighbor] = tail
                batch_entity_adj_mask[batch_idx, head, existing_num_neighbor] = True
                batch_entity_adj_relation[batch_idx, head, existing_num_neighbor] = self.relation2id[rel]
                batch_entity_num_edge[batch_idx, head] = existing_num_neighbor + 1 

        for batch_idx, triples in enumerate(batch_relevant_triples):
            for head, rel, tail in triples:
                existing_num_neighbor = batch_entity_num_edge[batch_idx, head]
                tail_index = (batch_entity_adj[batch_idx, head, :existing_num_neighbor] == tail).nonzero()
                if len(tail_index) == 0:
                    continue
                tail_index = tail_index[0].item()
                batch_entity_adj_relevant_relation_label[batch_idx, head, tail_index] = 1 

        max_question_len = (question_mask != 0).max(0)[0].nonzero(as_tuple=False)[-1].item() + 1 
        batch_question_indices = torch.arange(max_question_len)[None, :].expand(B, -1)
        batch_question_mask = batch_question_indices < question_length


        batch_entity_mention_indices = batch_entity_mention_indices + maxlen * batch_entity_mention_passage_indices
        batch_max_num_mention_per_entity = (batch_entity_mention_mask.reshape(-1, self.max_num_mention_per_entity) != 0).max(0)[0].nonzero(as_tuple=False)[-1].item() + 1 
        batch_entity_mention_indices = batch_entity_mention_indices[..., :batch_max_num_mention_per_entity]
        batch_entity_mention_mask = batch_entity_mention_mask[..., :batch_max_num_mention_per_entity]

        batch_max_num_entity_per_passage = (batch_passage_entity_mask.reshape(-1, max_num_entity_per_passage) != 0).max(0)[0].nonzero(as_tuple=False)[-1].item() + 1 
        batch_passage_entity_ids = batch_passage_entity_ids[..., :batch_max_num_entity_per_passage]
        batch_passage_entity_mask = batch_passage_entity_mask[..., :batch_max_num_entity_per_passage]

        batch_entity_text = [example["entity"] for example in batch]

        batch_entity_is_answer_label = self.get_entity_is_answer_label(batch, batch_max_num_entities)

        return (index, target_ids, target_mask, passage_ids, passage_masks, batch_question_text, batch_question_indices, batch_question_mask, \
                batch_entity_mention_indices, batch_entity_mention_mask, batch_entity_is_answer_label, batch_entity_text, \
                    batch_entity_adj, batch_entity_adj_mask, batch_entity_adj_relation, batch_entity_adj_relevant_relation_label, \
                        batch_passage_entity_ids, batch_passage_entity_mask)
    
    def maybe_truncate_question(self, question, max_num_words=100):
        words = question.split()
        if len(words) > max_num_words:
            question = " ".join(words[:max_num_words])
        return question 

    def is_valid_triple(self, head, rel, tail, max_num_entities, has_mention_entities):
        if head >= max_num_entities or tail >= max_num_entities:
            return False
        if rel not in self.relation2id:
            return False
        if head not in has_mention_entities or tail not in has_mention_entities:
            return False
        return True

    def get_entity_is_answer_label(self, batch, max_num_entities):

        batch_size = len(batch)
        entity_is_answer_label = torch.zeros((batch_size, max_num_entities), dtype=torch.long)
        for i, example in enumerate(batch):
            entity_is_answer_list = example["entity_is_answer_list"]
            num_entities = len(entity_is_answer_list)
            entity_is_answer_label[i, :num_entities] = torch.tensor(entity_is_answer_list, dtype=torch.long)

        return entity_is_answer_label


class ReaderTrainer(BaseTrainer):


    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)
        self.mask_passages = kwargs.get("mask_passages", False)
        self.num_passages_after_mask = kwargs.get("num_passages_after_mask", 10)
        if self.mask_passages:
            logger.info(f"---->> mask passages is True, will mask passages during evaluation and only use {self.num_passages_after_mask} passages!")

    def training_step(self, model, batch):
        
        (idx, labels, _, context_ids, context_mask, question_text, question_indices, question_mask, \
            ent_indices, ent_mask, ent_is_ans, entity_text, entity_adj, entity_adj_mask, \
                entity_adj_relation, entity_adj_relevant_relation_label, passage_entity_ids, passage_entity_mask) = batch
        
        calculate_ans_loss = True
        calculate_kg_loss = True

        train_loss = model(
            input_ids=context_ids, 
            attention_mask = context_mask, 
            question_indices=question_indices,
            question_mask=question_mask,
            ent_indices=ent_indices,
            ent_mask=ent_mask,
            entity_text=entity_text, 
            entity_adj=entity_adj, 
            entity_adj_mask=entity_adj_mask, 
            entity_adj_relation=entity_adj_relation,
            labels = labels, 
            entity_adj_relevant_relation_label = entity_adj_relevant_relation_label, 
            ent_is_ans_label=ent_is_ans,
            calculate_ans_loss=calculate_ans_loss,
            calculate_kg_loss=calculate_kg_loss,
            question_text=question_text, 
        )[0]

        return train_loss
    
    def evaluate_step(self, model, batch, dataloader):

        dataset = dataloader.dataset 

        (idx, labels, _, context_ids, context_mask, question_text, question_indices, question_mask, \
            ent_indices, ent_mask, ent_is_ans, entity_text, entity_adj, entity_adj_mask, \
                entity_adj_relation, entity_adj_relevant_relation_label, passage_entity_ids, passage_entity_mask) = batch
        
        mask_passages = self.mask_passages
        num_passages_after_mask = self.num_passages_after_mask 
        
        outputs = model.generate(
            input_ids=context_ids,
            attention_mask=context_mask,
            question_indices=question_indices,
            question_mask=question_mask,
            ent_indices=ent_indices,
            ent_mask=ent_mask,
            entity_text=entity_text, 
            entity_adj=entity_adj, 
            entity_adj_mask=entity_adj_mask, 
            entity_adj_relation=entity_adj_relation, 
            max_length=50,
            question_text=question_text, 
            mask_passages = mask_passages, 
            num_passages_after_mask = num_passages_after_mask, 
            passage_entity_ids = passage_entity_ids, 
            passage_entity_mask = passage_entity_mask
        )

        score_list = []
        for i, o in enumerate(outputs):
            ans = tokenizer.decode(o, skip_special_tokens=True)
            gold = dataset.get_example(idx[i])["answers"]
            score = src.evaluation.ems(ans, gold)
            score_list.append(score)

        return score_list
    
if __name__ == "__main__":

    options = Options()
    options.add_reader_options()
    options.add_optim_options()
    opt = options.parse()

    gpu_setup(opt.local_rank, opt.seed) 

    model_name = 't5-' + opt.model_size
    tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)

    tokenizer.add_special_tokens({'sep_token': '[SEP]', 'cls_token': '[CLS]'})
    tokenizer.add_tokens(["<e>", "</e>"])

    relation2id = pickle.load(open("/nfs/common/data/rebel_dataset/relation2id.pkl", "rb"))
    collator = FiDCollator(opt.text_maxlength, tokenizer=tokenizer, \
        relation2id=relation2id, answer_maxlength=opt.answer_maxlength, max_num_edge=opt.k)

    if not opt.test_only:

        train_dataset = TripleFiDDataset(opt.train_data, opt.n_context)
        val_dataset = TripleFiDDataset(opt.eval_data, opt.n_context)

        train_dataloader = get_dataloader(opt.local_rank, train_dataset, opt.per_gpu_batch_size, shuffle=True, collate_fn=collator)
        val_dataloader = get_dataloader(opt.local_rank, val_dataset, opt.per_gpu_batch_size, shuffle=False, collate_fn=collator)
        
    model = TripleKGFiDT5.from_pretrained(model_name, tokenizer=tokenizer, ent_dim=opt.ent_dim, k=opt.k, hop=opt.hop, alpha=opt.alpha, num_triples=opt.num_triples)
    model.resize_token_embeddings(len(tokenizer))
    relationid2name = pickle.load(open("/nfs/common/data/rebel_dataset/relationid2name.pkl", "rb"))
    relation_embedding = pickle.load(open(f"/nfs/common/data/rebel_dataset/relation_t5{opt.model_size}_embeddings.pkl", "rb"))
    model.relation_extraction_setup(
        relationid2name=relationid2name,
        relation_embedding=relation_embedding
    )

    dir_path = Path(opt.checkpoint_dir)/opt.name
    dir_path.mkdir(parents=True, exist_ok=True)
    options.print_options(opt)

    setup_logger(opt.local_rank, os.path.join(dir_path, "trainer.log"))
    
    trainer_params = {
        "model": model, 
        "local_rank": opt.local_rank, 
        "world_size": dist.get_world_size() if opt.local_rank >=0 else 1, 
        "default_root_dir": dir_path, 
        "learning_rate": opt.lr, 
        "weight_decay": opt.weight_decay, 
        "optim": "adamw", 
        "scheduler": opt.scheduler, 
        "warmup_steps": 0.2 * opt.total_steps, 
        "max_steps": opt.total_steps, 
        "accumulate_grad_batches": opt.accumulation_steps, 
        "gradient_clip_val": opt.clip, 
        "val_every_n_steps": opt.eval_freq, 
        "log_every_n_steps": 1, 
        "save_checkpoint_every_n_steps": opt.save_freq,
        "best_val_topk": 1, 
        "debug": not opt.use_wandb,
        "wandb_project": opt.wandb_project, 
        "wandb_name": opt.wandb_name,
        "pretrain": opt.pretrain, 
        "mask_passages": opt.mask_passages, 
        "num_passages_after_mask": opt.num_passages_after_mask
    }

    trainer = ReaderTrainer(**trainer_params)

    if not opt.test_only:
        trainer.train(train_dataloader, val_dataloader)

    if opt.local_rank <= 0:

        print("Loading test data from test_spacy.json ... ")
        test_file_name = os.path.basename(opt.eval_data).replace("dev", "test")
        test_data_path = os.path.join(os.path.dirname(opt.eval_data), test_file_name)

        test_dataset = TripleFiDDataset(test_data_path, opt.n_context)
        test_dataloader = get_dataloader(-1, test_dataset, opt.per_gpu_batch_size, shuffle=False, collate_fn=collator)

        for ckpt_path in glob.glob(os.path.join(dir_path, "best_val_*.ckpt")) + glob.glob(os.path.join(dir_path, "checkpoint_*.ckpt")):
            trainer.load_model_checkpoint(ckpt_path)
            metrics = trainer.evaluate(test_dataloader)
            logger.info(" ==== Test Results of {} ====".format(ckpt_path))
            logger.info("Exact Match: {:.2f}".format(100 * metrics["_main_eval_metric"]))
            logger.info("Avg Generate Time: {}".format(metrics.get("time", "NaN")))
            logger.info(" =======================================================")


    if opt.local_rank <=0 and opt.pretrain:
        ckpt_path = list(glob.glob(os.path.join(dir_path, "best_val_*.ckpt")))[0]
        new_ckpt_path = os.path.join(os.path.dirname(ckpt_path), "pretrain_best_val.ckpt")
        os.rename(ckpt_path, new_ckpt_path)

    if opt.local_rank >= 0:
        dist.barrier()

    if opt.local_rank >=0:
        cleanup()