import json
import os
import time
from typing import Iterable
import numpy as np

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from trlx.data.ppo_types import PPORLBatch, PPORLElement, GLMPPORLBatch, GLMPPORLElement
from trlx.pipeline import BaseRolloutStore

import requests
from fast_bleu import BLEU
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

import time

class MyRolloutStorage:
    '''
    rollout storage for calucalting diversity reward
    '''
    def __init__(self):
        self.history_str = []
        self.history_embedding = []
        self.chencherry = SmoothingFunction()
        
    def get_embedding(self, query):
        url = 'http://localhost:9576/get_embedding'
        resp = requests.post(url, data=json.dumps(query))
        embedding = resp.json()
        return embedding
        
    def push(self, exps):
        self.history_str += [i.split() for i in exps] # 分好词             
        self.history_embedding += self.get_embedding(exps)
        
    def get_selfbleu(self, query_list, max_size=10000):
        query_list = [i.split() for i in query_list]
        weights = {5: (0, 1./4, 1./4, 1./4, 1./4)}
        if max_size > len(self.history_str):
            candidates = self.history_str[-max_size:]
        else:
            candidates = self.history_str
        bleu = BLEU(candidates, weights=weights)
        scores = bleu.get_score(query_list)[5]
        # print(scores)
        scores = [1-i for i in scores]
        assert len(scores) == len(query_list)
        return scores
        
    def _get_selfbleu(self, query, max_size=10000):
        '''
        nltk计算
        '''
        # 返回1-selfbleu
        query = query.split()
        try:
            assert len(query) > 1
        except:
            print('error query = ', query)
            return 1
        max_n = min(5, len(query))
        min_n = 0 if max_n == 1 else 1
        weights = ()
        for i in range(min_n):
            weights += (0,)
        for i in range(min_n, max_n):
            weights += (1/(max_n-min_n),)
        # print(query)
        # print('max n = ', max_n)
        # print('weights = ', weights)
        
        if max_size > len(self.history_str):
            candidates = self.history_str[-max_size:]
        else:
            candidates = self.history_str
        
        bleu = sentence_bleu(candidates, query, weights=weights, smoothing_function=self.chencherry.method1)
        return 1 - bleu
    
    def get_cosine_sim(self, query_list, max_size=10000):
        # 返回1-cosine similarity
        query_embedding = np.array(self.get_embedding(query_list))
        if max_size > len(self.history_embedding):
            candidates = self.history_embedding[-max_size:]
        else:
            candidates = self.history_embedding
        # query_embedding = self.get_embedding(query)
        cos_sim = (np.dot(query_embedding, np.array(candidates).T) / (np.linalg.norm(query_embedding, axis=-1)[..., np.newaxis] * (np.linalg.norm(candidates, axis=-1)[np.newaxis,...]))).mean(axis=-1)
        assert len(cos_sim) == len(query_list)
        return 1 - cos_sim
        
    def get_reward(self, query_list):
        '''
        根据query和存储的history计算reward
        '''
        
        if not any(self.history_str):
            return torch.tensor([1] * len(query_list)), torch.tensor([1] * len(query_list))
        
        else:
            # 计算self-bleu
            start = time.time()
            selfbleu_rewards = self.get_selfbleu(query_list, max_size=10000)
            # selfbleu_rewards = []
            # for query in query_list:
                # selfbleu_rewards.append(self.get_selfbleu(query, max_size=10000))
            end = time.time()
            print(f"bleu time = ", end - start)
            
            # 计算cosine embedding similarity
            cosine_similarity = self.get_cosine_sim(query_list, max_size=10000)
            # cosine_similarity = []
            # for query in query_list:
                # cosine_similarity.append(self.get_cosine_sim(query, max_size=10000))
            end2 = time.time()
            print(f"cos time = ", end2 - end)
            
            return torch.tensor(selfbleu_rewards), torch.tensor(cosine_similarity)

class PPORolloutStorage(BaseRolloutStore):
    """
    Rollout storage for training PPO
    """

    def __init__(self, pad_token_id):
        super().__init__()

        self.pad_token_id = pad_token_id
        self.history: Iterable[PPORLElement] = [None]

    def push(self, exps: Iterable[PPORLElement]):
        self.history += exps

    def clear_history(self):
        self.history = []

    def export_history(self, location: str):
        assert os.path.exists(location)

        fpath = os.path.join(location, f"epoch-{str(time.time())}.json")

        def exp_to_dict(exp):
            {k: v.cpu().tolist() for k, v in exp.__dict__.items()}

        data = [exp_to_dict(exp) for exp in self.history]
        with open(fpath, "w") as f:
            f.write(json.dumps(data, indent=2))

    def __getitem__(self, index: int) -> PPORLElement:
        return self.history[index]

    def __len__(self) -> int:
        return len(self.history)

    def create_loader(
        self,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        def collate_fn(elems: Iterable[PPORLElement]):
            return PPORLBatch(
                # Left padding of already left-padded queries
                pad_sequence(
                    [elem.query_tensor.flip(0) for elem in elems],
                    padding_value=self.pad_token_id,
                    batch_first=True,
                ).flip(1),
                # Right pad the rest, to have a single horizontal query/response split
                pad_sequence(
                    [elem.response_tensor for elem in elems],
                    padding_value=self.pad_token_id,
                    batch_first=True,
                ),
                pad_sequence(
                    [elem.logprobs for elem in elems],
                    padding_value=0.0,
                    batch_first=True,
                ),
                pad_sequence([elem.values for elem in elems], padding_value=0.0, batch_first=True),
                pad_sequence(
                    [elem.rewards for elem in elems],
                    padding_value=0.0,
                    batch_first=True,
                ),
            )

        return DataLoader(self, batch_size, shuffle=shuffle, collate_fn=collate_fn)



class GLMPPORolloutStorage(BaseRolloutStore):
    """
    Rollout storage for training PPO
    """

    def __init__(self, pad_token_id):
        super().__init__()

        self.pad_token_id = pad_token_id
        self.history: Iterable[GLMPPORLElement] = [None]

    def push(self, exps: Iterable[GLMPPORLElement]):
        self.history += exps

    def clear_history(self):
        self.history = []

    def export_history(self, location: str):
        assert os.path.exists(location)

        fpath = os.path.join(location, f"epoch-{str(time.time())}.json")

        def exp_to_dict(exp):
            {k: v.cpu().tolist() for k, v in exp.__dict__.items()}

        data = [exp_to_dict(exp) for exp in self.history]
        with open(fpath, "w") as f:
            f.write(json.dumps(data, indent=2))

    def __getitem__(self, index: int) -> GLMPPORLElement:
        return self.history[index]

    def __len__(self) -> int:
        return len(self.history)

    def create_loader(
        self,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        def collate_fn(elems: Iterable[GLMPPORLElement]):
            return GLMPPORLBatch(
                [elem.query for elem in elems],
                [elem.response for elem in elems],
                # right padding
                pad_sequence(
                    [elem.response_tensor for elem in elems],
                    padding_value=self.pad_token_id,
                    batch_first=True,
                ),
                pad_sequence(
                    [elem.logprobs for elem in elems],
                    padding_value=0.0,
                    batch_first=True,
                ),
                pad_sequence([elem.values for elem in elems], padding_value=0.0, batch_first=True),
                pad_sequence(
                    [elem.rewards for elem in elems],
                    padding_value=0.0,
                    batch_first=True,
                ),
            )

        return DataLoader(self, batch_size, shuffle=shuffle, collate_fn=collate_fn)
