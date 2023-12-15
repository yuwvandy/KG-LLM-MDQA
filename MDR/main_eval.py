import os

from parse import parse_args
from dataset import load_dataset_inf
from utils import seed_everything, load_saved
import numpy as np

import torch
from transformers import (AutoConfig, AutoTokenizer)

from model import Retriever_inf
from tqdm import tqdm
import json


@torch.no_grad()
def run(d, model, tokenizer, args):
    model.eval()

    res = []
    scores_accumu = []

    #query 1-hop
    question = d['question']
    if question.endswith('?'):
        question = question[:-1]

    
    titles = [title for title, _ in d['title_chunks']]
    chunks = [chunk for _, chunk in d['title_chunks']]

    chunks_encode = tokenizer(text = titles, text_pair = chunks, max_length = args.max_len, return_tensors = 'pt', padding=True, truncation=True)
    query_encode = tokenizer(text = question, max_length = args.max_q_len, return_tensors = 'pt', padding=True, truncation=True)

    for key in chunks_encode:
        chunks_encode[key] = chunks_encode[key].to(args.device)
    for key in query_encode:
        query_encode[key] = query_encode[key].to(args.device)

    
    q_emb = model(query_encode['input_ids'], query_encode['attention_mask'])
    c_emb = model(chunks_encode['input_ids'], chunks_encode['attention_mask'])
    #dot product to select top-k candiate from candidates

    scores = torch.matmul(q_emb, c_emb.transpose(0, 1))
    scores = scores.squeeze(0).cpu().numpy()
    top_k = np.argsort(scores)[::-1][:args.top_k]

    for idx in top_k:
        res.append([chunks[idx]])
        scores_accumu.append(scores[idx])

    #query 2-hop
    q_c1_encode = tokenizer(text = [question for _ in range(len(res))], text_pair = [_[0] for _ in res], max_length = args.max_q_sp_len, return_tensors = 'pt', padding=True, truncation=True)
    for key in q_c1_encode:
        q_c1_encode[key] = q_c1_encode[key].to(args.device)
    
    q_c1_emb = model(q_c1_encode['input_ids'], q_c1_encode['attention_mask'])
    scores = torch.matmul(q_c1_emb, c_emb.transpose(0, 1))
    scores = scores.squeeze(0).cpu().numpy()
    #note that here scores is a 2d matrix
    top_k = np.argsort(scores, axis = 1)[:, ::-1][:, :3]


    res2 = [_[0] + '\n' + chunks[idx] for i, _ in enumerate(res) for idx in top_k[i] if _[0] != chunks[idx]]

    
    return res2[:30]


if __name__ == "__main__":
    args = parse_args()
    args.path = os.getcwd()

    args.device = torch.device("cuda:3")
    n_gpu = torch.cuda.device_count()

    seed_everything(args.seed)
    data = load_dataset_inf(args.dataset) 

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    bert_config = AutoConfig.from_pretrained(args.model_name)

    model = Retriever_inf(bert_config, args)

    if args.dataset in ['HotpotQA', '2WikiMQA', 'IIRC']:
        model = load_saved(model, f'./model/HotpotQA/model.pt', exact=False)
    else:
        model = load_saved(model, './model/{}/model.pt'.format(args.dataset), exact=False)

    model.to(args.device)

    context = []
    for d in tqdm(data):
        context.append(run(d, model, tokenizer, args))
    
    with open('./Context/{}/mdr_context.json'.format(args.dataset), 'w') as f:
        json.dump(context, f)
    

