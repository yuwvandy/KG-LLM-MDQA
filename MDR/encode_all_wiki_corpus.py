from parse import parse_args
from utils import load_saved, move_to_cuda
import numpy as np

import torch
from transformers import (AutoConfig, AutoTokenizer)

from model import Retriever_inf
from tqdm import tqdm
import json
from parse import parse_args
from loader import Dataset_collate_corpus, Dataset_enc_corpus
from torch.utils.data import DataLoader

args = parse_args()
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
bert_config = AutoConfig.from_pretrained(args.model_name)

model = Retriever_inf(bert_config, args)
model = load_saved(model, './model/hotpotqa/model.pt', exact=False)
model.to(args.device)
model = torch.nn.DataParallel(model)
model.eval()

@torch.no_grad()
def encode_corpus(corpus):
    dataset = Dataset_enc_corpus(corpus, tokenizer, args)
    dataloader = DataLoader(dataset, batch_size = 1024, pin_memory = True, collate_fn = Dataset_collate_corpus, num_workers = args.num_workers, shuffle=False)

    embs = []
    for batch in tqdm(dataloader):
        batch = move_to_cuda(batch)

        embs.append(model(batch['c_enc_btz'], batch['c_mask']))
    
    return torch.cat(embs, dim = 0).detach().cpu().numpy()



if __name__ == "__main__":
    corpus = json.load(open('wiki_id2doc.json', 'r'))
    corpus = [{'title': corpus[key]['title'], 'text': corpus[key]['text']} for key in corpus]

    corpus_embs = encode_corpus(corpus)
    np.save('corpus_embs.npy', corpus_embs)

    

