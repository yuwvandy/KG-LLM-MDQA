from parse import parse_args
from utils import load_saved, move_to_cuda
import numpy as np

import torch
from transformers import (AutoConfig, AutoTokenizer)

from model import Retriever_inf
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import faiss


args = parse_args()
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
bert_config = AutoConfig.from_pretrained(args.model_name)

model = Retriever_inf(bert_config, args)
model = load_saved(model, './model/hotpotqa/model.pt', exact=False)
model.to(args.device)
model.eval()

c_emb = np.load('corpus_embs.npy')

#cpu search
index = faiss.IndexFlatIP(c_emb.shape[1])

#gpu search
# res = faiss.StandardGpuResources()
# index_flat = faiss.IndexFlatIP(c_emb.shape[1])
# index = faiss.index_cpu_to_gpu(res, 0, index_flat)


index.add(c_emb)
corpus = json.load(open('wiki_id2doc.json', 'r'))
corpus = [{'title': corpus[key]['title'], 'text': corpus[key]['text']} for key in corpus]

print('Finish_indexing')

app = Flask(__name__)
CORS(app)
torch.set_float32_matmul_precision("high")
@app.route('/flask', methods=['POST'])
def ask_ow():
    question = request.json['question']
    print(question)

    if question.endswith('?'):
        question = question[:-1]

    query_encode = tokenizer(text = question, max_length = args.max_q_len, return_tensors = 'pt', padding=True, truncation=True)
    for key in query_encode:
        query_encode[key] = query_encode[key].to(args.device)

    q_emb = model(query_encode['input_ids'], query_encode['attention_mask']).detach().cpu().numpy()

    sims, indices = index.search(q_emb, 5)

    context_1 = [corpus[idx]['text'] for idx in indices[0]]
    sims_1 = sims[0]

    #query 2-hop
    context_2 = []
    scores_2 = []
        
    query_encode = tokenizer(text = [question for _ in range(len(context_1))], text_pair = context_1, max_length = args.max_q_sp_len, return_tensors = 'pt', padding=True, truncation=True)
    for key in query_encode:
        query_encode[key] = query_encode[key].to(args.device)
        
    q_embs = model(query_encode['input_ids'], query_encode['attention_mask']).detach().cpu().numpy()

    sims_2, indices = index.search(q_embs, 3)
    whole_indices = set(indices.flatten().tolist())

    for i, row in enumerate(indices):
        for j, idx in enumerate(row):
            if idx not in whole_indices:
                whole_indices.add(idx)

            context_2.append([context_1[i], corpus[idx]['text']])

            scores_2.append(sims_1[i]*sims_2[i, j])
            
    #select the top 5 according to scores_2 in context_2
    scores_2 = np.array(scores_2)
    top_k = np.argsort(scores_2)[::-1][:min(3, len(scores_2))]
    res = []
    for idx in top_k:
        res.extend(context_2[idx])

    return jsonify({'context': res})


if __name__ == "__main__":
    app.run(host = '0.0.0.0', port = 5000, debug = True, use_reloader = True)

    

