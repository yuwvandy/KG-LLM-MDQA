import torch
from multiprocessing import Pool
from utils import strip_string, window_encodings, get_encoder
import os
import json
import pickle as pkl
from tqdm import tqdm
from functools import partial
import multiprocessing as mp


def get_chunk_embs(i_d, stripping, encoder):
    _, data, gpu_idx = i_d

    group_idx = [0]
    data['chunks'] = [chunk[1] for chunk in data['title_chunks']]
    chunks = []
    strip_chunks = []

    for chunk in data['chunks']:
        if stripping:
            chunk = strip_string(chunk)

        strip_chunks.append(chunk)
        chunks.extend(window_encodings(chunk, window_size = 4, overlap = 2))
        group_idx.append(len(chunks))
    
    chunk_embeds = encoder.encode(chunks, device = 'cuda:{}'.format(gpu_idx))
    data['chunk_embeds'] = chunk_embeds
    data['group_idx'] = group_idx
    data['strip_chunks'] = strip_chunks

    return data


def worker(d, device, stripping, encoder):
    return get_chunk_embs(d, stripping, encoder, device)


def load_data(dataset, stripping, encoder):
    if os.path.exists('./dataset/{}/test_docs_emb.json'.format(dataset)):
        return json.load(open('./dataset/{}/test_docs_emb.json'.format(dataset), 'rb'))
    else:
        data = json.load(open('./dataset/{}/test_docs.json'.format(dataset), 'rb'))

        encoder = get_encoder(encoder)
        num_gpus = torch.cuda.device_count()  # Number of available GPUs

        func = partial(get_chunk_embs, stripping = stripping, encoder = encoder)

        data_with_index = [(i, d, i % num_gpus) for i, d in enumerate(data)]

        new_data = []
        with Pool(processes = 14) as p:
            for data in tqdm(p.imap_unordered(func, data_with_index), total = len(data_with_index)):
                new_data.append(data)


        pkl.dump(new_data, open('./dataset/{}/test_docs_emb.pkl'.format(dataset), 'wb'))

    return new_data

if __name__ == '__main__':
    mp.set_start_method('spawn') 

    for dataset in ['MuSiQue', '2WikiMQA', 'IIRC', 'HotpotQA']:
        stripping = True
        encoder = 'multi-qa-MiniLM-L6-cos-v1'

        data = load_data(dataset, stripping, encoder)