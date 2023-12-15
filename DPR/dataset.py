import json
import random
import os

def load_dataset(dataset):
    if dataset == 'HotpotQA':
        train_data = [json.loads(_) for _ in open('./train_data/HotpotQA/train_with_neg_v0.json').readlines() if len(json.loads(_)['neg_paras']) >= 2]
        val_data = [json.loads(_) for _ in open('./train_data/HotpotQA/dev_with_neg_v0.json').readlines()]

        chunk_pool = []
    
    elif dataset in ['2WikiMQA', 'IIRC', 'MuSiQue']:
        train_data = json.load(open('./train_data/{}/train.json'.format(dataset), 'r'))
        val_data = json.load(open('./train_data/{}/val.json'.format(dataset), 'r'))

        chunks = json.load(open('./train_data/{}/chunks.json'.format(dataset), 'r'))
        titles = json.load(open('./train_data/{}/titles.json'.format(dataset), 'r'))

        chunk_pool = [(chunk, title) for chunk, title in zip(chunks, titles)]

    return train_data, val_data, chunk_pool


def load_dataset_inf(dataset):
    return json.load(open('./QA-data/{}/test_docs.json'.format(dataset), 'r'))


if __name__ == "__main__":
    dataset = 'wikimultihop'
    
    load_dataset(dataset = dataset)
    
