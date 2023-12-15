import json

def load_dataset(dataset):
    if dataset == 'HotpotQA':
        train_data = [json.loads(_) for _ in open('./data/HotpotQA/train_with_neg_v0.json').readlines() if len(json.loads(_)['neg_paras']) >= 2]
        val_data = [json.loads(_) for _ in open('./data/HotpotQA/val_with_neg_v0.json').readlines()]

    elif dataset == 'MuSiQue':
        train_data = json.load(open('./data/MuSiQue/train_with_neg_v0.json', 'r'))
        val_data = json.load(open('./data/MuSiQue/val_with_neg_v0.json', 'r'))

    return train_data, val_data


def load_dataset_inf(dataset):
    return json.load(open('./data/{}/test_docs.json'.format(dataset), 'r'))


if __name__ == "__main__":
    dataset = '2WikiMQA'
    
    load_dataset(dataset = dataset)
    
