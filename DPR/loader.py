from torch.utils.data import Dataset
import random

                
class Dataset_process(Dataset):
    def __init__(self, data, tokenizer, args, train):
        super().__init__()

        self.tokenizer = tokenizer
        self.max_len = args.max_len
        self.max_q_len = args.max_q_len
        self.max_q_sp_len = args.max_q_sp_len

        self.data = data

        self.args = args
        self.train = train

    def encode_chunk_pair(self, chunk1, chunk2, max_len):
        #chunk: (title, chunk)
        return self.tokenizer(text = chunk1, text_pair = chunk2, max_length = max_len, return_tensors = 'pt', padding=True, truncation=True)
    
    def encode_chunk(self, chunk, max_len):
        #chunk: (title, chunk)
        return self.tokenizer(text = chunk, max_length = max_len, return_tensors = 'pt', padding=True, truncation=True)

    def __getitem__(self, index):
        d = self.data[index]

        question = d['question']
        if question.endswith('?'):
            question = question[:-1]

        random.shuffle(d['pos_paras'])
        random.shuffle(d['neg_paras'])
        
        pos_para = d['pos_paras'][0]
        neg_para = d['neg_paras'][0]

        c_enc = self.encode_chunk_pair(pos_para['title'].strip(), pos_para['text'].strip(), self.max_len)
        n_enc = self.encode_chunk_pair(neg_para['title'].strip(), neg_para['text'].strip(), self.max_len)
        q_enc = self.encode_chunk(question, max_len = self.max_q_len)

        
        return {'q_enc': q_enc, 'c_enc': c_enc, 'n_enc': n_enc}

    def __len__(self):
        return len(self.data)
    


class Dataset_process2(Dataset):
    def __init__(self, data, chunk_pool, tokenizer, args, train):
        super().__init__()

        self.tokenizer = tokenizer
        self.max_len = args.max_len
        self.max_q_len = args.max_q_len
        self.max_q_sp_len = args.max_q_sp_len
        self.chunk_pool = chunk_pool

        self.data = data

        self.args = args
        self.train = train

    def encode_chunk_pair(self, chunk1, chunk2, max_len):
        #chunk: (title, chunk)
        return self.tokenizer(text = chunk1, text_pair = chunk2, max_length = max_len, return_tensors = 'pt', padding=True, truncation=True)
    
    def encode_chunk(self, chunk, max_len):
        #chunk: (title, chunk)
        return self.tokenizer(text = chunk, max_length = max_len, return_tensors = 'pt', padding=True, truncation=True)

    def __getitem__(self, index):
        d = self.data[index]

        question = d['question']
        if question.endswith('?'):
            question = question[:-1]
        
        pos_para = random.sample(d['supports'], 1)[0]
        neg_para = random.sample(self.chunk_pool, 1)[0]

        c_enc = self.encode_chunk_pair(pos_para[0].strip(), pos_para[1].strip(), self.max_len)
        n_enc = self.encode_chunk_pair(neg_para[0].strip(), neg_para[1].strip(), self.max_len)
        q_enc = self.encode_chunk(question, max_len = self.max_q_len)

        
        return {'q_enc': q_enc, 'c_enc': c_enc, 'n_enc': n_enc}

    def __len__(self):
        return len(self.data)
        


def Dataset_collate(samples):
    if len(samples) == 0:
        return {}
    
    batch = {
        'q_enc_btz': collate_tokens([s['q_enc']['input_ids'].view(-1) for s in samples], 0),
        'q_mask': collate_tokens([s['q_enc']['attention_mask'][0] for s in samples], 0),
        
        'c_enc_btz': collate_tokens([s['c_enc']['input_ids'].view(-1) for s in samples], 0),
        'c_mask': collate_tokens([s['c_enc']['attention_mask'][0] for s in samples], 0),
        
        'n_enc_btz': collate_tokens([s['n_enc']['input_ids'].view(-1) for s in samples], 0),
        'n_mask': collate_tokens([s['n_enc']['attention_mask'][0] for s in samples], 0),
    }

    return batch




def collate_tokens(values, pad_idx, eos_idx=None, left_pad=False, move_eos_to_beginning=False):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    if len(values[0].size()) > 1:
        values = [v.view(-1) for v in values]
    size = max(v.size(0) for v in values)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            assert src[-1] == eos_idx
            dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res
        


