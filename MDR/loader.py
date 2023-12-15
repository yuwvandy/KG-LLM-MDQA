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
        if d['type'] == 'comparison':
            random.shuffle(d['pos_paras'])
            start_para, bridge_para = d['pos_paras'][0], d['pos_paras'][1]
        else:
            for para in d['pos_paras']:
                if para['title'] != d['bridge']:
                    start_para = para
                else:
                    bridge_para = para
        
        if self.train:
            random.shuffle(d['neg_paras'])
        
        c1_enc = self.encode_chunk_pair(start_para['title'].strip(), start_para['text'].strip(), self.max_len)
        c2_enc = self.encode_chunk_pair(bridge_para['title'].strip(), bridge_para['text'].strip(), self.max_len)

        n1_enc = self.encode_chunk_pair(d['neg_paras'][0]['title'].strip(), d['neg_paras'][0]['text'].strip(), self.max_len)
        n2_enc = self.encode_chunk_pair(d['neg_paras'][1]['title'].strip(), d['neg_paras'][1]['text'].strip(), self.max_len)

        q_enc = self.encode_chunk(question, max_len = self.max_q_len)
        q_c1_enc = self.encode_chunk_pair(question, start_para['text'].strip(), self.max_q_sp_len)

        
        return {'q_enc': q_enc, 'q_c1_enc': q_c1_enc, \
                'c1_enc': c1_enc, 'c2_enc': c2_enc, \
                'n1_enc': n1_enc, 'n2_enc': n2_enc}

    def __len__(self):
        return len(self.data)
    


class Dataset_process2(Dataset):
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
        
        start_para, bridge_para = d['pos_paras'][0], d['pos_paras'][1]
        
        if self.train:
            random.shuffle(d['neg_paras'])
        
        c1_enc = self.encode_chunk_pair(start_para['title'].strip(), start_para['text'].strip(), self.max_len)
        c2_enc = self.encode_chunk_pair(bridge_para['title'].strip(), bridge_para['text'].strip(), self.max_len)

        n1_enc = self.encode_chunk_pair(d['neg_paras'][0]['title'].strip(), d['neg_paras'][0]['text'].strip(), self.max_len)
        n2_enc = self.encode_chunk_pair(d['neg_paras'][1]['title'].strip(), d['neg_paras'][1]['text'].strip(), self.max_len)

        q_enc = self.encode_chunk(question, max_len = self.max_q_len)
        q_c1_enc = self.encode_chunk_pair(question, start_para['text'].strip(), self.max_q_sp_len)

        
        return {'q_enc': q_enc, 'q_c1_enc': q_c1_enc, \
                'c1_enc': c1_enc, 'c2_enc': c2_enc, \
                'n1_enc': n1_enc, 'n2_enc': n2_enc}

    def __len__(self):
        return len(self.data)
    


   
class Dataset_enc_corpus(Dataset):
    def __init__(self, data, tokenizer, args):
        super().__init__()

        self.tokenizer = tokenizer
        self.max_len = args.max_len

        self.data = data

        self.args = args

    def encode_chunk_pair(self, chunk1, chunk2, max_len):
        #chunk: (title, chunk)
        return self.tokenizer(text = chunk1, text_pair = chunk2, max_length = max_len, return_tensors = 'pt', padding=True, truncation=True)
    
    def __getitem__(self, index):
        title, chunk = self.data[index]['title'], self.data[index]['text']

        c1_enc = self.encode_chunk_pair(title.strip(), chunk.strip(), self.max_len)
        
        return c1_enc

    def __len__(self):
        return len(self.data)
        


def Dataset_collate(samples):
    if len(samples) == 0:
        return {}
    
    batch = {
        'q_enc_btz': collate_tokens([s['q_enc']['input_ids'].view(-1) for s in samples], 0),
        'q_mask': collate_tokens([s['q_enc']['attention_mask'][0] for s in samples], 0),

        'q_c1_enc_btz': collate_tokens([s['q_c1_enc']['input_ids'].view(-1) for s in samples], 0),
        'q_c1_mask': collate_tokens([s['q_c1_enc']['attention_mask'][0] for s in samples], 0),

        'c1_enc_btz': collate_tokens([s['c1_enc']['input_ids'].view(-1) for s in samples], 0),
        'c1_mask': collate_tokens([s['c1_enc']['attention_mask'][0] for s in samples], 0),

        'c2_enc_btz': collate_tokens([s['c2_enc']['input_ids'].view(-1) for s in samples], 0),
        'c2_mask': collate_tokens([s['c2_enc']['attention_mask'][0] for s in samples], 0),

        'n1_enc_btz': collate_tokens([s['n1_enc']['input_ids'].view(-1) for s in samples], 0),
        'n1_mask': collate_tokens([s['n1_enc']['attention_mask'][0] for s in samples], 0),

        'n2_enc_btz': collate_tokens([s['n2_enc']['input_ids'].view(-1) for s in samples], 0),
        'n2_mask': collate_tokens([s['n2_enc']['attention_mask'][0] for s in samples], 0),
    }

    return batch




def Dataset_collate_corpus(samples):
    if len(samples) == 0:
        return {}
    
    batch = {
                'c_enc_btz': collate_tokens([s['input_ids'].view(-1) for s in samples], 0),
                'c_mask': collate_tokens([s['attention_mask'][0] for s in samples], 0),
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
        


