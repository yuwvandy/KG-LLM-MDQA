import sys
from pathlib import Path

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import torch
from core.tokenizer import Tokenizer
from tqdm import tqdm
from typing import List
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer


def preprocess(string):
    stemmer = PorterStemmer()

    words = np.char.lower(string).tolist().split()

    new_words = ''
    stop_words = stopwords.words('english')
    for word in words:
        if word not in stop_words:
            new_words += ' ' + word
    
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in symbols:
        new_words = np.char.replace(new_words, i, ' ').tolist()
    
    np.char.replace(new_words, "'", "")

    new_text = ""
    for word in new_words.split():
        if len(word) > 1:
            new_text += ' ' + stemmer.stem(word)
    
    return new_text

def kw_extract(nei_data):
    nei_data = [preprocess(_) for _ in nei_data]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(nei_data)
    kws = vectorizer.get_feature_names_out()

    score = X.todense()
    kws = list(kws[(-score).argsort()[:, :50]][0])

    return kws
    

def transform_pure(data: List, nei_data: List, ini_sent_len: int, n_nei: int, idxs: list):
    """Transforms the data into a list of instruction-tuning samples."""


    return [{"instruction": 'Complete the [Text]', \
             "input": '[Text]: ' + ' '.join(d.split()[:ini_sent_len]), \
             "output": ' '.join(d.split()[ini_sent_len:]),\
             "idx": idx} for idx, d in zip(idxs, data)]

def transform_topo(data: List, nei_data: List, ini_sent_len: int, n_nei: int, idxs: list):
    """Transforms the data into a list of instruction-tuning samples."""
    output =[]
    instruction = 'Refer to the relevant [Contexts] and Complete the [Text]'

    for i, d in enumerate(data):
        output.append({"instruction": instruction, \
                     "input": '[Contexts]: ' + '\n\n'.join(nei_data[i][:n_nei]) + '\n\n[Text]: ' + ' '.join(d.split()[:ini_sent_len]), \
                     "output": ' '.join(d.split()[ini_sent_len:]),\
                     "idx": idxs[i]})
    
    return output

# def transform_topo_kw(data: List, nei_data: List, ini_sent_len: int, n_nei: int, idxs: list):
#     """Transforms the data into a list of instruction-tuning samples."""
#     output =[]
#     instruction = 'Refer to the relevant [Keywords] and Complete the [Text]'

#     for i, d in enumerate(data):
#         keywords = kw_extract(nei_data[i]) if nei_data[i] else []

#         # print(keywords)
#         output.append({"instruction": instruction, \
#                      "input": '[Keywords]: ' + ' '.join(keywords) + '\n\n[Text]: ' + ' '.join(d.split()[:-ini_sent_len]), \
#                      "output": ' '.join(d.split()[-ini_sent_len:]),\
#                      "idx": idxs[i]})
    
#     #keyword extraction
#     return output


def prepare_sample(example: dict, tokenizer: Tokenizer, max_length: int, mask_inputs: bool = True, ignore_index: int = -1):
    """Processes a single sample.
    
    Each sample in the dataset consists of:
    - instruction: A string describing the task
    - input: A string holding a special input value for the instruction.
        This only applies to some samples, and in others this is empty.
    - output: The response string

    This function processes this data to produce a prompt text and a label for
    supervised training. The input text is formed as a single message including all
    the instruction, the input (optional) and the response.
    The label/target is the same message but can optionally have the instruction + input text
    masked out (mask_inputs=True).

    Finally, both the prompt and the label get tokenized. If desired, all tokens
    in the label that correspond to the original input prompt get masked out (default).
    """
    # print(example)
    full_prompt = generate_prompt(example)
    full_prompt_and_response = full_prompt + example["output"]
    encoded_full_prompt = tokenize(tokenizer, full_prompt, max_length=max_length, eos=False)
    encoded_full_prompt_and_response = tokenize(tokenizer, full_prompt_and_response, eos=True, max_length=max_length)

    # The labels are the full prompt with response, but with the prompt masked out
    labels = encoded_full_prompt_and_response.clone()
    if mask_inputs:
        labels[:len(encoded_full_prompt)] = ignore_index

    return {**example, \
            "input_ids": encoded_full_prompt_and_response, \
            "input_ids_no_response": encoded_full_prompt, \
            "labels": labels}


def tokenize(tokenizer: Tokenizer, string: str, max_length: int, eos = True) -> torch.Tensor:
    return tokenizer.encode(string, bos = True, eos = eos, max_length = max_length)


def generate_prompt(example):
    """Generates a standardized message to prompt the model with an Instruction, optional Input and a Response field."""

    return (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:"
    )


def prepare(destination_path: Path, 
            tokenizer_path: Path, 
            max_seq_length: int, 
            mask_inputs: bool, 
            data_file_name: str, 
            ini_sent_len: int, 
            n_nei: int, 
            task: str):
    """Prepare the Collaborative-generation dataset for instruction tuning.
    
    The output is a training, validation, testing dataset saved as `train.pt, val.pt, test.pt`,
    which stores the preprocessed and tokenized prompts and labels.
    """
    
    destination_path.mkdir(parents=True, exist_ok=True)
    file_path = destination_path / data_file_name

    tokenizer = Tokenizer(tokenizer_path)
    
    data = torch.load(file_path)

    train_nei = [data.train_edge_index[1][data.train_edge_index[0] == _].tolist() for _ in data.train_ids]
    val_nei = [data.val_edge_index[1][data.val_edge_index[0] == _].tolist() for _ in data.val_ids]
    test_nei = [data.edge_index[1][data.edge_index[0] == _].tolist() for _ in data.test_ids]

    if task == 'none':
        transform = transform_pure
    elif task == 'topo_kw':
        transform = transform_topo_kw
    elif task == 'topo':
        transform = transform_topo
    
    train_nei_texts = [[data.raw_text[_] for _ in nei] for nei in train_nei]
    val_nei_texts = [[data.raw_text[_] for _ in nei] for nei in val_nei]
    test_nei_texts = [[data.raw_text[_] for _ in nei] for nei in test_nei]

    train_data = transform([data.raw_text[_] for _ in data.train_ids], train_nei_texts, ini_sent_len, n_nei, data.train_ids.tolist())
    val_data = transform([data.raw_text[_] for _ in data.val_ids], val_nei_texts, ini_sent_len, n_nei, data.val_ids.tolist())
    test_data = transform([data.raw_text[_] for _ in data.test_ids], test_nei_texts, ini_sent_len, n_nei, data.test_ids.tolist())

    print(f"train has {len(train_data):,} samples")
    print(f"val has {len(val_data):,} samples")
    print(f"test has {len(test_data):,} samples")

    print("Processing train split ...")
    train_data = [prepare_sample(sample, tokenizer, max_seq_length, mask_inputs) for sample in tqdm(train_data)]
    torch.save(train_data, file_path.parent / "train_{}.pt".format(task))

    print("Processing val split ...")
    val_data = [prepare_sample(sample, tokenizer, max_seq_length, mask_inputs) for sample in tqdm(val_data)]
    torch.save(val_data, file_path.parent / "val_{}.pt".format(task))

    print("Processing test split ...")
    test_data = [prepare_sample(sample, tokenizer, max_seq_length, mask_inputs) for sample in tqdm(test_data)]
    torch.save(test_data, file_path.parent / "test_{}.pt".format(task))


if __name__ == "__main__":
    data = 'cora'
    prepare(Path(f"data/{data}"), Path("checkpoints/lit-llama2/tokenizer.model"), 256, False, f"{data}.pt", 25, 3, 'none')
    
    prepare(Path(f"data/{data}"), Path("checkpoints/lit-llama2/tokenizer.model"), int(256*3), False, f"{data}.pt", 25, 3, 'topo')

    # print(kw_extract(nei_data = ["I am a \' boy at Vanderbilt", "Alice plays dancing at Harvard"]))