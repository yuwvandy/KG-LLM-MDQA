import spacy
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import torch

nlp = spacy.load('en_core_web_lg')
stops = set(stopwords.words('english'))
symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
stemmer = PorterStemmer()


def strip_string(string, only_stopwords = False):
    if only_stopwords:
        return ' '.join([str(t) for t in nlp(string) if not t.is_stop])
    else:
        return ' '.join([str(t) for t in nlp(string) if t.pos_ in ['NOUN', 'PROPN']])


def preprocess(sentence):
    sentence = np.char.lower(sentence).tolist().split(' ')

    sentence = [stemmer.stem(_) for _ in sentence if _ not in stops and _ not in symbols and len(_) > 1]
    
    return strip_string(' '.join(sentence))


def load_saved(model, path, exact=True):
    try:
        state_dict = torch.load(path)
    except:
        state_dict = torch.load(path, map_location=torch.device('cpu'))
    
    def filter(x): return x[7:] if x.startswith('module.') else x
    if exact:
        state_dict = {filter(k): v for (k, v) in state_dict.items()}
    else:
        state_dict = {filter(k): v for (k, v) in state_dict.items() if filter(k) in model.state_dict()}
    model.load_state_dict(state_dict)
    
    return model


if __name__ == "__main__":
    print(preprocess('I am 123423!! a basketball-lover!'))




