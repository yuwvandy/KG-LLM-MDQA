from collections import defaultdict
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import networkx as nx
import multiprocessing as mp
from utils import preprocess
import concurrent.futures
import requests
from sentence_transformers import SentenceTransformer
import spacy
import torch
from transformers import (AutoConfig, AutoTokenizer)
from mdr_encode import Retriever_inf, run
from utils import load_saved
import pickle as pkl
from sklearn.metrics.pairwise import cosine_similarity


nlp = spacy.load('en_core_web_lg')
encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

MY_GCUBE_TOKEN = '07e1bd33-c0f5-41b0-979b-4c9a859eec3f-843339462'

class WATAnnotation:
    # An entity annotated by WAT

    def __init__(self, d):

        # char offset (included)
        self.start = d['start']
        # char offset (not included)
        self.end = d['end']

        # annotation accuracy
        self.rho = d['rho']
        # spot-entity probability
        self.prior_prob = d['explanation']['prior_explanation']['entity_mention_probability']

        # annotated text
        self.spot = d['spot']

        # Wikpedia entity info
        self.wiki_id = d['id']
        self.wiki_title = d['title']


    def json_dict(self):
        # Simple dictionary representation
        return {'wiki_title': self.wiki_title,
                'wiki_id': self.wiki_id,
                'start': self.start,
                'end': self.end,
                'rho': self.rho,
                'prior_prob': self.prior_prob
                }


def wat_entity_linking(text):
    # Main method, text annotation with WAT entity linking system
    wat_url = 'https://wat.d4science.org/wat/tag/tag'
    payload = [("gcube-token", MY_GCUBE_TOKEN),
               ("text", text),
               ("lang", 'en'),
               ("tokenizer", "nlp4j"),
               ('debug', 9),
               ("method",
                "spotter:includeUserHint=true:includeNamedEntity=true:includeNounPhrase=true,prior:k=50,filter-valid,centroid:rescore=true,topk:k=5,voting:relatedness=lm,ranker:model=0046.model,confidence:model=pruner-wiki.linear")]

    response = requests.get(wat_url, params=payload)

    return [WATAnnotation(a) for a in response.json()['annotations']]


def wat_annotations(wat_annotations):
    json_list = [w.json_dict() for w in wat_annotations]
    
    return json_list


def wiki_kw_extract_chunk(chunk, prior_prob = 0.8):
    title, chunk = chunk

    wat_annotations = wat_entity_linking(chunk)
    json_list = [w.json_dict() for w in wat_annotations]
    kw2chunk = defaultdict(set)
    chunk2kw = defaultdict(set)
    
    for wiki in json_list:
        if wiki['wiki_title'] != '' and wiki['prior_prob'] > prior_prob:
            kw2chunk[wiki['wiki_title']].add(chunk)
            chunk2kw[chunk].add(wiki['wiki_title'])
    
    kw2chunk[title].add(chunk)
    chunk2kw[chunk].add(title)

    return kw2chunk, chunk2kw


def tagme_extract(data, num_processes, prior_prob):
    for d in tqdm(data):
        kw2chunk = defaultdict(set) #kw1 -> [chunk1, chunk2, ...]
        chunk2kw = defaultdict(set) #chunk -> [kw1, kw2, ...]

        with concurrent.futures.ProcessPoolExecutor() as executor:
            future_to_chunk = {executor.submit(wiki_kw_extract_chunk, chunk, prior_prob): chunk for chunk in d['title_chunks']}
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk, inv_chunk = future.result()
                for key, value in chunk.items():
                    kw2chunk[key].update(value)
                for key, value in inv_chunk.items():
                    chunk2kw[key].update(value)

        for key in kw2chunk:
            kw2chunk[key] = list(kw2chunk[key])

        for key in chunk2kw:
            chunk2kw[key] = list(chunk2kw[key])

        d['kw2chunk'] = kw2chunk
        d['chunk2kw'] = chunk2kw

    return data


def wiki_spacy_extract_chunk(d):
    nlp = spacy.load('en_core_web_lg')
    nlp.add_pipe("entityLinker", last=True)

    kw2chunk = defaultdict(set) #kw1 -> [chunk1, chunk2, ...]
    chunk2kw = defaultdict(set) #chunk -> [kw1, kw2, ...]

    for title, chunk in d['title_chunks']:
        doc = nlp(chunk)

        for entity in doc._.linkedEntities:
            entity = entity.get_span().text

            kw2chunk[entity].add(chunk)
            chunk2kw[chunk].add(entity)
        
        kw2chunk[title].add(chunk)
        chunk2kw[chunk].add(title)

    for key in kw2chunk:
        kw2chunk[key] = list(kw2chunk[key])

    for key in chunk2kw:
        chunk2kw[key] = list(chunk2kw[key])

    d['kw2chunk'] = kw2chunk
    d['chunk2kw'] = chunk2kw

    return d


def wiki_spacy_extract(data, num_processes):
    # partial assign parameter to process_d
    func = partial(wiki_spacy_extract_chunk)

    with Pool(num_processes) as p:
        data = list(tqdm(p.imap(func, data), total=len(data)))

    return data


def tfidf_kw_extract_chunk(d, n_kw, ngram_l, ngram_h):
    kw2chunk = defaultdict(set) #kw1 -> [chunk1, chunk2, ...]
    chunk2kw = defaultdict(set) #chunk -> [kw1, kw2, ...]

    chunks = []
    titles = set()
    for title, chunk in d['title_chunks']:
        chunks.append(preprocess(chunk))
        titles.add(title)


    tfidf_vectorizer = TfidfVectorizer(stop_words = 'english', ngram_range = (ngram_l, ngram_h))
    X = tfidf_vectorizer.fit_transform(chunks)
    term = tfidf_vectorizer.get_feature_names_out()
    score = X.todense()
    kws = list(set(list(term[(-score).argsort()[:, :n_kw]][0]) + list(titles)))

    # print(kws)
    vec = CountVectorizer(vocabulary = kws, binary=True, ngram_range = (ngram_l, ngram_h), token_pattern = '[a-zA-Z0-9$&+,:;=?@#|<>.^*()%!/-]+', lowercase=True)
    bow = vec.fit_transform(chunks).toarray()

    bow_tile = np.tile(bow, (bow.shape[0], 1))
    bow_repeat = np.repeat(bow, bow.shape[0], axis = 0)
    common_kw = (bow_tile * bow_repeat).reshape(bow.shape[0], bow.shape[0], -1)
    node1, node2, kw_id = common_kw.nonzero()

    for n1, n2, kw in zip(node1, node2, kw_id):
        if n1 != n2:
            kw2chunk[kws[kw]].add(d['title_chunks'][n1][1])
            kw2chunk[kws[kw]].add(d['title_chunks'][n2][1])

            chunk2kw[d['title_chunks'][n1][1]].add(kws[kw])
            chunk2kw[d['title_chunks'][n2][1]].add(kws[kw])

    for key in kw2chunk:
        kw2chunk[key] = list(kw2chunk[key])

    for key in chunk2kw:
        chunk2kw[key] = list(chunk2kw[key])

    d['kw2chunk'] = kw2chunk
    d['chunk2kw'] = chunk2kw

    return d


def tfidf_kw_extract(data, n_kw, ngram_l, ngram_h, num_processes):
    func = partial(tfidf_kw_extract_chunk, n_kw = n_kw, ngram_l = ngram_l, ngram_h = ngram_h)

    with Pool(num_processes) as p:
        data = list(tqdm(p.imap(func, data), total=len(data)))

    return data


def kw_graph_construct(i_d):
    idx, d = i_d

    G = nx.MultiGraph()

    chunk2id = {}
    for i, chunk in enumerate(d['title_chunks']):
        _, chunk = chunk

        G.add_node(i, chunk = chunk)
        chunk2id[chunk] = i
    
    for kw, chunks in d['kw2chunk'].items():
        for i in range(len(chunks)):
            for j in range(i+1, len(chunks)):
                G.add_edge(chunk2id[chunks[i]], chunk2id[chunks[j]], kw = kw)
    
    return idx, G


def kw_process_graph(docs):
    pool = mp.Pool(mp.cpu_count())
    graphs = [None] * len(docs)

    for idx, G in tqdm(pool.imap_unordered(kw_graph_construct, enumerate(docs)), total=len(docs)):
        graphs[idx] = G

    pool.close()
    pool.join()

    return graphs

def knn_embs(i_d):
    idx, d = i_d

    chunks = []
    
    for title, chunk in d['title_chunks']:
        chunks.append(preprocess(chunk))

    emb = encoder.encode(chunks, device = 'cuda:{}'.format(idx % torch.cuda.device_count()))

    return idx, emb

def knn_embs_construct(dataset, docs, num_processes):
    pool = mp.Pool(num_processes)
    embs = [None] * len(docs)

    for idx, emb in tqdm(pool.imap_unordered(knn_embs, enumerate(docs)), total=len(docs)):
        embs[idx] = emb

    pool.close()
    pool.join()

    pkl.dump(embs, open('./{}/knn_embs.pkl'.format(dataset), 'wb'))

def knn_graph(i_d, k_knn, embs, strategy = 'cos'):
    idx, d = i_d

    emb = embs[idx]

    #build a knn Graph
    if strategy == 'cos':
        sim = cosine_similarity(emb, emb)

    elif strategy == 'dp':
        sim = np.matmul(emb, emb.transpose(1, 0))

    #topk
    top_idx = np.argsort(-sim, axis = 1)[:, 1:k_knn+1]

    tail_nodes = np.arange(top_idx.shape[0]).repeat(k_knn)
    head_nodes = top_idx.reshape(-1)
    edges = [(node1, node2) for node1, node2 in zip(tail_nodes, head_nodes)]

    G = nx.DiGraph()
    G.add_edges_from(edges)

    return idx, G

def knn_graph_construct(dataset, docs, k_knn, num_processes, strategy = 'cos'):
    pool = mp.Pool(num_processes)
    graphs = [None] * len(docs)

    func = partial(knn_graph, k_knn = k_knn, embs = pkl.load(open('./{}/knn_embs.pkl'.format(dataset), 'rb')), strategy = strategy)

    for idx, G in tqdm(pool.imap_unordered(func, enumerate(docs)), total=len(docs)):
        graphs[idx] = G

    pool.close()
    pool.join()

    return graphs

def mdr_embs_construct(dataset, docs, args):
    tokenizer = AutoTokenizer.from_pretrained(args.mdr_model_name)
    bert_config = AutoConfig.from_pretrained(args.mdr_model_name)

    model = Retriever_inf(bert_config, args)

    model = load_saved(model, './{}/mdr_encoder.pt'.format(args.dataset), exact=False)

    model.to(args.device)

    embs = []
    for d in tqdm(docs):
        embs.append(run(d, model, tokenizer, args))

    pkl.dump(embs, open('./{}/mdr_embs.pkl'.format(dataset), 'wb'))
    
def knn_mdr_graph_construct(dataset, docs, k_knn, num_processes, strategy = 'dp'):
    pool = mp.Pool(num_processes)
    graphs = [None] * len(docs)

    func = partial(knn_graph, k_knn = k_knn, embs = pkl.load(open('./{}/mdr_embs.pkl'.format(dataset), 'rb')), strategy = strategy)

    for idx, G in tqdm(pool.imap_unordered(func, enumerate(docs)), total=len(docs)):
        graphs[idx] = G

    pool.close()
    pool.join()

    return graphs

