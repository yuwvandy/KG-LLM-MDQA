from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from rank_bm25 import BM25Okapi
from utils import tf_idf, get_encoder, strip_string
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from itertools import chain
import torch
from torch_scatter import segment_csr
from Levenshtein import distance as levenshtein_distance
from utils import cal_local_llm_llama, cal_local_llm_t5
import time



class No_retriever(object):
    def __init__(self, k):
        self.k = k
    
    def retrieve(self, data):
        return []
    

class Golden_retriever(object):
    def __init__(self, k):
        self.k = k
    
    def retrieve(self, data):
        return [c for _, c in data['supports']]


class KNN_retrieval(object):
    def __init__(self, text_encoder, k, k_emb):
        self.encoder = get_encoder(text_encoder)
        self.k_emb = k_emb
        self.k = k
        
    def index(self, data):
        self.strip_chunks = data['strip_chunks']
        self.group_idx = np.array(data['group_idx'])
        self.chunks = data['chunks']
        self.np_chunks = np.array(self.chunks)
        self.chunk_embeds = data['chunk_embeds']
    
    def query_encode(self, query):
        return self.encoder.encode([strip_string(query)])
    
    def retrieve(self, data):
        self.index(data)
        
        query_embed = self.query_encode(data['question'])

        #Search by embedding similarity
        scores = cosine_similarity(query_embed, self.chunk_embeds).flatten()
        group_scores = segment_csr(torch.tensor(scores), torch.tensor(self.group_idx), reduce = 'max').numpy()
        topk = np.argsort(group_scores)[-self.k_emb:].tolist()

        #Search by text fuzzy matching
        dist = [levenshtein_distance(data['question'], chunk) for chunk in self.strip_chunks]
        idxs = np.argsort(dist)

        for idx in idxs:
            if idx not in topk:
                topk.append(idx)
                if len(topk) == self.k:
                    break

        return [data['title_chunks'][_][1] for _ in topk]
    

class MDR_retrieval(object):
    def __init__(self, corpus):
        self.corpus = corpus
    
    def retrieve(self, data, i):
        return self.corpus[i]


class DPR_retrieval(object):
    def __init__(self, corpus):
        self.corpus = corpus
    
    def retrieve(self, data, i):
        corpus = self.corpus[i]

        return [c[0] for c in corpus]


class TF_IDF_retriever(object):
    def __init__(self, k):
        self.vectorizer = TfidfVectorizer()
        self.k = k
    
    def retrieve(self, data):
        corpus = [c for _, c in data['title_chunks']]
        tfidf_matrix = self.vectorizer.fit_transform(corpus)
        query_emb = self.vectorizer.transform([data['question']])

        cosine_sim = cosine_similarity(query_emb, tfidf_matrix).flatten()

        return [corpus[idx] for idx in cosine_sim.argsort()[-self.k:][::-1]]


class BM25_retriever(object):
    def __init__(self, k):
        self.k = k
    
    def retrieve(self, data):
        corpus = [c for _, c in data['title_chunks']]
        self.bm25 = BM25Okapi([c.split(" ") for c in corpus])

        scores = self.bm25.get_scores(data['question'].split(" "))

        return [corpus[idx] for idx in scores.argsort()[-self.k:][::-1]]
    

class KG_retriever(object):
    def __init__(self, k):
        self.k = k
    
    def retrieve(self, data, G):
        corpus = [c for _, c in data['title_chunks']]
        candidates_idx = list(range(len(corpus)))

        seed = data['question']
        retrieve_idxs = []

        prev_length = 0
        count = 0
        retrieve_num = [10, 5, 5, 5, 3, 2, 2, 2, 2, 2, 2]
        while len(retrieve_idxs) < self.k:
            idxs = tf_idf(seed, candidates_idx, corpus, k = retrieve_num[count], visited = retrieve_idxs)
            retrieve_idxs.extend(idxs[:max(0, self.k - len(retrieve_idxs))])
            
            candidates_idx = set(chain(*[list(G.neighbors(node)) for node in idxs]))
            candidates_idx = list(candidates_idx.difference(retrieve_idxs))

            if len(retrieve_idxs) == prev_length:
                break
            else:
                prev_length = len(retrieve_idxs)
            
            count += 1

        return [corpus[idx] for idx in retrieve_idxs], None, None, None


class llm_retriever_LLaMA(object):
    def __init__(self, k, k_nei, port):
        self.k = k
        self.k_nei = k_nei
        self.port = port
    
    def retrieve(self, data):
        corpus = [c for _, c in data['title_chunks']]
        candidates_idx = list(range(len(corpus)))

        seed = data['question']
        contexts = []
        
        idxs = tf_idf(seed, candidates_idx, corpus, k = self.k//self.k_nei, visited = [])

        for idx in idxs:
            context = seed + '\n' + corpus[idx]

            next_reason = cal_local_llm_llama(context, self.port)

            next_contexts = tf_idf(next_reason, candidates_idx, corpus, k = self.k_nei, visited = [])

            if next_contexts != []:
                contexts.extend([corpus[idx] + '\n' + corpus[_] for _ in next_contexts if corpus[_] != corpus[idx]])
            else:
                contexts.append(corpus[idx])

        return contexts
    

class llm_retriever_T5(object):
    def __init__(self, k, k_nei, port):
        self.k = k
        self.k_nei = k_nei
        self.port = port
    
    def retrieve(self, data):
        corpus = [c for _, c in data['title_chunks']]
        candidates_idx = list(range(len(corpus)))

        seed = data['question']
        contexts = []
        
        idxs = tf_idf(seed, candidates_idx, corpus, k = self.k//self.k_nei, visited = [])

        cur_contexts = [seed + ' ' + corpus[_] for _ in idxs]
        next_reasons = cal_local_llm_t5(cur_contexts, self.port)

        for idx, next_reason in zip(idxs, next_reasons):
            next_contexts = tf_idf(next_reason, candidates_idx, corpus, k = self.k_nei, visited = [])

            if next_contexts != []:
                contexts.extend([corpus[idx] + '\n' + corpus[_] for _ in next_contexts if corpus[_] != corpus[idx]])
            else:
                contexts.append(corpus[idx])

        return contexts
    

class llm_retriever_KG_T5(object):
    def __init__(self, k, k_nei, port):
        self.k = k
        self.k_nei = k_nei
        self.port = port
    
    def retrieve(self, data, G):
        corpus = [c for _, c in data['title_chunks']]
        candidates_idx = list(range(len(corpus)))

        seed = data['question']
        contexts = []
        
        start = time.time()
        idxs = tf_idf(seed, candidates_idx, corpus, k = self.k//self.k_nei, visited = [])
        t1 = time.time() - start

        start = time.time()
        cur_contexts = [seed + ' ' + corpus[_] for _ in idxs]
        next_reasons = cal_local_llm_t5(cur_contexts, self.port)
        t2 = time.time() - start

        start = time.time()
        for idx, next_reason in zip(idxs, next_reasons):
            nei_candidates_idx = list(G.neighbors(idx))

            next_contexts = tf_idf(next_reason, nei_candidates_idx, corpus, k = self.k_nei, visited = [])

            if next_contexts != []:
                contexts.extend([corpus[idx] + '\n' + corpus[_] for _ in next_contexts if corpus[_] != corpus[idx]])
            else:
                contexts.append(corpus[idx])

        t3 = time.time() - start

        return contexts, t1, t2, t3
    

class llm_retriever_KG_LLaMA(object):
    def __init__(self, k, k_nei, port):
        self.k = k
        self.k_nei = k_nei
        self.port = port
    
    def retrieve(self, data, G):
        corpus = [c for _, c in data['title_chunks']]
        candidates_idx = list(range(len(corpus)))

        seed = data['question']
        contexts = []
        
        idxs = tf_idf(seed, candidates_idx, corpus, k = self.k//self.k_nei, visited = [])

        for idx in idxs:
            context = seed + '\n' + corpus[idx]

            next_reason = cal_local_llm_llama(context, self.port)

            nei_candidates_idx = list(G.neighbors(idx))

            next_contexts = tf_idf(next_reason, nei_candidates_idx, corpus, k = self.k_nei, visited = [])

            if next_contexts != []:
                contexts.extend([corpus[idx] + '\n' + corpus[_] for _ in next_contexts if corpus[_] != corpus[idx]])
            else:
                contexts.append(corpus[idx])

        return contexts
    


class llm_retriever_il(object):
    def __init__(self, k, llm):
        self.k = k
        self.llm = llm
    
    def retrieve(self, data):
        corpus = [c for _, c in data['title_chunks']]
        candidates_idx = list(range(len(corpus)))

        seed = data['question']
        retrieve_idxs = []

        while len(retrieve_idxs) < self.k:
            idxs = tf_idf(seed, candidates_idx, corpus, k = 5, visited = retrieve_idxs)
            retrieve_idxs.extend(idxs[:min(0, self.k - len(retrieve_idxs))])

            tmp = PromptTemplate(input_variables = [], template = prompt_il(corpus, retrieve_idxs, data['question']))
            qa_chain = LLMChain(llm = self.llm, prompt = tmp)

            seed = qa_chain.run({})
        
        return [corpus[idx] for idx in retrieve_idxs]
