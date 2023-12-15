import json
import pickle as pkl
import os
from parse import parse_args
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import chain
import numpy as np
import networkx as nx

class tf_idf_retriever(object):
    def __init__(self, k):
        self.vectorizer = TfidfVectorizer()
        self.k = k
    
    def retrieve(self, data):
        corpus = [c for _, c in data['title_chunks']]
        tfidf_matrix = self.vectorizer.fit_transform(corpus)
        query_emb = self.vectorizer.transform([data['question']])

        cosine_sim = cosine_similarity(query_emb, tfidf_matrix).flatten()

        return cosine_sim.argsort()[-self.k:][::-1].tolist()

#evaluate on Multi-Graph
def multiG_ratio_nei_overlap(retriever, d, G):
    corpus = [c for _, c in d['title_chunks']]

    idx_0 = retriever.retrieve(d)

    idx_1 = list(chain(*[list(G.neighbors(node)) for node in idx_0])) + idx_0
    idx_2 = list(chain(*[list(G.neighbors(node)) for node in idx_1])) + idx_1

    corpus_0 = set([corpus[i] for i in idx_0])
    corpus_1 = set([corpus[i] for i in idx_1])
    corpus_2 = set([corpus[i] for i in idx_2])

    set_s = set([_[1] for _ in d['supports']])


    recall, precision, hit, num_nei = [], [], [], []
    for corpus_i in [corpus_0, corpus_1, corpus_2]:
        recall.append(len(corpus_i.intersection(set_s)) / len(set_s))
        precision.append(len(corpus_i.intersection(set_s)) / len(corpus_i))
        hit.append((len(corpus_i.intersection(set_s)) == len(set_s))*1.0)
        num_nei.append(len(corpus_i))

    density = nx.density(G)

    return recall, precision, hit, num_nei, density

#evaluate on Di-Graph
def diG_ratio_nei_overlap(retriever, d, G):
    corpus = [c for _, c in d['title_chunks']]

    idx_0 = retriever.retrieve(d)

    idx_1 = list(chain(*[list(G.successors(node)) for node in idx_0])) + idx_0
    idx_2 = list(chain(*[list(G.successors(node)) for node in idx_1])) + idx_1

    corpus_0 = set([corpus[i] for i in idx_0])
    corpus_1 = set([corpus[i] for i in idx_1])
    corpus_2 = set([corpus[i] for i in idx_2])

    set_s = set([_[1] for _ in d['supports']])

    recall, precision, hit, num_nei = [], [], [], []
    for corpus_i in [corpus_0, corpus_1, corpus_2]:
        recall.append(len(corpus_i.intersection(set_s)) / len(set_s))
        precision.append(len(corpus_i.intersection(set_s)) / len(corpus_i))
        hit.append((len(corpus_i.intersection(set_s)) == len(set_s))*1.0)
        num_nei.append(len(corpus_i))

    density = nx.density(G)

    return recall, precision, hit, num_nei, density


if __name__ == '__main__':
    args = parse_args()
    args.path = os.getcwd()

    if args.kg == 'TAGME':
        info = 'TAGME_{}'.format(args.prior_prob)
        func = multiG_ratio_nei_overlap
    elif args.kg == 'wiki_spacy':
        info = 'wiki_spacy'
        func = multiG_ratio_nei_overlap
    elif args.kg == 'TF-IDF':
        info = 'TF-IDF_{}_{}_{}'.format(args.n_kw, args.min_n, args.max_n)
        func = multiG_ratio_nei_overlap
    elif args.kg == 'KNN':
        info = 'KNN_{}'.format(args.k_knn)
        func = diG_ratio_nei_overlap
    elif args.kg == 'MDR-KNN':
        info = 'MDR-KNN_{}'.format(args.k_knn)
        func = diG_ratio_nei_overlap
        

    data = json.load(open('./{}/test_docs.json'.format(args.dataset), 'r'))
    Gs = pkl.load(open('./{}/graph_{}.pkl'.format(args.dataset, info), 'rb'))

    data_idx = [(i, d, Gs[i]) for i, d in enumerate(data[:200])]

    retriever = tf_idf_retriever(5)

    recalls, precisions, hits, num_neis, densitys = [], [], [], [], []
    for i, d, G in data_idx:
        recall, precision, hit, num_nei, density = func(retriever, d, G)

        recalls.append(recall)
        precisions.append(precision)
        hits.append(hit)
        num_neis.append(num_nei)
        densitys.append(density)
        
    print('===================')
    print('Recall:', np.array(recalls).mean(axis = 0))
    print('Precision:', np.array(precisions).mean(axis = 0))
    print('SP_EM:', np.array(hits).mean(axis = 0))
    print('Nei_Ratios:', np.array(num_neis).mean(axis = 0)/len(G.nodes()))
    print('Density:', np.array(densitys).mean())
    print('====================')