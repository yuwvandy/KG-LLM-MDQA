from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import spacy
import requests
import json

nlp = spacy.load('en_core_web_lg')


def tf_idf(seed, candidates_idx, corpus, k, visited):
    vectorizer = TfidfVectorizer()
    
    try:
        tfidf_matrix = vectorizer.fit_transform([corpus[_] for _ in candidates_idx])

        query_emb = vectorizer.transform([seed])
        cosine_sim = cosine_similarity(query_emb, tfidf_matrix).flatten()
        idxs = cosine_sim.argsort()[::-1]

        tmp_idxs = []
        for idx in idxs:
            if candidates_idx[idx] not in visited:
                tmp_idxs.append(candidates_idx[idx])
            
            k -= 1

            if k == 0:
                break

        return tmp_idxs

    except Exception as e:
        return []


def tf_idf2(question, corpus, corpus_idx, k):
    sub_corpus = [corpus[_] for _ in corpus_idx]
    vectorizer = TfidfVectorizer()
    
    try:
        tfidf_matrix = vectorizer.fit_transform(sub_corpus)

        query_emb = vectorizer.transform([question])
        cosine_sim = cosine_similarity(query_emb, tfidf_matrix).flatten()
        idxs = cosine_sim.argsort()[::-1][:k]

        return [sub_corpus[_] for _ in idxs]
    
    except Exception as e:
        return []


def get_encoder(encoder_type):
    return SentenceTransformer(encoder_type)


def strip_string(string, only_stopwords = False):
    if only_stopwords:
        return ' '.join([str(t) for t in nlp(string) if not t.is_stop])
    else:
        return ' '.join([str(t) for t in nlp(string) if t.pos_ in ['NOUN', 'PROPN']])
    

def window_encodings(sentence, window_size, overlap):
    """Compute encodings for a string by splitting it into windows of size window_size with overlap"""
    tokens = sentence.split()

    if len(tokens) <= window_size:
        return [sentence]
    
    return [' '.join(tokens[i:i + window_size]) for i in range(0, len(tokens) - window_size, overlap)]


def cal_local_llm_llama(input, port):
    # Define the url of the API
    url = "http://localhost:{}/api/ask".format(port)

    # Define the headers for the request
    headers = {
        'Content-Type': 'application/json',
    }

    # Define the data to be sent in the POST request
    # Replace with actual instruction and input data
    data = {
        'instruction': 'What evidence do we need to answer the question given the current evidence?',
        'input': input
        }

    # print(data)
    # Convert the data to JSON format
    data_json = json.dumps(data)

    # Make the POST request
    response = requests.post(url, headers=headers, data=data_json)

    # Get the json response
    response_json = response.json()

    return response_json['answer']


def cal_local_llm_t5(input, port):
    # Define the url of the API
    url = "http://localhost:{}/api/ask".format(port)

    # Define the headers for the request
    headers = {
        'Content-Type': 'application/json',
    }

    # Define the data to be sent in the POST request
    # Replace with actual instruction and input data
    data = {
        'source_text': input
        }

    # Convert the data to JSON format
    data_json = json.dumps(data)

    # Make the POST request
    response = requests.post(url, headers=headers, data=data_json)

    # Get the json response
    response_json = response.json()

    return response_json['answer']