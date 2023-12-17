from parse import parse_args
from tqdm import tqdm
import os
from functools import partial
from multiprocessing import Pool
import json

from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.llms import OpenAI
import nltk
from prompt import prompt_qac_wiki, prompt_qa_wiki, prompt_eval
from retriever import *
import pickle as pkl

os.environ["OPENAI_API_KEY"] = "Your API KEY HERE"

def run(i_d, retriever, prompt_qa, prompt_eval, llm, args):
    if args.retriever in ['KGP w/o LLM', 'KGP-T5', 'KGP-LLaMA']:
        i, d, G = i_d
        corpus, t1, t2, t3 = retriever.retrieve(d, G)
    else:
        i, d = i_d

        if args.retriever in ['MDR', 'DPR']:
            corpus = retriever.retrieve(d, i)
            t1, t2, t3 = None, None, None
        else:
            corpus = retriever.retrieve(d)
            t1, t2, t3 = None, None, None

    if corpus == None:
        return i, d['question'], d['answer'], 'System mistake', None, None, None, corpus, d['supports'], t1, t2, t3
    try:
        qa_prompt = PromptTemplate(input_variables = ['question'], template = prompt_qa(corpus))
        qa_chain = LLMChain(llm = llm, prompt = qa_prompt)
        
        eval_prompt = PromptTemplate(input_variables = ['question', 'answer', 'prediction'], template = prompt_eval())
        eval_chain = LLMChain(llm = llm, prompt = eval_prompt)
        
        pred = qa_chain.run(d['question'])
        
        grade = eval_chain.run({'question': d['question'], 'answer': d['answer'], 'prediction': pred})

        # print(pred, grade)

        c_tokens = sum([len(nltk.word_tokenize(c)) for c in corpus])
        p_tokens = len(nltk.word_tokenize(pred))

        # print('===================================', flush = True)
        # print('Question:', d['question'], flush = True)
        # print('Answer:', d['answer'], flush = True)
        # print('Prediction:', pred, flush = True)
        # print(qa_prompt.template, flush = True)
        # print(d['supports'], flush = True)
        # print('===================================')
        return i, d['question'], d['answer'], pred, grade, c_tokens, p_tokens, corpus, d['supports'], t1, t2, t3
                
    except Exception as e:
        return i, d['question'], d['answer'], 'System mistake', None, None, None, corpus, d['supports'], t1, t2, t3


if __name__ == '__main__':
    args = parse_args()
    args.path = os.getcwd()

    llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.0)
    data = json.load(open('./dataset/{}/test_docs.json'.format(args.dataset), 'r')) #For figure analysis, it is just 200 data samples; For table result, it is the whole testing data

    args.run_info = f'{args.retriever}_{args.k}'

    data_idx = [(i, d) for i, d in enumerate(data)]

    if args.retriever in ['Golden', 'No',
                          'BM25', 'TF-IDF', 'KNN', 
                          'MDR', 'DPR',
                          'KGP w/o LLM', 'LLaMA', 'T5',
                          'KGP-T5', 'KGP-LLaMA']:
        prompt_qa = prompt_qac_wiki
    elif args.retriever == 'No':
        prompt_qa = prompt_qa_wiki



    if args.retriever == 'Golden':
        retriever = Golden_retriever(args.k)

    elif args.retriever == 'No':
        retriever = No_retriever(args.k)

    elif args.retriever == 'BM25':
        retriever = BM25_retriever(args.k)

    elif args.retriever == 'TF-IDF':
        retriever = TF_IDF_retriever(args.k)

    elif args.retriever == 'KNN':
        retriever = KNN_retrieval(args.text_encoder, args.k, args.k_emb)
        args.run_info = f'{args.retriever}_{args.k_emb}_{args.k}'
        data = pkl.load(open('./dataset/{}/test_docs_emb.pkl'.format(args.dataset), 'rb'))
        data_idx = [(i, d) for i, d in enumerate(data)]

    elif args.retriever == 'LLaMA':
        retriever = llm_retriever_LLaMA(args.k, args.k_nei, args.port)

    elif args.retriever == 'T5':
        retriever = llm_retriever_T5(args.k, args.k_nei, args.port)

    elif args.retriever == 'MDR':
        corpus = json.load(open('./dataset/{}/MDR_context.json'.format(args.dataset), 'r'))
        retriever = MDR_retrieval(corpus)

    elif args.retriever == 'DPR':
        corpus = json.load(open('./dataset/{}/DPR_context.json'.format(args.dataset), 'r'))
        retriever = DPR_retrieval(corpus)
    
    elif args.retriever == 'KGP w/o LLM':
        retriever = KG_retriever(args.k)
        Gs = pkl.load(open('./dataset/{}/{}.pkl'.format(args.dataset, args.kg), 'rb'))
        data_idx = [(i, d, Gs[i]) for i, d in enumerate(data)]
        args.run_info = f'{args.retriever}_{args.kg}_{args.k}'

    elif args.retriever == 'KGP-T5':
        retriever = llm_retriever_KG_T5(args.k, args.k_nei, args.port)
        Gs = pkl.load(open('./dataset/{}/{}.pkl'.format(args.dataset, args.kg), 'rb'))
        data_idx = [(i, d, Gs[i]) for i, d in enumerate(data)]
        args.run_info = f'{args.retriever}_{args.kg}_{args.k}'
    
    elif args.retriever == 'KGP-LLaMA':
        retriever = llm_retriever_KG_LLaMA(args.k, args.k_nei)
        Gs = pkl.load(open('./dataset/{}/{}.pkl'.format(args.dataset, args.kg), 'rb'))
        data_idx = [(i, d, Gs[i]) for i, d in enumerate(data)]
        args.run_info = f'{args.retriever}_{args.kg}_{args.k}'

    func = partial(run, retriever = retriever, prompt_qa = prompt_qa, \
                   prompt_eval = prompt_eval, llm = llm, args = args)

    res = []
    with Pool(processes = args.n_processes) as p:
        for i, question, answer, pred, grade, c_tokens, p_tokens, corpus, supports, t1, t2, t3 in \
            tqdm(p.imap_unordered(func, data_idx), total = len(data_idx)):
        
            res.append({'idx': i, 
                        'question': question,
                        'answer': answer,
                        'prediction': pred, 
                        'grade': grade, 
                        'c_tokens': c_tokens,
                        'p_tokens': p_tokens, 
                        'corpus': corpus, 
                        'supports': supports,
                        't_seed': t1, 
                        't_reason': t2,
                        't_nei': t3
                        })

    json.dump(res, open('./result/{}/{}.json'.format(args.dataset, args.run_info), 'w'))
