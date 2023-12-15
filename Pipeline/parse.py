import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", nargs="?", default="IIRC", choices=["HotpotQA", 'IIRC', '2WikiMQA', 'MuSiQue'])
    parser.add_argument("--seed", type=int, default=1028)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--n_processes", type=int, default = 8)

    parser.add_argument("--llm", type=str, default='turbo')

    parser.add_argument("--retriever", type=str, default='BM25')
    parser.add_argument("--k", type = int, default = 30)
    parser.add_argument("--k_nei", type = int, default = 3)
    parser.add_argument("--k_emb", type = int, default = 15)

    parser.add_argument("--round", type = int, default = 5)
    parser.add_argument("--text_encoder", type = str, default = 'multi-qa-MiniLM-L6-cos-v1')

    parser.add_argument("--kg", type = str, default = 'TF-IDF')

    parser.add_argument("--n_data", type = int, default = 10)
    parser.add_argument("--port", type = int, default = 5000)

    
    return parser.parse_args()