import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", nargs="?", default="HotpotQA", choices=["HotpotQA", 'IIRC', '2WikiMQA', 'MuSiQue'])
    parser.add_argument("--n_processes", type=int, default = 8)

    parser.add_argument("--kg", nargs="?", default="MDR-KNN", choices=["TF-IDF", "KNN", "wiki_spacy", "TAGME", "MDR-KNN"])

    parser.add_argument("--n_kw", type=int, default=50)
    parser.add_argument("--min_n", type=int, default=1)
    parser.add_argument("--max_n", type=int, default=4)

    parser.add_argument("--prior_prob", type=float, default=0.8)

    parser.add_argument("--k_knn", type=int, default=10)

    parser.add_argument("--mhop_model_name", nargs="?", default="bert-base-uncased")
    parser.add_argument("--max_len", type=int, default=300)
    parser.add_argument("--max_q_sp_len", type=int, default=350)
    
    
    return parser.parse_args()