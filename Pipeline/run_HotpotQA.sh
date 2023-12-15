echo "No"
python3 main.py --retriever='No' --dataset='HotpotQA' --n_processes=16

echo "Golden"
python3 main.py --retriever='Golden' --dataset='HotpotQA' --n_processes=16

echo "BM25"
python3 main.py --retriever='BM25' --k=30 --dataset='HotpotQA' --n_processes=16

echo "TF-IDF"
python3 main.py --retriever='TF-IDF' --k=30 --dataset='HotpotQA' --n_processes=16

echo "MDR"
python3 main.py --retriever='MDR' --dataset='HotpotQA' --k=30 --n_processes=8

echo "DPR"
python3 main.py --retriever='DPR' --dataset='HotpotQA' --k=30 --n_processes=8

echo "KNN"
python3 main.py --retriever='KNN' --k=30 --k_emb=15 --dataset='HotpotQA'

echo "KGP w/o LLM"
python3 main.py --retriever='KGP w/o LLM' --dataset='HotpotQA' --k=30 --n_processes=8 --kg="kg_TAGME_0.8"

echo "T5"
python3 main.py --retriever='T5' --dataset='HotpotQA' --k=30 --n_processes=1 --port=5000

echo "LLaMA"
python3 main.py --retriever='LLaMA' --dataset='HotpotQA' --k=30 --n_processes=1 --port=5000

echo "KGP-T5"
python3 main.py --retriever='KGP-T5' --dataset='HotpotQA' --k=30 --n_processes=1 --port=5000 --kg="kg_TAGME_0.8"

echo "KGP-LLaMA"
python3 main.py --retriever='KGP-LLaMA' --dataset='HotpotQA' --k=30 --n_processes=1 --port=5000  --kg="kg_TAGME_0.8"



