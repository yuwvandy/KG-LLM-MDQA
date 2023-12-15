# KG-LLM-Doc
This repository includes the code and demo of [Knowledge Graph Prompting for Multi-Document Question Answering](https://arxiv.org/abs/2308.11730).


# Folder Architecture
* **Data-Collect**: Codes for querying/collecting Documents based on QA datasets from existing literature.
* **DPR**: Codes for training DPR, dense passage retrieval.
* **MDR**: Codes for training MDR, multi-hop dense passage retrieval.
* **T5**: Codes for instruction fine-tuning T5 based on reasoning data of HotpotQA and 2WikiMQA, the pre-trained T5 would be used as the agent for intelligent graph traversal.
* **LLaMA**: Codes for instruction fine-tuning LLaMA-7B based on reasoning data of HotpotQA and 2WikiMQA, the pre-trained LLaMA would be used as the agent for intelligent graph traversal.
* **Pipeline**: Codes for reproducing our KGP-LLM algorithm and other models in the main Table in the paper

All model checkpoints and real datasets are separately stored in the [Dropbox](https://www.dropbox.com/scl/fo/y9ydmvv0bj846klkfdin0/h?rlkey=epyzclz2kbcf2g4iuz0tojlm9&dl=0)!

# Environment Configuration
```
conda install -c anaconda python=3.8
pip install -r requirements.txt
pip install langchain
pip install nltk
pip install -U scikit-learn
pip install rank_bm25
pip install -U sentence-transformers
pip install -U pip setuptools wheel
pip install -U spacy
python -m spacy download en_core_web_lg
pip install torch-scatter
pip install Levenshtein
pip install openai==0.28
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install sentencepiece
pip install transformers
```


# Work Flow
### 1. Document Collection
We query Wikipedia based on the QA data from existing literature.
* **Input**: train/val.json
* **Output**: train/val/test_docs.json
```
cd Data-Collect/{dataset}
python3 process.py
```

### 2. Knowledge Graph Construction
We generate KG for each set of documents (paired with a question) using TF-IDF/KNN/TAGME of various densities. For the main Table experiment, we only use the graph constructed by TAGME with prior_prob 0.8.
* **Input**: train/val/test_docs.json
* **Output**: .pkl
```
cd Data-Collect
bash run_{dataset}.sh
```

### 3. DPR
We fine-tune the DPR model based on training queries and supporting passages and further use it for obtaining DPR baseline performance.

* HotpotQA: we already have well-curated negative samples in train/val_with_neg_v0 obtained [here](https://github.com/facebookresearch/multihop_dense_retrieval) provided in the [Dropbox](https://www.dropbox.com/scl/fo/y9ydmvv0bj846klkfdin0/h?rlkey=epyzclz2kbcf2g4iuz0tojlm9&dl=0).
* MuSiQue/Wiki2MQA/IIRC: we randomly sample negative passages before each training epoch.
```
cd DPR
bash run.sh
```
After run.sh, you will get **model.pt** and **retrieved contexts** for supporting facts for questions in test_docs.json.

### 4. MDR
We fine-tune the MDR model based on training queries and supporting passages and further use it for obtaining DPR baseline performance. Please note that we only fine-tune MDR for HotpotQA and MuSiQue as we have access to their multi-hop inference training data while for Wiki2MQA and IIRC, we directly use model.pt from HotpotQA to retrieve contexts.

* HotpotQA: Multi-hop training data obtained [here](https://github.com/facebookresearch/multihop_dense_retrieval), we already provided in the [Dropbox](https://www.dropbox.com/scl/fo/y9ydmvv0bj846klkfdin0/h?rlkey=epyzclz2kbcf2g4iuz0tojlm9&dl=0).
* MuSiQue: Multi-hop training data obtained ourselves, we already provided in the [Dropbox](https://www.dropbox.com/scl/fo/y9ydmvv0bj846klkfdin0/h?rlkey=epyzclz2kbcf2g4iuz0tojlm9&dl=0).
* IIRC/Wiki2MQA: We use model.pt from HotpotQA for retrieval.
```
cd MDR
bash run.sh
```
After run.sh, you will get **model.pt** and **retrieved contexts** for supporting facts for questions in test_docs.json.

### 5. T5
See README.md in T5 Folder 

### 6. LLaMA
See README.md in LLaMA Folder 


### 7. Reproducing Main Table Results.
After completing all the above steps, we need to put the **testing documents, retrieved contexts from the model, and generated knowledge graph** in the corresponding place. Then, we have everything ready for the main Table in the paper. Specifically, see README.md in Pipeline to configure the files.

* **DPR**: require DPR_context.json (get from following 3. DPR instruction)
* **MDR**: require MDR_context.json (get from following 4. MDR instruction)
* **LLaMA**: require open LLaMA API (open following 6. LLaMA instruction, ensure the port number is consistent, we use localhost with port number 5000 by default)
* **T5**: open T5 API (open following 5. T5 instruction, ensure the port number is consistent, we use localhost with port number 5000 by default)
* **KGP w/o LLM**: require the type of KG our LLM is traversing on (get from following 1.-2. instructions)
* **KGP-T5/LLaMA/MDR**: require KG file (get from following 1-2 instructions) and LLM API is open (5-6 instructions)
* **KGNN**: require passage embeddings for each passage in the documents for each testing question (obtained by running knn_emb.py)

Then run the following commands:
```
cd Pipeline
bash run_{dataset}.sh
```
> [!important]  
> We use multi-parallel processing to call LLM to process each question for each set of documents, which would incur a large amount of consumption when calling with OpenAI API call, therefore, please adaptively change the number of CPUs when parallel calling API according to your budget.

For final evaluation the generated answer:
```
cd evaluation
jupyter notebook eval.ipynb
Run the corresponding kernels
```

