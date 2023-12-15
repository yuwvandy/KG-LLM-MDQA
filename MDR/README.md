# KG-LLM-Doc - MDR
This repository includes the MDR code

## Fine-tuning/Evaluating MDR
* main.py
* dataset.py: pre-process/load data
* loader.py: how to load data
* learn.py: train/inference
* parse.py: arguments
* model.py: model
* utils.py: general purposes
* main_eval.py: After fine-tuning, run this for retrieval supporting facts for questions in test set

```
bash run.sh
```

## Run MDR as an API (used as the graph traversal agent in KG-LLM)
### MDR for documental-based QA
Run API and performing real-time sentences encoding when new document was loaded
```
python3 mdr_doc_api.py
```
### MDR for open-world QA (setting aligned with [here](https://github.com/facebookresearch/multihop_dense_retrieval/tree/main))
Encode all pre-stored Wikipedia sentences and run API:
```
python3 encode_all_wiki_corpus.py
python3 mdr_ow_api.py
```
