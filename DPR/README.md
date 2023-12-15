# KG-LLM-Doc - DPR
This repository includes the DPR code

## Fine-tuning/Evaluating DPR
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

## Data
* HotpotQA: we directly use train/dev_with_neg_v0.json from [here](https://github.com/facebookresearch/multihop_dense_retrieval/tree/main), we already provided in the Dropbox.
* MuSiQue/IIRC/2WikiMQA: train/val.json from the previous steps in the main repo and chunks from, we already provided in the Dropbox.
