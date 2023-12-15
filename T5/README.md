# KG-LLM-Doc - T5
This repository includes the code for instruction-fine-tuning T5 for enhancing reasoning capability. The code is modified based on the jupter-notebook [here](https://github.com/Shivanandroy/T5-Finetuning-PyTorch/blob/main/notebook/T5_Fine_tuning_with_PyTorch.ipynb).

## Instruction Fine-tuning T5
* main.py
* dataset.py: pre-process/load data
* learn.py: train/inference
* parse.py: arguments
* utils.py: general purposes

* t5_api.py: open fine-tuned T5 as a API and perform reasoning as a graph traversal agent

```
python -m torch.distributed.launch --nproc_per_node={num_gpus} main.py
```

## Data
reason_instruction.json: we rearange the training data from HotpotQA and MuSiQue and organize as the following format, e.g.,:
* **Instruction**: What evidence do we need to answer the question given the current evidence?
* **Input**: Which magazine was started first Arthur's Magazine or First for Women?\nArthur's Magazine (1844–1846) was an American literary periodical published in Philadelphia in the 19th century.
* **Output**: First for Women is a woman's magazine published by Bauer Media Group in the USA.
  
