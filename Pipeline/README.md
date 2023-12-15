# KG-LLM-Doc - Pipeline
This repository includes the code for the pipeline of the main table

## Folder architecture
* knn_emb.py: the pre-processing file to obtain sentence embeddings for KNN search retriever
* main.py
* parse.py: arguments
* prompt.py: prompt template
* retriever.py: all different retriever methods
* run_{dataset}.sh
* utils.py: general purposes

* evaluation: jupyter notebooks for evaluate the final answers and calculate metrics in main Table
* result: save the answers and retrieved sentences
* dataset: testing documents and generated knowledge graphs for each dataset
