# Dataset
This repository includes the data we need to prepare for running for results in the main table.

## What files do we need to prepare in advance for each dataset folder?
* **test_docs.json**: set of documents for each question in the testing set, obtained from **step 1 Document Collection** in the main work flow.
* **test_docs_graph.pkl/graph_tagme_0.8.pkl**: the constructed knowledge graph for each set of documents, obtained from **step 2 Knowledge Graph Construction** in the main work flow, note that we provide several ways to construct knowledge graphs and the main table use the TAGME one.
* **dpr_context.json**: retrieved supporting facts for testing questions by DPR, obtained from **step 3 DPR retrieval** in the main work flow.
* **mdr_context.json**: retrieved supporting facts for testing questions by MDR, obtained from **step 4 MDR retrieval** in the main work flow.
* **test_docs_emb.pkl**: the embeddings for each passage from the set of documents for each testing question, obtained by sentence transformer by running knn_emb.py in **step 7**

