### KG-LLM-Document Collection
We query Wikipedia based on the QA data from existing literature.
* **Input**: train/val.json
* **Output**: train/val/test_docs.json
```
cd Data-Collect/{dataset}
python3 process.py
```


After completing the above process, you should see the following files in each corresponding dataset folder:

* train.json
* dev.json: 
* text_process.py
* text_split.py
* process.py

* **all_docs.json**: all documents for each question
* **process_train_data.json**
* **process_val_data.json**
* **train_docs.json**： Set of documents for questions in training set
* **val_docs.json**: Set of documents for questions in validation set
* **test_docs.json**: Set of documents for questions in testing set



### KG-LLM-Knowledge Graph Construction
We generate KG for each set of documents (paired with a question) using TF-IDF/KNN/TAGME of various densities. For the main Table experiment, we only use the graph constructed by TAGME with prior_prob 0.8.
* **Input**: train/val/test_docs.json
* **Output**: .pkl
```
cd Data-Collect
bash run_{dataset}.sh
```

After completing the above process, you should see the following files in each corresponding dataset folder:

* train.json
* dev.json: 
* text_process.py
* text_split.py
* process.py

* **all_docs.json**: all documents for each question
* **process_train_data.json**
* **process_val_data.json**
* **train_docs.json**: Set of documents for questions in training set
* **val_docs.json**: Set of documents for questions in validation set
* **test_docs.json**: Set of documents for questions in testing set
* **KG_TAGME_0.8.pkl**

