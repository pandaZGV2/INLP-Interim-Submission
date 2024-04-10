# Transformer Models for Text Coherence Assessment

In this repository, we are working on Text Coherence Assessment of [paper](https://arxiv.org/abs/2109.02176).

Install Preprocessed dataset from [here](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/aaron_monis_students_iiit_ac_in/EnBVUZweaENOnrA6iSG-E0kBdAsc_26RksbdyNDm14fKmQ?e=33G21d) and add in folder processed_data, folder.

To train the model, you have to use the run.sh file and change the parameters in it as required. Then simply do the following:

The metrics are as follows:

- `corpus` can take one of 'gcdc' or 'wsj'.
- `sub_corpus` can take anyone value from  'Clinton', 'Enron', 'Yelp' or 'Yahoo' if `corpus` is gcdc
- `arch` can take one of vanilla, hierarchical
- `task` can take one of 3-way-classification, minority-classification,sentence-ordering or sentence-score-prediction for GCDC dataset and only sentence-ordering for WSJ dataset
- `model_name` defines transformer model to use. (by-default its's roberta-base)

```
bash run.sh
```

For evaluating on datasets, do the following:

```
bash eval.sh
```

We also have submitted the model [here](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/shreyansh_agarwal_students_iiit_ac_in/EmYpALMH13JJt76HK81e4kgBF8gH2CqYGZU_DH7GZeQ-PQ?e=wQuPSx) 

The current model is pretrained on vanilla transformer on sentence ordering task for vanilla transformers.
The logs for running the model have also been attached here.
