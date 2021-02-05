# ARCUS: An Adaptive Framework for Autoencoder-based Anomaly Detection from a Complex Evolving Data Stream


## Required packages
- Tensorflow 2.2.0
- Python 3.8.3
- Scikit-learn 0.23.1
- Numpy 1.18.5
- Pandas 1.0.5

## Links for large data sets (exceeding 100MB)
- MNIST_AbrRec: [link](https://drive.google.com/file/d/1KD7gInFa3el08oyHPLDtriF6PDwvkMzM/view?usp=sharing)
- F_MNIST_AbrRec: [link](https://drive.google.com/file/d/1WSd0KSmswqgwhC5OM-eZ3hX6KoAgDJDg/view?usp=sharing)
- MNIST_GrdRec: [link](https://drive.google.com/file/d/1drEgdYDLlaMH7P565Du6KUeabqm4mYbJ/view?usp=sharing)
- F_MNIST_GrdRec: [link](https://drive.google.com/file/d/1PNK9bliTGpEiEimnfy8Saf9p0G3BFLBo/view?usp=sharing)

## Default parameter values
- batch = 512
- min_batch = 32
- init_epoch = 5
- intm_epoch = 1
- hidden_dims = The number of dimensionality explaining at least 70% of the variance in PCA
- model_type = one of ["RAPP", "RSRAE", "DAGMM"]
- inf_type = on eof ["INC", "ADP"] # "INC" for drift-unaware (incremental) and "ADP" for drift-aware (adaptive, proposed)

## Example usage

Edit test.py or test.ipynb

```
python test.py
----------------------------
Data set: MNIST_AbrRec
Model type:  RAPP
Reliability threshold:  0.99
Similarity threshold:  0.8
Learning rate:  0.0001
AUC: 0.874
```

