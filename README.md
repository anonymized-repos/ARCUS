# ARCUS: An Adaptive Framework for Autoencoder-based Anomaly Detection from a Complex Evolving Data Stream

## Introduction
Online anomaly detection from a data stream is critical for the safety and security of many applications, but is facing severe challenges recently, due to complex and evolving data streams from IoT devices and cloud-based infrastructures. Unfortunately, state-of-the-art approaches fall too short for these challenges. Specifically, shallow model approaches (e.g., k nearest neighbors) bear the burden of crafted feature engineering to handle the complexity; deep model approaches (e.g., deep autoencoder) may be off such a burden, but none exists that can handle the evolving data distribution. This paper presents a framework for autoencoder-based anomaly detection, ARCUS, equipped to handle the challenges using a model pooling approach. The framework can be instantiated with any current autoencoder-based anomaly detection models, and is characterized by two novel techniques: concept-driven inference and drift-aware model pool update; the former detects anomalies with a combination of models most appropriate for the complexity, and the latter adapts the model pool dynamically to fit the evolving data. A comprehensive experiment using four synthetic and six real data sets showed that ARCUS improved the anomaly detection accuracy of state-of-the-art autoencoder-based models by up to 25%, and outperformed other conventional models by up to 65%.

## Required packages
- Tensorflow 2.2.0
- Python 3.8.3
- Scikit-learn 0.23.1
- Numpy 1.18.5
- Pandas 1.0.5

## Data sets description
![dataset](https://drive.google.com/uc?export=view&id=1DlOpvv4DG1Vg5MucHoT7rdqrn78mxzzn)

### Links for large data sets (exceeding 100MB)
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

### Encoder layer sizes (reversed in a decoder) and learning rates which achieved the best AUC results
![hyperparameters](https://drive.google.com/uc?export=view&id=1idoOEHKQTsQsjdY9WW2jzJQ0KlkdxA5E)

## Example usage

Change the parameter values in test.py or test.ipynb following your test scenario and run the file.

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

## Example concept drift adaptation of ARCUS in INSECTS data sets
![casestudy](https://drive.google.com/uc?export=view&id=1MppDlvxLx32b6sPc2U0xi6fEg460jmUN)

