import numpy as np
import time
import random
from ARCUS import *
tf.keras.backend.set_floatx('float64')

def load_dataset(dataset_name):
    if dataset_name in ["MNIST_AbrRec", "F_MNIST_AbrRec", "MNIST_GrdRec", "F_MNIST_GrdRec"]:
        values = np.loadtxt("datasets/"+dataset_name+"_value.csv", delimiter = ',', dtype=np.float64)
        labels = np.loadtxt("datasets/"+dataset_name+"_label.csv", delimiter = ',', dtype=np.float64)
        return tf.data.Dataset.from_tensor_slices((values, labels))
    else:
        Other = pd.read_csv("datasets/"+dataset_name+".csv", dtype=np.float64)
        Other_label = Other.pop('label')
        return tf.data.Dataset.from_tensor_slices((Other.values, Other_label.values))

ARCUS_instance = ARCUS()

# Problem-setting related hyper parameters
batch = 512
min_batch = 32
init_epoch = 5
intm_epoch = 1
hidden_dims = {
    "MNIST_AbrRec": 24,  "MNIST_GrdRec": 25, "F_MNIST_AbrRec": 9,  "F_MNIST_GrdRec": 15, "GAS": 2, "RIALTO": 2 
    ,'INSECTS_Abr': 6,'INSECTS_Incr': 7, 'INSECTS_IncrGrd': 8,'INSECTS_IncrRecr': 7
} # The number of pricipal component explaining at least 70% of variance.


# Algorithm related hyper parameters
model_type = "RAPP" # "RAPP, "RSRAE", "DAGMM"
inf_type = "ADP" # "INC" - drift-unaware, "ADP"- drift-aware
dataset_name = "MNIST_AbrRec" # "MNIST_AbrRec", "MNIST_GrdRec", "F_MNIST_AbrRec",  "F_MNIST_GrdRec", "GAS", "RIALTO", 'INSECTS_Abr','INSECTS_Incr', 'INSECTS_IncrGrd','INSECTS_IncrRecr'
layer_num = 3
RSRAE_hidden_layer_size = [32, 64, 128] # Suggested by the author. The one or two layers of them may be used according to data sets.
learning_rate = 1e-4
reliability_thred = 0.95 
similarity_thred = 0.8

#################################################

dataset = load_dataset(dataset_name)
hidden_dim = hidden_dims[dataset_name]
rand_seed = random.randint(0,1000)
returned, auc_hist, anomaly_scores = ARCUS_instance.simulator(dataset, model_type, inf_type, batch, min_batch, learning_rate, layer_num, hidden_dim, init_epoch, intm_epoch, reliability_thred, similarity_thred, rand_seed, RSRAE_hidden_layer_size)

if(returned):
    print("Data set:",dataset_name)
    print("Model type: ", model_type)
    print("Reliability threshold: ", reliability_thred)
    print("Similarity threshold: ", similarity_thred)
    print("Learning rate: ", learning_rate)
    print("AUC:", np.round(np.mean(auc_hist),3))
else:
    print("Error occurred")
