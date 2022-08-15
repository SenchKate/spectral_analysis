Spectral Analysis 

The current paper is an analysis of the mathod offered by the authors of the paper "Unmasking Clever Hans predictors and assessing what machines really learn". 

Thre current git provides the functions needed to create LRP heat maps for 7 different datasets:

Eurosat - https://github.com/phelber/EuroSAT

Tiny Imagenet - https://paperswithcode.com/dataset/tiny-imagenet

CIFAR-10  https://www.cs.toronto.edu/~kriz/cifar.html

Fashion MNIST https://www.kaggle.com/datasets/zalando-research/fashionmnist

SEN12MS https://paperswithcode.com/dataset/sen12ms

RSICD https://paperswithcode.com/dataset/rsicd

xView2 https://xview2.org

So2Sat LCZ42 https://paperswithcode.com/paper/so2sat-lcz42-a-benchmark-dataset-for-global

The LRP studies how trained VGG16 process Tiny Imagenet dataset and trained ResNet50 - all other datasets.

The Spectral Relevance analysis of LRP heatmaps is implemented here: 
https://colab.research.google.com/drive/1O32UqpTSlG80nTxWWrQmnFuMktMAZ1Zn?usp=sharing
The notebook also contrains OOD/ID, hyperparameter and sensitivity analyzes.

The playground witht he example how to repeat the experiments are presented here:
https://colab.research.google.com/drive/1b-Ls1SLt7rDifUQFAIqRLdxXdRtHO_9Z?usp=sharing

