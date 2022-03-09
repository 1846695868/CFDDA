# CFDDA
Code and Datasets for "Collaborative Filtering Drug-Disease Associations"
## Datasets
* data/PREDICT/DiDrA.txt is the disease-drug association matrix, which contain 1933 associations between 313 diseases and 593 drugs.<br><br>
* data/PREDICT/DiseaseSim.txt is the disease similarity matrix of 313 diseases, which is calculated based on disease mesh descriptors.<br><br>
* data/PREDICT/DrugSim.txt is the drug similarity matrix of 593 drugs, which is calculated based on SMILES.<br><br>
* data/PREDICT/diseaseSimEmbedSorted.txt is the potential vector representation of the 313 diseases, which is obtained by performing weighted random walk on the disease similarity matrix.<br><br>
* data/PREDICT/drugSimEmbedSorted.txt is the potential vector representation of the 593 drugs, which is obtained by performing weighted random walk on the drug similarity matrix.<br><br>
* data/PREDICT/train(1-10).rating are positive samples in the training set by 10-fold division.<br><br>
* data/PREDICT/test(1-10).rating are positive samples in the test set by 10-fold division.<br><br>
* data/PREDICT/test(1-10).negative are negative samples in the test set.<br><br>
The CDatasets and PREDICT are of the same type and will not be repeated.
## Code
__Environment Requirement__<br>
The code has been tested running under Python 3.6. The required packages are as follows:
* numpy == 1.16.0
* scipy == 1.1.0
* sklearn == 0.21.2
* tensorflow == 1.13.1

__Usage__<br>
```
git clone https://github.com/1846695868/CFDDA.git
cd CFDDA/code
python NeuMF.py
```
