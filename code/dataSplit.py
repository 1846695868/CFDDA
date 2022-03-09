import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import KFold

def cv10split(Afile, trainfile, testfile, testNegfile):
    DiDrMat = np.loadtxt(Afile)  # 读入邻接矩阵
    DrDiMat = DiDrMat.transpose()
    num_users, num_items = DrDiMat.shape
    posMat = sp.dok_matrix((num_users, num_items), dtype=np.float32)
    posArray, negArray = [], []
    posTriple = []  # 正样本三元组
    for u in range(num_users):
        posLine, negLine = [], []
        for i in range(num_items):
            if (DrDiMat[u][i] == 1):
                posLine.append(i)
                posTriple.append([int(u), int(i), 1])
                posMat[u, i] = 1
            else:
                negLine.append(i)
        posArray.append(posLine)
        negArray.append(negLine)

    kf = KFold(n_splits=10, shuffle=True)  # 10fold
    no = 1
    for train_index, test_index in kf.split(X=posTriple):
        testPosTriple, trainPosTriple, testNegatives = [], [], []
        for ii in train_index:
            x = posTriple[ii]
            trainPosTriple.append(x)
        for jj in test_index:
            x = posTriple[jj]
            testPosTriple.append(x)
            u, i = int(x[0]), int(x[1])
            s = '(' + str(u) + ',' + str(i) + ')'
            for n in negArray[u]:
                s = s + '\t' + str(n)
            testNegatives.append(s)
        f = trainfile + str(no)
        np.savetxt(f, trainPosTriple, fmt='%d', delimiter='\t')
        np.savetxt(testfile + str(no), testPosTriple, fmt='%d', delimiter='\t')
        np.savetxt(testNegfile + str(no), testNegatives, fmt='%s')
        no += 1
    return


if __name__ == '__main__':
    path = '../data/'
    cv10split(path + "DiDrA.txt", "../data/train.rating", "../data/test.rating", '../data/test.negative')
