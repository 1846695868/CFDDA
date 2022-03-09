import scipy.sparse as sp
import numpy as np


class Dataset(object):

    def __init__(self, path, fold):
        self.posMatrix, self.num_users, self.num_items = self.AdjtoDict(path + "DiDrA.txt")
        self.trainMatrix = self.load_rating_file_as_matrix(path + "train.rating" + str(fold))
        self.testRatings = self.load_rating_file_as_list(path + "test.rating" + str(fold))
        self.testNegatives = self.load_negative_file(path + "test.negative" + str(fold))
        assert len(self.testRatings) == len(self.testNegatives)

    # 读邻接矩阵中的gold data放入字典中(userid, item id)
    def AdjtoDict(self, Afile):  # 三元组变成字典
        DiDrMat = np.loadtxt(Afile)  # 读入邻接矩阵
        DrDiMat = DiDrMat.transpose()
        num_users, num_items = DrDiMat.shape
        posMat = sp.dok_matrix((num_users, num_items), dtype=np.float32)
        for u in range(num_users):
            for i in range(num_items):
                if ( DrDiMat[u][i] == 1):
                    posMat[u, i] = 1
        return posMat, num_users, num_items

    def load_rating_file_as_matrix(self, filename):   #三元组变成字典
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            line.strip('\n')
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        # Construct matrix
        mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                if (rating > 0):
                    mat[user, item] = 1.0    #没有负样本
                line = f.readline()
        return mat

    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            line=line.strip('\n')
            while line != None and line != "":
                arr = line.split('\t')
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                line = f.readline().strip('\n')
        return ratingList

    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[1: ]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList
