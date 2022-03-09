import heapq
import math
import numpy as np
from sklearn import metrics
import tensorflow as tf

# Global variables that are shared across processes
_sess = None
_predict = None
_train_input_user = None
_train_input_item = None
_train_input_user_sim = None
_train_input_item_sim = None
_drugSim = None
_diseaseSim = None
_testRatings = None
_testNegatives = None
_K = None


def evaluate_model(sess, predict, train_input_user, train_input_item, train_input_user_sim, train_input_item_sim, drugSimEmbed, diseaseSimEmbed, testRatings, testNegatives, K):

    global _sess
    global _predict
    global _train_input_user
    global _train_input_item
    global _train_input_user_sim
    global _train_input_item_sim
    global _drugSimEmbed
    global _diseaseSimEmbed
    global _testRatings
    global _testNegatives
    global _K

    _sess = sess
    _predict = predict
    _train_input_user = train_input_user
    _train_input_item = train_input_item
    _train_input_user_sim = train_input_user_sim
    _train_input_item_sim = train_input_item_sim
    _drugSimEmbed = drugSimEmbed
    _diseaseSimEmbed = diseaseSimEmbed
    _testRatings = testRatings
    _testNegatives = testNegatives
    _K = K

    # Single thread
    hits = evaluteHits()
    (auc, fpr, tpr, aupr, precision, recall) = evaluteAUC()

    return hits, auc, fpr, tpr, aupr, precision, recall

def evaluteAUC():
    users, items, label = [], [], []
    for idx in range(len(_testRatings)):   #遍历正样本对
        rating = _testRatings[idx]  # 单个正样本
        u = int(rating[0])
        i = int(rating[1])
        users.extend([u])
        items.extend([i])
        label.extend([1])

        negItem = _testNegatives[idx]  # 对应的负样本
        negUser = np.full(len(negItem), int(u), dtype='int32')  # len(items)长的数组，用u来填充
        users.extend(negUser)
        items.extend(negItem)
        label.extend(np.zeros(len(negItem)))
    user_sim = []
    item_sim = []
    for i in users:
        user_sim.append(_drugSimEmbed[i])
    for j in items:
        item_sim.append(_diseaseSimEmbed[j])

    # Get prediction scores
    predictions = _sess.run(_predict, feed_dict={_train_input_user: users, _train_input_item: items, _train_input_user_sim: user_sim, _train_input_item_sim: item_sim})

    fpr, tpr, thre = metrics.roc_curve(label, predictions, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    precision, recall, _thresholds = metrics.precision_recall_curve(label, predictions)
    aupr = metrics.auc(recall, precision)

    return auc, fpr, tpr, aupr, precision, recall

def  evaluteHits():  # 1:99逐个样本评估和测试
    hits = []
    for idx in range(len(_testRatings)):
        hr = eval_one_rating(idx)
        hits.append(hr)
    return hits

def eval_one_rating(idx):   #一条测试语料的评估
    rating = _testRatings[idx]    # 单个正样本
    items = _testNegatives[idx]  # 对应的负样本
    u = rating[0]  #user
    gtItem = rating[1]   # item  #disease正样本
    items.append(gtItem)  # 1正样本加入到99个负样本;

    # Get prediction scores
    map_item_score = {}
    users = np.full(len(items), u, dtype='int32')   # len(items)长的数组，用u来填充
    user_sim = []
    item_sim = []
    for i in users:
        user_sim.append(_drugSimEmbed[i])
    for j in items:
        item_sim.append(_diseaseSimEmbed[j])
    predictions = _sess.run(_predict, feed_dict={_train_input_user: users, _train_input_item: items, _train_input_user_sim: user_sim, _train_input_item_sim: item_sim})  # 返回100个概率值

    for i in range(len(items)):  # 100个预测结果排序
        item = items[i]   # disease id
        map_item_score[item] = predictions[i]
    items.pop()   # gtItem去掉

    # Evaluate top rank list
    ranklist = heapq.nlargest(_K, map_item_score, key=map_item_score.get)  # _K=10
    hr = getHitRatio(ranklist, gtItem)    # ranglist 是topK的item下标
    return hr


def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0
