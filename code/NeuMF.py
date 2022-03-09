import tensorflow as tf
import scipy.sparse as sp
import numpy as np
import heapq
import math
import argparse
from loadData import Dataset
from evaluate import evaluate_model


#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run NeuMF.")
    parser.add_argument('--dataset', nargs='?', default='../data/',  # 选择数据集
                        help='Choose a dataset.')
    parser.add_argument('--adj_type', nargs='?', default='norm',
                        help='Specify the type of the adjacency (laplacian) matrix from {norm}.')
    parser.add_argument('--emb_size', type=int, default=16,
                        help='Embedding size.')
    parser.add_argument('--layer_size', nargs='?', default='[32, 32, 32]',
                        help='Output sizes of every layer')
    parser.add_argument('--fold', type=int, default=1, help='choose a different dataset to train/test')
    parser.add_argument('--epochs', type=int, default=80,  # 训练轮数
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=64,  # 批训练大小
                        help='Batch size.')
    parser.add_argument('--num_mf', type=int, default=16,
                        help='Embedding size of MF model.')
    parser.add_argument('--num_mlp', nargs='?', default='[128,64,32,16]',
                        help="MLP layers. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--num_mlp_drugSim', nargs='?', default='[128, 64, 32, 16]')
    parser.add_argument('--num_mlp_diseaseSim', nargs='?', default='[128, 64, 32, 16]')
    parser.add_argument('--reg_mf', type=float, default=0.01,
                        help='Regularization for MF embeddings.')
    parser.add_argument('--reg_mlp', nargs='?', default='[0,0,0,0]',
                        help="Regularization for each MLP layer. reg_layers[0] is the regularization for embeddings.")
    parser.add_argument('--num_neg', type=int, default=5,  # 负采样率
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,  # 学习率
                        help='Learning rate.')
    parser.add_argument('--topK', type=int, default=10, help='topk test item predict')  # topK预测
    parser.add_argument('--verbose', type=int, default=1,  # 迭代多少次评估一次
                        help='Show performance per X iterations')

    return parser.parse_args()


def get_train_instances(posMat, train, num_negatives):  # 负样本比例10
    user_input, item_input, labels = [], [], []
    num_users, num_items = train.shape[0], train.shape[1]
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative instances
        for t in range(num_negatives):  # 10个负样本从正样本外随机取
            j = np.random.randint(num_items)
            while (u, j) in posMat:  # train.has_key((u, j)):
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels


def get_adj_mat(path, fold, num_users, num_items):
    train_file = path + "train.rating" + str(fold)
    R_mat = create_R_mat(train_file, num_users, num_items)
    norm_adj_mat = create_adj_mat(R_mat, num_users, num_items)

    return norm_adj_mat

def create_R_mat(Afile, num_users, num_items):
    R = sp.dok_matrix((num_users, num_items), dtype=np.float32)
    with open(Afile) as f_train:
        for l in f_train.readlines():
            if len(l) == 0:
                break
            l = l.strip('\n')
            item = [int(i) for i in l.split('\t')]
            R[item[0], item[1]] = 1.
    return R

def create_adj_mat(R_mat, num_users, num_items):
    adj_mat = sp.dok_matrix((num_users + num_items, num_users + num_items), dtype=np.float32)
    adj_mat = adj_mat.tolil()
    R = R_mat.tolil()
    # prevent memory from overflowing
    for i in range(5):
        adj_mat[int(num_users * i / 5.0):int(num_users * (i + 1.0) / 5), num_users:] = \
        R[int(num_users * i / 5.0):int(num_users * (i + 1.0) / 5)]
        adj_mat[num_users:, int(num_users * i / 5.0):int(num_users * (i + 1.0) / 5)] = \
        R[int(num_users * i / 5.0):int(num_users * (i + 1.0) / 5)].T
    adj_mat = adj_mat.todok()
    print('already create adjacency matrix', adj_mat.shape)

    norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))

    print('already normalize adjacency matrix')
    return norm_adj_mat.tocsr()

def normalized_adj_single(adj):
    rowsum = np.array(adj.sum(1))

    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)

    norm_adj = d_mat_inv.dot(adj)
    print('generate single-normalized adjacency matrix.')
    return norm_adj.tocoo()

def init_weights(num_users, num_items, emb_dim, weight_size):
    all_weights = dict()
    # tf.contrib.layers.xavier_initializer()
    all_weights['user_embedding'] = tf.Variable(tf.random_normal(shape=[num_users, emb_dim], mean=0.0, stddev=0.01))
    all_weights['item_embedding'] = tf.Variable(tf.random_normal(shape=[num_items, emb_dim], mean=0.0, stddev=0.01))
    print('using random initialization')  # print('using xavier initialization')
    return all_weights


def create_lightgcn_embed(num_users, num_items, weights, norm_adj, weight_size, n_fold=100):

    A_fold_hat = split_A_hat(norm_adj, num_users, num_items, n_fold)

    ego_embeddings = tf.concat([weights['user_embedding'], weights['item_embedding']], axis=0)
    all_embeddings = [ego_embeddings]

    for k in range(0, len(weight_size)):

        temp_embed = []
        for f in range(n_fold):
            temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))

        side_embeddings = tf.concat(temp_embed, 0)
        ego_embeddings = side_embeddings
        all_embeddings += [ego_embeddings]
    all_embeddings = tf.stack(all_embeddings, 1)
    all_embeddings = tf.reduce_mean(all_embeddings, axis=1, keepdims=False)
    u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [num_users, num_items], 0)
    return u_g_embeddings, i_g_embeddings

def split_A_hat(adj, num_users, num_items, n_fold):
    A_fold_hat = []

    fold_len = (num_users + num_items) // n_fold
    for i_fold in range(n_fold):
        start = i_fold * fold_len
        if i_fold == n_fold -1:
            end = num_users + num_items
        else:
            end = (i_fold + 1) * fold_len

        A_fold_hat.append(convert_sp_mat_to_sp_tensor(adj[start:end]))
    return A_fold_hat


def convert_sp_mat_to_sp_tensor(adj):
    coo = adj.tocoo().astype(np.float32)
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)


if __name__ == '__main__':
    args = parse_args()
    fold = args.fold
    num_epochs = args.epochs
    batch_size = args.batch_size
    mf_dim = args.num_mf
    mlp_dim = eval(args.num_mlp)
    mlp_dim_drugSim = eval(args.num_mlp_drugSim)
    mlp_dim_diseaseSim = eval(args.num_mlp_diseaseSim)
    reg_mf = args.reg_mf
    reg_mlp = eval(args.reg_mlp)
    emb_dim = args.emb_size
    weight_size = eval(args.layer_size)
    num_negatives = args.num_neg
    learning_rate = args.lr
    verbose = args.verbose
    topK = args.topK
    print("NeuMF arguments: %s " % (args))

    # Loading data
    dataset = Dataset(args.dataset, fold)
    posMat, train = dataset.posMatrix, dataset.trainMatrix
    num_users, num_items = train.shape
    testRatings, testNegatives = dataset.testRatings, dataset.testNegatives
    print("Load data done. #drug=%d, #disease=%d" % (num_users, num_items))

    config = dict()
    config['num_users'] = num_users
    config['num_items'] = num_items

    drugSimEmbed = np.loadtxt('../data/drugSimEmbedSorted.txt', dtype=float)
    diseaseSimEmbed = np.loadtxt('../data/diseaseSimEmbedSorted.txt', dtype=float)

    norm_adj = get_adj_mat(args.dataset, fold, num_users, num_items)
    config['norm_adj'] = norm_adj
    print('use the normalized adjacency matrix')

    user_input, item_input, labels = get_train_instances(posMat, train, num_negatives)

    batch_index = []
    for i in range(len(user_input)):
        if i % batch_size == 0:
            batch_index.append(i)
    batch_index.append(len(user_input))

    # Input layer
    train_input_user = tf.placeholder(tf.int32, shape=(None,))
    train_input_item = tf.placeholder(tf.int32, shape=(None,))
    train_input_user_sim = tf.placeholder(tf.float32, shape=(None, 128))
    train_input_item_sim = tf.placeholder(tf.float32, shape=(None, 128))
    y = tf.placeholder(tf.float32, shape=(None,))
    yo = tf.reshape(y, [-1, 1])

    # DDI GCN weights
    gcn_weights = init_weights(num_users, num_items, emb_dim, weight_size)

    # DDI GCN embedding
    ua_embeddings, ia_embeddings = create_lightgcn_embed(num_users, num_items, gcn_weights, config['norm_adj'], weight_size)
    u_g_embeddings = tf.nn.embedding_lookup(ua_embeddings, train_input_user)
    i_g_embeddings = tf.nn.embedding_lookup(ia_embeddings, train_input_item)

    gcn_embed_inner = tf.multiply(u_g_embeddings, i_g_embeddings)

    # NCF embedding layer
    mlp_embedding_user = tf.Variable(tf.random_normal(shape=[num_users, int(mlp_dim[0]/2)], mean=0.0, stddev=0.01))
    mlp_embedding_item = tf.Variable(tf.random_normal(shape=[num_items, int(mlp_dim[0]/2)], mean=0.0, stddev=0.01))
    gmf_embedding_user = tf.Variable(tf.random_normal(shape=[num_users, mf_dim], mean=0.0, stddev=0.01))
    gmf_embedding_item = tf.Variable(tf.random_normal(shape=[num_items, mf_dim], mean=0.0, stddev=0.01))

    mlp_embed_user = tf.nn.embedding_lookup(mlp_embedding_user, train_input_user)
    mlp_embed_item = tf.nn.embedding_lookup(mlp_embedding_item, train_input_item)
    gmf_embed_user = tf.nn.embedding_lookup(gmf_embedding_user, train_input_user)
    gmf_embed_item = tf.nn.embedding_lookup(gmf_embedding_item, train_input_item)

    # MLP part
    embed_concat = tf.concat([mlp_embed_user, mlp_embed_item], 1)
    w = []
    hidden = []
    for i in range(len(mlp_dim)-1):
        w.append(tf.Variable(tf.random_normal(shape=[mlp_dim[i], mlp_dim[i+1]], mean=0.0, stddev=0.01)))
        if i == 0:
            hidden.append(tf.nn.relu(tf.matmul(embed_concat, w[i])))
        else:
            hidden.append(tf.nn.relu(tf.matmul(hidden[i-1], w[i])))
    mlp_dense = hidden[-1]

    # MLP sim part
    w_1 = []
    hidden_1 = []
    for i in range(len(mlp_dim_drugSim) - 1):
        w_1.append(tf.Variable(tf.random_normal(shape=[mlp_dim_drugSim[i], mlp_dim_drugSim[i + 1]], mean=0.0, stddev=0.01)))
        if i == 0:
            hidden_1.append(tf.nn.relu(tf.matmul(train_input_user_sim, w_1[i])))
        else:
            hidden_1.append(tf.nn.relu(tf.matmul(hidden_1[i - 1], w_1[i])))
    mlp_dense_1 = hidden_1[-1]

    w_2 = []
    hidden_2 = []
    for i in range(len(mlp_dim_diseaseSim) - 1):
        w_2.append(tf.Variable(tf.random_normal(shape=[mlp_dim_diseaseSim[i], mlp_dim_diseaseSim[i + 1]], mean=0.0, stddev=0.01)))
        if i == 0:
            hidden_2.append(tf.nn.relu(tf.matmul(train_input_item_sim, w_2[i])))
        else:
            hidden_2.append(tf.nn.relu(tf.matmul(hidden_2[i - 1], w_2[i])))
    mlp_dense_2 = hidden_2[-1]
    sim_embed_inner = tf.multiply(mlp_dense_1, mlp_dense_2)

    # MF part
    gmf_dense = tf.multiply(gmf_embed_user, gmf_embed_item)

    neu_dense = tf.concat([gmf_dense, mlp_dense, sim_embed_inner, gcn_embed_inner], 1)
    w_neu = tf.Variable(tf.random_normal(shape=[4*mf_dim, 1], mean=0.0, stddev=0.01))

    # Final prediction layer
    predict = tf.nn.sigmoid(tf.matmul(neu_dense, w_neu))

    loss = -tf.reduce_mean(yo*tf.log(tf.clip_by_value(predict, 1e-10, 1.0)) + (1-yo)*tf.log(tf.clip_by_value(1-predict, 1e-10, 1.0)))
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    best_auc = 0
    best_tpr = 0
    best_fpr = 0
    best_aupr = 0
    best_precision = 0
    best_recall = 0
    best_hr = 0
    best_epoch = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        epochs = num_epochs
        for epoch in range(epochs):
            user_input, item_input, labels = get_train_instances(posMat, train, num_negatives)
            user_sim = []
            item_sim = []
            for i in user_input:
                user_sim.append(drugSimEmbed[i])
            for j in item_input:
                item_sim.append(diseaseSimEmbed[j])
            for step in range(len(batch_index)-1):
                t, l = sess.run([train_step, loss], feed_dict={train_input_user: user_input[batch_index[step]: batch_index[step+1]], train_input_user_sim: user_sim[batch_index[step]: batch_index[step+1]], train_input_item: item_input[batch_index[step]: batch_index[step+1]], train_input_item_sim: item_sim[batch_index[step]: batch_index[step+1]], y: labels[batch_index[step]: batch_index[step+1]]})
                # Evaluation
                if epoch % verbose == 0:
                    hits, auc, fpr, tpr, aupr, precision, recall = evaluate_model(sess, predict, train_input_user, train_input_item, train_input_user_sim, train_input_item_sim, drugSimEmbed, diseaseSimEmbed, testRatings, testNegatives, topK)
                    hr = np.array(hits).mean()
                    print('Iteration %d: auc = %.4f, hit = %.4f,  aupr = %.4f, loss = %.4f' % (epoch, auc, hr, aupr, l))
                    if auc > best_auc:
                        best_auc = auc
                        best_tpr = tpr
                        best_fpr = fpr
                        best_epoch = epoch
                    if aupr > best_aupr:
                        best_aupr = aupr
                        best_precision = precision
                        best_recall = recall
                    if hr > best_hr:
                        best_hr = hr
        print("End the model. Best epoch %d:  auc = %.4f, hit = %.4f, aupr = %.4f." % (best_epoch, best_auc, best_hr, best_aupr))