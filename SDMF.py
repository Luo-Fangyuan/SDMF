import numpy as np
from collections import defaultdict
from util import get_matrix, K_func, get_DataSet
from NDCG import calDCG_k
from K_Means import get_anchor


def cal_loss(path, B, D, P, Q, binary_train_matrix, u_cluster_label, u_centroids, i_cluster_label, i_centroids, user_map, item_map, n_factors, k_num, alpha1, alpha2, beta1, beta2, lamda):
    loss = 0
    pre_true_dict = defaultdict(list)
    u_token = [0] * len(user_map)
    i_token = [0] * len(item_map)
    rating_num = 0
    for index, line in enumerate(get_DataSet(path, minVal=1, maxVal=5)):
        user_id, item_id, rating = line
        if(user_id in user_map.keys() and item_id in item_map.keys()):
            rating_num += 1
            u = user_map[user_id]
            i = item_map[item_id]
            bu = B[u, :]
            di = D[i, :]
            pred = 0.5 + np.dot(bu, di) / (2 * n_factors)
            pre_true_dict[user_id].append([pred, rating])
            pu = P[u, :]
            qi = Q[i, :]
            group_u = u_centroids[u_cluster_label[u]]
            group_i = i_centroids[i_cluster_label[i]]
            loss += (rating - pred) ** 2
            if(u_token[u] == 0):
                loss += alpha1 * (np.linalg.norm(group_u - bu) ** 2) + alpha2 * (np.linalg.norm(pu - bu) ** 2)
                u_token[u] = 1
            if(i_token[i] == 0):
                loss += beta1 * (np.linalg.norm(group_i - di) ** 2) + beta2 * (np.linalg.norm(qi- di) ** 2)
                i_token[i] = 1
        elif(user_id in user_map.keys() and item_id not in item_map.keys()):
            u = user_map[user_id]
            pred = np.sum(binary_train_matrix[u, :]) / (np.count_nonzero(binary_train_matrix[u, :]))
            pre_true_dict[user_id].append([pred, rating])
        elif(user_id not in user_map.keys() and item_id in item_map.keys()):
            i = item_map[item_id]
            pred = np.sum(binary_train_matrix[:, i]) / (np.count_nonzero(binary_train_matrix[:, i]))
            pre_true_dict[user_id].append([pred, rating])
        else:
            pred = globalmean
            pre_true_dict[user_id].append([pred, rating])
    loss += lamda * (np.linalg.norm(np.sum(B, axis = 1)) ** 2 + np.linalg.norm(np.sum(D, axis = 1)) ** 2)
    ndcg_value = defaultdict()
    for k in k_num:
        ndcg_value[int(k)] = calDCG_k(pre_true_dict, int(k))
    return loss / rating_num, ndcg_value



def train(B, D, P, Q, u_cluster_label, u_centroids, i_cluster_label, i_centroids, train_matrix, binary_train_matrix, user_map, item_map, n_factors, k_num, alpha1, alpha2, beta1, beta2, lamda):
    last_loss = 0
    last_delta_loss = 0
    threshold = 1e-4
    max_epochs = 50
    for epoch in range(max_epochs):
        epoch = epoch + 1
        master_flag = 0
        # print('update users!!!!!!!')
        for user_id in user_map.keys():
            u = user_map[user_id]
            while(1):
                flag = 0
                pu = P[u]
                bu = B[u]
                group_u = u_centroids[u_cluster_label[u]]
                for k in range(n_factors):
                    dk = D[:, k]
                    buk_hat = (np.sum((train_matrix[u, :] - 0.5 - np.dot(D, bu.T) / (2 * n_factors)) * dk * binary_train_matrix[u, :])) / n_factors + np.count_nonzero(binary_train_matrix[u, :]) * bu[k] / 2 + 2 * alpha1 * group_u[k] + 2 * alpha2 * pu[k] - 2 * lamda * np.sum(B[:, k])
                    buk_new = np.sign(K_func(buk_hat, bu[k]))
                    if(bu[k] != buk_new):
                        flag = 1
                        bu[k] = buk_new
                if(flag == 0):
                    break
                B[u, :] = bu
                master_flag = 1
        # print('update items!!!!!!!')
        for item_id in item_map.keys():
            i = item_map[item_id]
            while(1):
                flag = 0
                qi = Q[i]
                di = D[i]
                group_i = i_centroids[i_cluster_label[i]]
                for k in range(n_factors):
                    bk = B[:, k]
                    dik_hat = (np.sum((train_matrix[:, i] - 0.5 - np.dot(B, di.T) / (2 * n_factors)) * bk * binary_train_matrix[:, i])) / n_factors + np.count_nonzero(binary_train_matrix[:, i]) * di[k] / 2 + 2 * beta1 * group_i[k] + 2 * beta2 * qi[k] - 2 * lamda * np.sum(D[:, k])
                    dik_new = np.sign(K_func(dik_hat, di[k]))
                    if(di[k] != dik_new):
                        flag = 1
                        di[k] = dik_new
                if(flag == 0):
                    break
                D[i, :] = di
                master_flag = 1
        # print('cal loss!!!!!!!')
        train_loss, train_ndcg = cal_loss(train_path, B, D, P, Q, binary_train_matrix, u_cluster_label, u_centroids, i_cluster_label, i_centroids, user_map, item_map, n_factors, k_num, alpha1, alpha2, beta1, beta2, lamda)
        valid_loss, valid_ndcg = cal_loss(valid_path, B, D, P, Q, binary_train_matrix, u_cluster_label, u_centroids, i_cluster_label, i_centroids, user_map, item_map, n_factors, k_num, alpha1, alpha2, beta1, beta2, lamda)
        delta_loss = train_loss - last_loss
        print('Epoch=%d, train loss=%.5f, valid_loss=%.5f, delta_loss=%.5f' %(epoch, train_loss, valid_loss, delta_loss))
        for key in train_ndcg.keys():
            print('train_NDCG@' + str(key) + ' = ' + str(train_ndcg[key]), end=' ')
        print('')
        for key in valid_ndcg.keys():
            print('valid_NDCG@' + str(key) + ' = ' + str(valid_ndcg[key]), end=' ')
        print('')
        if(master_flag == 0):
            break
        if (abs(delta_loss) < threshold or abs(delta_loss) == abs(last_delta_loss)):
            break
        if(epoch > max_epochs):
            break
        last_delta_loss = delta_loss
        last_loss = train_loss
    test_loss, test_ndcg = cal_loss(test_path, B, D, P, Q, binary_train_matrix, u_cluster_label, u_centroids, i_cluster_label, i_centroids, user_map, item_map, n_factors, k_num, alpha1, alpha2, beta1, beta2, lamda)
    print('test loss=%.5f' %(test_loss))
    for key in test_ndcg.keys():
        print('test_NDCG@' + str(key) + ' = ' + str(test_ndcg[key]), end=' ')
    print('')


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_factors", type=int, default=8, help='dimension of users and items')
    parser.add_argument("--alpha1", type=float, default=10, help='param for group_wise of users')
    parser.add_argument("--alpha2", type=float, default=10, help='param for point_wise of users')
    parser.add_argument("--beta1", type=float, default=3, help='param for group_wise of items')
    parser.add_argument("--beta2", type=float, default=5, help='param for point_wise of items')
    parser.add_argument("--lamda", type=float, default=0.01, help='param for regularization')
    parser.add_argument("--cluster_num_u", type=int, default=50, help='the cluster num for users')
    parser.add_argument("--cluster_num_i", type=int, default=100, help='the cluster num for items')

    args = parser.parse_args()

    n_factors = args.n_factors
    alpha1 = args.alpha1
    alpha2 = args.alpha2
    beta1 = args.beta1
    beta2 = args.beta2
    lamda = args.lamda
    cluster_num_u = args.cluster_num_u
    cluster_num_i = args.cluster_num_i

    k_num = ['4']
    str_k = '-'.join(k_num)
    Dataset = 'ML100K'

    P_path = 'data/'+Dataset+'/P_Q/P_'+str(n_factors)+'_'+str_k+'.csv'
    Q_path = 'data/'+Dataset+'/P_Q/Q_'+str(n_factors)+'_'+str_k+'.csv'

    P = np.loadtxt(P_path, delimiter=',', usecols=(range(1, n_factors+1)))
    Q = np.loadtxt(Q_path, delimiter=',', usecols=(range(1, n_factors+1)))

    u_cluster_label, u_centroids, i_cluster_label, i_centroids = get_anchor(P, Q, cluster_num_u, cluster_num_i)

    B = np.sign(P)
    D = np.sign(Q)
    user_list = np.loadtxt(P_path, delimiter=',')[:, 0]
    item_list = np.loadtxt(Q_path, delimiter=',')[:, 0]
    u_dict = dict(zip(user_list, list(range(len(user_list)))))
    i_dict = dict(zip(item_list, list(range(len(item_list)))))
    u_inv_dict = dict(zip(list(range(len(user_list))), user_list))
    i_inv_dict = dict(zip(list(range(len(item_list))), item_list))


    train_path = 'data/'+Dataset+'/'+Dataset+'_train.txt'
    valid_path = 'data/'+Dataset+'/'+Dataset+'_valid.txt'
    test_path = 'data/'+Dataset+'/'+Dataset+'_test.txt'

    train_matrix, binary_train_matrix, globalmean = get_matrix(train_path, u_dict, i_dict)
    train(B, D, P, Q, u_cluster_label, u_centroids, i_cluster_label, i_centroids, train_matrix, binary_train_matrix, u_dict, i_dict, n_factors, k_num, alpha1, alpha2, beta1, beta2, lamda)









