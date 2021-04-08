import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from scipy.sparse import coo_matrix
"计算用户和物品数目，如果ID从0开始，用户和物品数目为最大ID+1"

def get_mapping(uirs, minVal, maxVal):
	user_item = defaultdict(dict)
	item_user = defaultdict(dict)
	user = defaultdict()
	item = defaultdict()
	for (u, i, r) in uirs:
		user_item[u][i] = float((r - minVal) / (maxVal - minVal)) + 0.01
		item_user[i][u] = float((r - minVal) / (maxVal - minVal)) + 0.01
		if u not in user:
			user[u] = len(user)
		if i not in item:
			item[i] = len(item)
	return user, item, user_item, item_user

def only_get_uir(uirs, minVal, maxVal):
	user_item = defaultdict(dict)
	for (u, i, r) in uirs:
		user_item[u][i] = float((r - minVal) / (maxVal - minVal)) + 0.01
	return user_item

def count_users(uirs, ID_start_zer0 = True):
	max_user_idx = 0
	for (u, _, _) in uirs:
		max_user_idx = max(max_user_idx, u)
	if ID_start_zer0 == True:
		return max_user_idx + 1
	else:
		return max_user_idx


def count_items(uirs, ID_start_zero = True):
	max_item_idx = 0
	for (_, i, _) in uirs:
		max_item_idx = max(max_item_idx, i)
	if ID_start_zero == True:
		return max_item_idx + 1
	else:
		return max_item_idx

def plot_pic(x, y, category):
	assert category in ['Loss', 'NDCG']
	plt.plot(x, y, label=category)
	plt.xlabel("Epoch")
	plt.ylabel(category)
	plt.legend()
	plt.show()
	plt.savefig('/home/lfyuan/DRM/picture/' + category + '.jpg')

def get_DataSet(path, minVal, maxVal):
	with open(path, 'r') as f:
		for index, line in enumerate(f):
			u, i, r = line.strip('\r\n').split(',')
			r = (float(r) - minVal) / (maxVal - minVal) + 0.01
			yield (int(u), int(i), float(r))

def get_matrix(path, user_map, item_map):
	rating_list = []
	rating_list_bin = []
	interact = int(1)
	globalmean = 0.0
	for index, line in enumerate(get_DataSet(path, minVal=1, maxVal=5)):
		user_id, item_id, rating = line
		u = user_map[user_id]
		i = item_map[item_id]
		rating_list.append([u, i, rating])
		rating_list_bin.append([u, i, interact])
		globalmean += rating
	globalmean = globalmean / (len(user_map))
	row = np.array(rating_list)[:, 0]
	col = np.array(rating_list)[:, 1]
	rating = np.array(rating_list)[:, 2]
	rating_matrix = coo_matrix((rating, (row, col)), shape=(len(user_map), len(item_map))).toarray()
	row1 = np.array(rating_list_bin)[:, 0]
	col1 = np.array(rating_list_bin)[:, 1]
	rating1 = np.array(rating_list_bin)[:, 2]
	rating_matrix_bin = coo_matrix((rating1, (row1, col1)), shape=(len(user_map), len(item_map))).toarray()
	return rating_matrix, rating_matrix_bin, globalmean

def K_func(x, y):
	return x if x != 0 else y



