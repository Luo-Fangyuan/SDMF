# SDMF
This is our implementation for the paper:

Jun Wu, Fangyuan Luo, Yujia Zhang, Haishuai Wang. Semi-discrete Matrix Factorization. IEEE Intelligent System. 35(5): 73-83 (2020)

**Please cite our paper if you use our codes. Thanks!**

```
@article{Wu20SDMF,
  author    = {Jun Wu and
               Fangyuan Luo and
               Yujia Zhang and
               Haishuai Wang},
  title     = {Semi-discrete Matrix Factorization},
  journal   = {{IEEE} Intell. Syst.},
  volume    = {35},
  number    = {5},
  pages     = {73--83},
  year      = {2020}
}
```

# Environment Settings
python version: 3.7
numpy version: 1.16.4


# Example to run the codes.
python SDMF.py --n_factors 8 --alpha1 10 --alpha2 10 --beta1 3 --beta2 5 --lamda 0.01 --cluster_num_u 50 --cluster_num_i 100

# Dataset
We provide one processed dataset: MovieLens 100K and a part of the real-valued embeddings of users and items from Matrix Factorization.
ML100K_train.txt
* train file
* each line is a training instance: userID, itemID, rating

ML100K_valid.txt
* valid file
* each line is a valid instance: userID, itemID, rating

ML100K_test.txt
* test file
* each line is a test instance: userID, itemID, rating

P_8_4.csv('8' refers to n_factors, '4' refers to the k value of NDCG@k)
* real-valued embedding of users from well-trained matrix factorization
* the first number of each line is the user ID, and the rest is user's real-valued embedding

Q_8_4.csv('8' refers to n_factors, '4' refers to the k value of NDCG@k)
* real-valued embedding of items from well-trained matrix factorization
* the first number of each line is the item ID, and the rest is item's real-valued embedding
