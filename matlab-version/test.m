S = getfield(load('ml100k_train.mat','ml100k_train'),'ml100k_train');
S = S./ 5;
maxS = 5;
minS = 1;
ST = S';
r = 8;
lr = 0.001;
lamda = 0.001;
maxItr_init = 10;
%%Train MF
[P, Q] = SDMFinit(S, ST, r, lamda, lr, maxItr_init);
%%Kmeans
u_nums = 10; %用户类别个数
i_nums = 10; %物品类别个数
[ulabel_idx, KP] = kmeans(P, u_nums);
[ilabel_idx, KQ] = kmeans(Q, i_nums);

alpha = 0.001;
alpha1 = 0.001;
alpha2 = 0.001;
beta = 0.001;
beta1 = 0.001;
beta2 = 0.001;


[B, D, X, Y] = SDMF(S, P', Q', KP', KQ', ulabel_idx, ilabel_idx,r,alpha,alpha1,alpha2,beta,beta1,beta2);
disp('-----------------Testing-----------------');
R = getfield(load('ml100k_test.mat','ml100k_test'),'ml100k_test');
ndcg = zeros(10,1);
for kvalue = 1:10
    [ndcgvalue] = NDCG(R, B, D, kvalue);
    ndcg(kvalue) = ndcgvalue;
    disp(['The SDMF test_ndcg@',int2str(kvalue),' is ',num2str(ndcgvalue)]);
end
cut_ratings = [5,4,3,2];
cut_offs = [2,4,6,8,10,15,20];
[recall, map] = Recall_Precision(R, B, D, cut_ratings, cut_offs);
disp(recall);
disp(map);