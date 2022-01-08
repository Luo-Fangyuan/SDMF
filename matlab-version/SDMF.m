function [ B, D, X, Y ] = SDMF( S, P, Q, KP, KQ, ulabel_idx, ilabel_idx,r,alpha,alpha1,alpha2,beta,beta1,beta2 )
%UNTITLED3 此处显示有关此函数的摘要
%   此处显示详细说明

B = sign(P); B(B == 0) = 1;
D = sign(Q); D(D == 0) = 1;
X = UpdateSVD(B);
Y = UpdateSVD(D); 
converge = false;
it = 0;
[m, n] = size(S);
maxItr2 = 5;
ST = S';
IDX = (S~=0);
IDXT = IDX';
maxItr = 20;

while ~converge
    B0 = B;
    D0 = D;
    parfor i = 1:m
        d = D(:,IDXT(:,i));
        b = B(:,i);
        pi = P(:, i);
        kpi = KP(:, ulabel_idx(i));
        DCDmex(b,d*d',[d*ScaleScore01(nonzeros(ST(:,i)),r),alpha1 * pi, alpha2 * kpi], alpha*X(:,i),maxItr2);
        B(:,i) = b;
    end
    parfor j = 1:n
        b = B(:,IDX(:,j));
        d = D(:,j);
        qj = Q(:, j);
        kqj = KQ(:, ilabel_idx(j));
        DCDmex(d,b*b',[b*ScaleScore01(nonzeros(S(:,j)),r), beta1*qj, beta2*kqj], beta*Y(:,j),maxItr2);
        D(:,j)=d;
    end
    X = UpdateSVD(B);
    Y = UpdateSVD(D);

    [loss_value] = SDMFobj(S,B,D,X,Y, P, Q, KP, KQ,alpha,alpha1,alpha2,beta,beta1,beta2, ulabel_idx, ilabel_idx);
    disp(['loss value = ',num2str(loss_value)]);
%     disp(['obj value = ',num2str(obj)]);
%     disp(ndcgvalue);
    disp(['SDMF at bit ',int2str(r),' Iteration:',int2str(it)]);

    if it >= maxItr || (sum(sum(B~=B0)) == 0 && sum(sum(D~=D0)) == 0)
        converge = true;
    end
    
    it = it+1;
    
end


end

function [loss_value] = SDMFobj(S,B,D,X,Y, P, Q, KP, KQ,alpha,alpha1,alpha2,beta,beta1,beta2, ulabel_idx, ilabel_idx)
[m,n] = size(S);
loss_u = zeros(1,m);
loss_j = zeros(1,n);
pos_I = S>0;
valid_num = 0;
r = size(B,1);
parfor i = 1:m
    Si = nonzeros(S(i, :));
    if isempty(Si)
        continue;
    end
    valid_num = valid_num + 1;
    loss_u(i) = sum((ScaleScore01(S(i,:),r) - (B(:, i)' * D) .* pos_I(i, :)).^2) + alpha2 * sum((B(:, i)-KP(:, ulabel_idx(i))).^2);
end
parfor j = 1:n
    Sj = nonzeros(S(:, j));
    if isempty(Sj)
        continue;
    end
    loss_j(j) = beta2 * sum((D(:, j)-KQ(:, ilabel_idx(j))).^2)
end
loss_value = sum(loss_u) / valid_num + sum(loss_j) + alpha * norm(B-X,'fro') + beta * norm(D-Y,'fro') + alpha1 * norm(B-P,'fro') + beta1 * norm(D-Q,'fro');
end


