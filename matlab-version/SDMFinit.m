function [ P, Q ] = SDMFinit( S, ST, r, lamda, lr, maxItr_init )
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
[m,n] = size(S);
P = rand(m, r);
Q = rand(n, r);
tol = 1e-5;
converge = false;
it = 1;
pos_I = S>0;
pos_IT = pos_I';
while ~converge
    P0 = P;
    Q0 = Q;
    parfor i = 1:m
        Si = nonzeros(S(i,:));
        if isempty(Si)
            continue;
        end
        err = S(i,:) - (P(i, :) * Q') .* pos_I(i, :);
        P(i, :) = P(i, :) + lr * (sum(err' * P(i, :)) - lamda * P(i, :));
    end
    parfor j = 1:n
        Sj = nonzeros(S(:,j));
        if isempty(Sj)
            continue;
        end
        err = ST(j, :) - (Q(j, :) * P') .* pos_IT(j, :);
        Q(j, :) = Q(j, :) + lr * (sum(err' * Q(j, :)) - lamda * Q(j, :));
    end

    
    disp(['SDMFinit Iteration:',int2str(it-1)]);
    if it >= maxItr_init || max([norm(P-P0,'fro') norm(Q-Q0,'fro')]) < max([m n])*tol
        converge = true;
    end
    
    disp(['obj value = ',num2str(SDMFinitObj(S,P,Q,lamda))]);

    it = it+1;
end
end

function loss_value = SDMFinitObj(S, P, Q, lamda)
[m,~] = size(S);
loss = zeros(1,m);
pos_I = S>0;
valid_num = 0;
parfor i = 1:m
    Si = nonzeros(S(i, :));
    if isempty(Si)
        continue;
    end
    valid_num = valid_num + 1;
    loss(i) = sum((S(i,:) - (P(i, :) * Q') .* pos_I(i, :)).^2);
end
loss_value = sum(loss) / valid_num + lamda * (norm(P,'fro') + norm(Q,'fro'));
end



