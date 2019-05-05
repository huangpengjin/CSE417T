function [ train_err, test_err ] = AdaBoost2( X_tr, y_tr, X_te, y_te, n_trees )
%AdaBoost: Implement AdaBoost using decision stumps learned
%   using information gain as the weak learners.
%   X_tr: Training set
%   y_tr: Training set labels
%   X_te: Testing set
%   y_te: Testing set labels
%   n_trees: The number of trees to use
% 

[m,n] = size(X_tr);
[m1,n1] = size(X_te);
y_tr(find(y_tr == min(y_tr))) = -1;
y_tr(find(y_tr == max(y_tr))) = 1;
y_te(find(y_te == min(y_te))) = -1;
y_te(find(y_te == max(y_te))) = 1;
w=ones(m,1)*(1.0/m);
H_te=ones(m,1);
H_tr=ones(m1,1);
train_err=zeros(n_trees,1);
test_err=zeros(n_trees,1);

for i = 1:n_trees
    tree = fitctree(X_tr, y_tr, 'SplitCriterion', 'deviance', 'Weights', w, 'MaxNumSplits',1);
    h=predict(tree, X_tr);
    er=sum(w(find(h ~= y_tr)));
    alpha=log((1-er)/er)/2;
    w(find(h == y_tr))=w(find(h == y_tr))*exp(-alpha);
    w(find(h ~= y_tr))=w(find(h ~= y_tr))*exp(alpha);
    w=w/sum(w);
    
    temp=alpha*h;
    if i==1
        H_tr = temp;
    else
        H_tr = H_tr + temp;
    end
    H = sign(H_tr);
    train_err(i) = mean(H ~= sign(y_tr));
    
    temp=alpha*predict(tree, X_te);
    if i==1
        H_te = temp;
    else
        H_te = H_te + temp;
    end
    H = sign(H_te);
    test_err(i) = mean(H ~= sign(y_te));
end

end

