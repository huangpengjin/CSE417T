function [ ooberr ] = BaggedTrees( X, Y, numBags )
%BAGGEDTREES Returns out-of-bag classification error of an ensemble of
%numBags CART decision trees on the input dataset, and also plots the error
%as a function of the number of bags from 1 to numBags
%   Inputs:
%       X : Matrix of training data
%       Y : Vector of classes of the training examples
%       numBags : Number of trees to learn in the ensemble
%
%   You may use "fitctree" but do not use "TreeBagger" or any other inbuilt
%   bagging function


[row_n,col_n] = size(X); %training_data);


% Main for loop for calculating out-of-bag error.
out_begindexround = [];
l_num_sum = zeros(row_n, 1);

for i = 1:numbags
    % Get subsample and the out-of-bag data
    % subsample
    labels = Y; 
    features = X; 
    randindex = randsample(row_n, row_n, true);
    sorted_labels = sort(labels);
    l_1 = sorted_labels(1, :);
    l_2 = sorted_labels(end, :);
    
    % out-of-bag data
    out_bagindex = setdiff((1:row_n)', unique(randindex));
    
    % Get the training data
    sub_sample_label = labels(randindex,:);
    sub_sample_features = features(randindex,:);
    
    
    % Build the Tree
    decision_tree = fitctree(sub_sample_features, sub_sample_label);
    
    % Get the out-of-bag data and label
    out_bagfeatures = X(out_bagindex, :);
    out_baglabel = Y(out_begindexround, :);
    out_begindexround = unique([out_begindexround; out_bagindex]);
    length_out_bagindex = length(out_begindexround);
    
    % Get the predicted labels
    l_pred = predict(decision_tree, out_bagfeatures);
    l_pred(l_pred==l_1) = 1;
    l_pred(l_pred==l_2) = -1;
    
    % Vote for majority
    l_num_sum(out_bagindex) = l_num_sum(out_bagindex) + l_pred;
    
    final_l = l_num_sum(out_begindexround);
    final_l(final_l >= 0) = l_1;
    final_l(final_l < 0) = l_2;
    
    % Count the number of different labels
    ooberr(i)=sum(abs(out_baglabel - final_l)/2)/length_out_bagindex;
    
end
    
    plot((1:numbags), ooberr);
    xlabel('number of bag');
    ylabel('out-of-bag error');
    title('numBags VS. out-of-bag error');

end
