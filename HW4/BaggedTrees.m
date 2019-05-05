function [ oobErr ] = BaggedTrees( X, Y, numBags )
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
[row,col] = size(X);
oobIndex = [];
Ysum = zeros(row, 1);
oobErr=zeros(1,numBags);

%%%Create subsets and OOB data
for i = 1:numBags
    labels = Y; 
    features = X; 
    subset = randsample(1:row, row, true);
    OOB_data = setdiff((1:row)', unique(subset));
    
    % subset labels and features
    subset_label = labels(subset,:);
    subset_features = features(subset,:);
    tree = fitctree(subset_features, subset_label);
    
    % Get the OOB features and label
    OOB_features = X(OOB_data, :);
    oobIndex = unique([oobIndex; OOB_data]);
    OOB_label = Y(oobIndex, :);
    
    % Get the predicted labels
    predicted_Label = predict(tree, OOB_features); 
    [pLrow,pLcol]=size(predicted_Label);
    for j=1:pLrow
        if predicted_Label(j)==min(Y)
            predicted_Label(j)=1;
        elseif predicted_Label(j)==max(Y)
            predicted_Label(j)=-1;
        end
    end
    
    % majority vote
    Ysum(OOB_data) = Ysum(OOB_data) + predicted_Label;
    Y_OOB = Ysum(oobIndex);
    [rr,cc]=size(Y_OOB);
    for k=1:rr
        if Y_OOB(k) >= 0
            Y_OOB(k)=min(Y);
        elseif Y_OOB(k)<0
            Y_OOB(k)=max(Y);
        end
    end
    
    % Compute OOB Error
    L=length(oobIndex);
    oobErr(i)=sum(abs(OOB_label - Y_OOB)/(max(Y)-min(Y)))/L;
    
end
end

