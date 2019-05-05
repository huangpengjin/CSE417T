function [ train_set test_set ] = generate_dataset( Q_f, N_train, N_test, sigma )
%GENERATE_DATASET Generate training and test sets for the Legendre
%polynomials example
%   Inputs:
%       Q_f: order of the hypothesis
%       N_train: number of training examples
%       N_test: number of test examples
%       sigma: standard deviation of the stochastic noise
%   Outputs:
%       train_set and test_set are both 2-column matrices in which each row
%       represents an (x,y) pair

xTrain=2*rand(N_train,1)-1;
xTest=2*rand(N_test,1)-1;
aq=normrnd(0,1,Q_f+1,1);
q=0:Q_f;
normalize=sqrt(sum(1./(2*q+1)));
epsilon1=normrnd(0,1,length(xTrain),1);
epsilon2=normrnd(0,1,length(xTest),1);
yTrain=computeLegPoly(length(xTrain),Q_f)*aq*normalize+sigma*epsilon1;
yTest=computeLegPoly(length(xTest),Q_f)*aq*normalize+sigma*epsilon2;
train_set=[xTrain yTrain];
test_set=[xTest yTest];
