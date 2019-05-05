function [ w, e_in, iter] = logistic_reg( X, y, w_init, max_its, eta )
%LOGISTIC_REG Learn logistic regression model using gradient descent
%   Inputs:
%       X : data matrix (without an initial column of 1s)
%       y : data labels (plus or minus 1)
%       w_init: initial value of the w vector (d+1 dimensional)
%       max_its: maximum number of iterations to run for
%       eta: learning rate
    
%   Outputs:
%       w : weight vector
%       e_in : in-sample error (as defined in LFD)
[row, column]=size(X);
term=zeros(row,column);
y=2*y-1;
ite=0;
mag=100000; 
w_temp=w_init;
while mag>0.000001 && ite<max_its
    for i=1:row
    term(i,:)=y(i).*X(i,:)./(1+exp(y(i).*X(i,:)*w_temp));
    end
    gt=(-1/row)*sum(term);
    ite=ite+1;
    mag=max(abs(gt));
    w_temp=w_temp-eta*gt';
end
w=w_temp;
e_in=(1/row)*sum(log(1+exp(-y.*X*w)));
iter=ite;
end

