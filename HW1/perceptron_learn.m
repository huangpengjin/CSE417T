function [ w iterations ] = perceptron_learn( data_in )
%perceptron_learn Run PLA on the input data
%   Inputs: data_in: Assumed to be a matrix with each row representing an
%                    (x,y) pair, with the x vector augmented with an
%                    initial 1, and the label (y) in the last column
%   Outputs: w: A weight vector (should linearly separate the data if it is
%               linearly separable)
%            iterations: The number of iterations the algorithm ran for
wv=zeros(11,1);
t=0;
flag=false;
while flag==false
    flag=true;
    for i=1:100
        if sign(data_in(i,1:11)*wv)~=data_in(i,12)
        wv=wv+(data_in(i,12).*data_in(i,1:11))';
        t=t+1;
        flag=false;
        end
    end
end
w=wv;
iterations=t;
end