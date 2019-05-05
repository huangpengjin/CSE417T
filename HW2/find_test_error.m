function [ test_error ] = find_test_error( w, X, y )
%FIND_TEST_ERROR Find the test error of a linear separator
%   This function takes as inputs the weight vector representing a linear
%   separator (w), the test examples in matrix form with each row
%   representing an example (X), and the labels for the test data as a
%   column vector (y). X does not have a column of 1s as input, so that 
%   should be added. The labels are assumed to be plus or minus one. 
%   The function returns the error on the test examples as a fraction. The
%   hypothesis is assumed to be of the form (sign ( [1 x(n,:)] * w )
[row,column]=size(X);
w=[0;w];
h=sign([ones(row,1) X]*w);
y=2*y-1;
n=0;
for i=1:row
    if h(i)~=y(i)
        n=n+1;
    end
end
test_error=n/row;
end

