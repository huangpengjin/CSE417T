function [ z ] = computeLegPoly( x, Q )
%COMPUTELEGPOLY Return the Qth order Legendre polynomial of x
%   Inputs:
%       x: vector (or scalar) of reals in [-1, 1]
%       Q: order of the Legendre polynomial to compute
%   Output:
%       z: matrix where each column is the Legendre polynomials of order 0 
%          to Q, evaluated atthe corresponding x value in the input
[row column]=size(x);
if Q==0
    z=ones(row,column);
elseif Q==1
    z=[ones(row,column) x];
else
    z=zeros(row,Q+1);
    z(:,1)=ones(row,column);
    z(:,2)=x;
    q=Q+1;
    for i=3:q
    z(:,i)=(2*i-1)*x.*z(:,i-1)/i-(i-1)*z(:,i-2)/i;
    end
end
end