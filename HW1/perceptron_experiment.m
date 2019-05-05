function [ num_iters bounds] = perceptron_experiment ( N, d, num_samples )
%perceptron_experiment Code for running the perceptron experiment in HW1
%   Inputs: N is the number of training examples
%           d is the dimensionality of each example (before adding the 1)
%           num_samples is the number of times to repeat the experiment
%   Outputs: num_iters is the # of iterations PLA takes for each sample
%            bound_minus_ni is the difference between the theoretical bound
%               and the actual number of iterations
%      (both the outputs should be num_samples long)
bounds=zeros(1,num_samples);
iteration=zeros(1,num_samples);
difference=zeros(1,num_samples);
w=[0;rand(d,1)];
for j=1:num_samples
    x=[ones(1,N); -1+2*rand(d,N)];
    y=sign(w'*x);
    b=x';
    p=min(y.*(w'*x));
    a=zeros(N,1);
    for i=1:N
    a(i)=norm(b(i,:));
    end
    R=max(a);
    bounds(j)=R^2*(norm(w))^2/p^2;
    data_in=[x' y'];
    [weight,iteration(j)]=perceptron_learn(data_in);
    difference(j)=bounds(j)-iteration(j);
end
num_iters=iteration
bounds
figure(1)
histogram(num_iters)
xlabel('No. of Iteration')
ylabel('Times')

figure(2)
histogram(log10(difference))
xlabel('log10(difference)')
ylabel('Times')
end