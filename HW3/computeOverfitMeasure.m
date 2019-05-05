function [ overfit_m ] = computeOverfitMeasure( true_Q_f, N_train, N_test, var, num_expts )
%COMPUTEOVERFITMEASURE Compute how much worse H_10 is compared with H_2 in
%terms of test error. Negative number means it's better.
%   Inputs
%       true_Q_f: order of the true hypothesis
%       N_train: number of training examples
%       N_test: number of test examples
%       var: variance of the stochastic noise
%       num_expts: number of times to run the experiment
%   Output
%       overfit_m: vector of length num_expts, reporting each of the
%                  differences in error between H_10 and H_2
Eout_g2=zeros(num_expts,1);
Eout_g10=zeros(num_expts,1);
for i=1:num_expts
    [ train_set test_set ] = generate_dataset( true_Q_f, N_train, N_test, sqrt(var) );
    w2=glmfit(computeLegPoly(train_set(:,1),2),train_set(:,2),'normal','constant','off');
    w10=glmfit(computeLegPoly(train_set(:,1),10),train_set(:,2),'normal','constant','off');
    g2=computeLegPoly(test_set(:,1),2)*w2;
    g10=computeLegPoly(test_set(:,1),10)*w10;
    Eout_g2(i)=mean((g2-test_set(:,2)).^2);
    Eout_g10(i)=mean((g10-test_set(:,2)).^2);
end
overfit_m=Eout_g10-Eout_g2;
end