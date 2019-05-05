% Script to load data from zip.train, filter it into datasets with only one
% and three or three and five, and compare the performance of plain
% decision trees (cross-validated) and bagged ensembles (OOB error)
%load zip.train;
load zip.train

fprintf('Working on the one-vs-three problem...\n\n');
subsample = zip(find(zip(:,1)==1 | zip(:,1) == 3),:);
Y = subsample(:,1);
X = subsample(:,2:257);
ct = fitctree(X,Y,'CrossVal','on');
fprintf('The cross-validation error of decision trees is %.4f\n', ct.kfoldLoss);
figure(1)
bee = BaggedTrees(X, Y, 200);
fprintf('The OOB error of 200 bagged decision trees is %.4f\n', bee);
plot((1:200), bee);
xlabel('numBags');
ylabel('OOB Error');
title('training error for 1 vs 3');


fprintf('\nNow working on the three-vs-five problem...\n\n');
subsample = zip(find(zip(:,1)==3 | zip(:,1) == 5),:);
Y = subsample(:,1);
X = subsample(:,2:257);
ct = fitctree(X,Y,'CrossVal','on');
fprintf('The cross-validation error of decision trees is %.4f\n', ct.kfoldLoss);
figure(2)
bee = BaggedTrees(X, Y, 200);
fprintf('The OOB error of 200 bagged decision trees is %.4f\n', bee);
plot((1:200), bee);
xlabel('numBags');
ylabel('OOB Error');
title('training error for 3 vs 5');