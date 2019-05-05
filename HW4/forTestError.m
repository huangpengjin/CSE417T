train=load('zip.train');
[row,col]=size(train);
subset = train((train(:,1)==3 | train(:,1) == 5),:);
[rtrain,ctrain]=size(subset);
sub_X = subset(:,2:ctrain);
sub_Y = subset(:,1);

test=load('zip.test');
subtest=test((test(:,1)==3 | test(:,1) == 5),:);
[rrtest,cctest]=size(subtest);
sub1_X = subtest(:,2:cctest);
sub1_Y = subtest(:,1);

%single tree
Traintree=fitctree(sub_X,sub_Y);
ytest=predict(Traintree,sub1_X);
Error=sum((abs(sub1_Y-ytest))/(5-3));
test_error=Error/rrtest;

%Ensemble of 20 trees
Ysum=zeros(rrtest,1);
for m=1:200
i=randsample(rtrain,rtrain,true); %generate numbers from 1 to N,replace.
T=sub_X(i,:);     %generate training set T
Ysub2=sub_Y(i,:);
traintree=fitctree(T,Ysub2);

Ytest=predict(traintree,sub1_X);
Ytest(Ytest==3)=1;
Ytest(Ytest==5)=-1;

Ysum=Ysum+Ytest;
Ysum1=Ysum;
Ysum1(Ysum1>=0)=3;
Ysum1(Ysum1<0)=5;

error=sum((abs(sub1_Y-Ysum1))/(5-3));
Test_error=error/rrtest;

end
