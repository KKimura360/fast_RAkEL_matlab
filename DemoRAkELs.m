
% test experiments 
addpath('function')
addpath('evaluation')

%loading dataset
dataset='CAL500';
load(['data/',dataset,'.mat']);

%Set seed
rng('default');

% 5-CV
num_fold = 5;
indices = crossvalind('Kfold',size(data,1),num_fold);
data=sparse(data);
numL=size(target,1);

% parameters
param.k=3;
param.m=2*numL;
param.a=0.1;
param.b=0.3; % in most of all datasets
param.ops='-s 0 -q'; % options for liblinear, -C is not tuned this demo
param.labelSet=[];

for fold= 1:num_fold
 

    test = (indices == fold); train = ~test; 
    %Separate dataset into training and test
    data=sparse(data);
    train_data=data(train,:);
    test_data=data(test,:);
    train_target=target(:,train')';
    test_target=target(:,test')';
    
    % conduct fRAkEL 
    [Res1]=fRakel(train_data,train_target,test_data,param);
    param.labelSet=Res1.labelSet;
    % conduct RAkEL with the same label subsets
    [Res2]=Rakel(train_data,train_target,test_data,param);
    param.labelSet=[];
 
    %threshold for majority voting is not tuned in this demo
    tmp_time=cumsum(Res1.time);
    frakel_time=tmp_time(end)+Res1.regtime;
    tmp_time=cumsum(Res2.time);
    rakel_time=tmp_time(end);
  
    frakel_result=Eval_F1(Res1.Yhat',test_target');
    rakel_result=Eval_F1(Res2.Yhat',test_target');
    
    fprintf('Fold %d,fRAkEL: max Macro-F1 %1.3f max Micro-F1 %1.3f Time %3.2f \n',fold,max(frakel_result(1,:)),max(frakel_result(2,:)),frakel_time);     
    fprintf('Fold %d, RAkEL: max Macro-F1 %1.3f max Micro-F1 %1.3f Time %3.2f \n',fold,max(rakel_result(1,:)),max(rakel_result(2,:)),rakel_time)  
end