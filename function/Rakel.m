function [Result]=Rakel(train_data,train_target,test_data,param)
%Input
% train_data = N x d matrix
% train_target = N x L matrix
% test_data = Nt x d matrix
% test_targer = Nt x L matrix
% param.m = number of subset for Rakelo
% param.k = number of labels in the subsets for Rakeld and Rakelo
% param.labelSet = [] 
%                = Cell with label indices. use for label selection instead
%                random sampling 

% Initialization for result and some values
[numN numL]=size(train_target);
[numNt,~]=size(test_data);
Summat=zeros(numNt,numL);
Countmat=Summat;
Result.labelSet=cell(param.m,1);


%% Sampling Step
if ~isempty(param.labelSet);
    % Using pre-defined label subset 
    Result.labelSet=param.labelSet;
    param.m=length(Result.labelSet);
  
else
    for i= 1:param.m
        %random sampling 
        Result.labelSet{i}=randsample(numL,param.k);
    end
end

  
%% Main part: Constructing M classifiers and conducting testing M times
for i=1:param.m
    
    tic;
    % Extract ith label subset
    labelSet=Result.labelSet{i};
    % Error check 
    if isempty(labelSet)
        break;
    end
      
           
    % Construct a smalle label matrix  (smaller instances and labels)
    tmp_train_target=train_target(:,labelSet);
    % transforming MLC to MCC (n x L matrix to n x 1 vector) 
    [replabelSet, ~, new_train_target]=unique(tmp_train_target,'rows');
    
   
        % Constructing classifiers
        model = train(new_train_target,train_data,param.ops);
        % testing 
       [pred]=predict(zeros(size(test_data,1),1),test_data,model,'-q');
   
       % keep results as a matrix for majority voting
        % retransforming MCC to MLC ->replabelSet(pred,:)
        % add the result to coresspond parts
        
        Summat(:,labelSet)= Summat(:,labelSet)+replabelSet(pred,:);
        Countmat(:,labelSet)=Countmat(:,labelSet)+1;
        Result.time(i)=toc;
end

% Majority voting
Result.Yhat = Summat ./ Countmat;
% for the labels which have never been selected 
Result.Yhat(isnan(Result.Yhat))=0;