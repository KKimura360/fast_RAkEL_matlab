function [Result]=fRakel(train_data,train_target,test_data,param)
%Input
% train_data = N x d matrix
% train_target = N x L matrix
% test_data = Nt x d matrix
% test_targer = Nt x L matrix
% param.m = number of subset for Rakelo
% param.k = number of labels in the subsets for Rakeld and Rakelo
% param.a = ridge parameter
% param.b = thresholds for RM
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
    %Initialization
    U=zeros(numL,param.m);
    for i=1:param.m
         U(Result.labelSet{i},i)=1;
    end
else
    %Initialization
    U=zeros(numL,param.m);
    for i= 1:param.m
        %random sampling 
        Result.labelSet{i}=randsample(numL,param.k);
        % m x L will be used to construct membership matrix Z  
        U(Result.labelSet{i},i)=1;
    end
end

%% Regression Step
tic;
%Construct the target matrix Z
Z = train_target * U;   % N x K matrix
Z(Z>0)=1;
Z(Z<0)=0;

% Ridge regression to assign test instances to meta-labels     
V = ridgereg(Z,train_data,param.a); % F x K matrix
% predict membership vector (matrix)
Z_hat= [ones(size(test_data,1),1),test_data] * V; % N x K matrix 
Z_hat(Z_hat>=param.b)=1;  
Z_hat(Z_hat<param.b)=0;
Result.regtime=toc;
  
%% Main part: Constructing M classifiers and conducting testing M times
for i=1:param.m
    
    tic;
    % Extract ith label subset
    labelSet=Result.labelSet{i};
    % Error check 
    if isempty(labelSet)
        break;
    end
    
    % logical vector represents index of test instances classified to the model 
    test_ind=logical(Z_hat(:,i));
    tmp_test_data=test_data(test_ind,:);
  
    % Skip unnecessary construction
    if sum(test_ind)==0
        Result.time(i)=toc;
         continue;
    end
        
    % logical vector represents index of instances which have at least one
    % label in the label subset
    train_ind= logical(Z(:,i));
    
    % Construct a smaller feature matrix  
    tmp_train_data=train_data(train_ind,:);
    % Construct a smalle label matrix  (smaller instances and labels)
    tmp_train_target=train_target(train_ind,labelSet);
    % transforming MLC to MCC (n x L matrix to n x 1 vector) 
    [replabelSet, ~, new_train_target]=unique(tmp_train_target,'rows');
    
   
    if sum(U(:,i))==1
        % if the size of label subset =1
        pred=ones(size(tmp_test_data,1),1);
    else
        % Constructing classifiers
        model = train(new_train_target,tmp_train_data,param.ops);
        % testing 
       [pred]=predict(zeros(size(tmp_test_data,1),1),tmp_test_data,model,'-q');
    end
        % keep results as a matrix for majority voting
        % retransforming MCC to MLC ->replabelSet(pred,:)
        % add the result to coresspond parts
        
        Summat(test_ind,labelSet)= Summat(test_ind,labelSet)+replabelSet(pred,:);
        Countmat(:,labelSet)=Countmat(:,labelSet)+1;
        Result.time(i)=toc;
end

% Majority voting
Result.Yhat = Summat ./ Countmat;
% for the labels which have never been selected 
Result.Yhat(isnan(Result.Yhat))=0;


