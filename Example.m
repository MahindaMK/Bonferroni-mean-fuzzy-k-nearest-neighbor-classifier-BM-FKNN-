
% An example for the use of BM-FKNN classifier
% Created by Mahinda Mailagaha Kumbure & Pasi Luukka, 10/2020


clear all; close all; clc

% Load the data (example data of ionosphere)
load ionosphere
    % X: features
    % Y: cell array of the class labels (g:good and b:bad)

% Convert class labels to numeric 
Y      = categorical(Y);
labels = zeros(length(Y),1);
labels(Y=='g') = 1;
labels(Y=='b') = 2;

% If the input data contains negative values, then it is possible to get
% Bonferroni mean vectors, including complex values in cases, for example,
% when p=1.5 and q=0.5. To avoid this issue, the data matrix needs to be normalized into 0 and 1 range. 

X = normalize(X,'range');

data = [X labels];

% Cross validation
val = 0.8; % Percentage for holdout validation
cv  = cvpartition(size(data,1),'HoldOut', val);
idx = cv.test;

% Separate to training and test data
Xtrain  = data(~idx,1:end-1); % train data with n patterns and m features
Ytrain  = data(~idx,end); % class labels of train patters 

Xtest   = data(idx,1:end-1); % test data with D patterns and m features
Ytest   = data(idx,end); % class labels of test patterns

K = 10; % Initialization of the number of nearest neighbors
p = 0.5; % Parameter p for Bonferroni mean operator
q = 3; % Parameter p for Bonferroni mean operator
m = 1.5; % Fuzzy strength values

% BM-FKNN function call
[predicted, memberships, numhits] = BM_FKNN(Xtrain, Ytrain, Xtest, Ytest, K, p, q, m);

% Classification accuracy
classification_accuracy  = numhits/length(Xtest)
