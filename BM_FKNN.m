function [predicted, memberships, numhits] = BM_FKNN(Xtrain, Ytrain, Xtest, Ytest, K, p, q, m)

% Bonferroni-mean based fuzzy k-nearest neighbor (BM-FKNN) algorithm
% INPUTS:
    % Xtrain: Train data is a n-by-m data matrix consisting of 
    % n patterns and m features(variables)
    % Ytrain: n dimensional class vector of Xtrain data (class labels should be in numerical form, eg. 1,2)
    % Xtest: Test data is a D-by-m data matrix consisting of D
    % patterns and m features
    % Ytest: D dimensional class vector of Xtest data
    % K: Number of nearest neighbors to be selected
    % p, q: Parameter values for Bonferroni mean operator
    % m: Scaling factor for fuzzy weights

% OUTPUTS:
    % predicted: Predicted class label for each test pattern in Xtest
    % memberships: Fuzzy class memberships values for each test pattern in Xtest
    % numhits: Number of correctly predicted test patterns
    
% 'Bonferroni_mean.m' is needed. 
% This file is needed to compute Bonferroni mean vectors of the set
% of nearest neighbor in each class

% Reference:
    % Kumbure, M.M., Luukka,P.& Collan M.(2020) A new fuzzy k-nearest neighbor classifier based on 
    % the Bonferroni mean. Pattern Recognition Letters, 140, 172-178.

% Created by Mahinda Mailagaha Kumbure & Pasi Luukka, 10/2020 
% Based on Keller's definition of the fuzzy k-nearest neighbor algorithm.

%============================================================================================================

% Start

% Set the default value for m 
if nargin < 8
    m = 2;
end

num_train = size(Xtrain,1); % Find the number of patterns in the train set
num_test  = size(Xtest,1);  % Find the number of patterns in the test set

max_class = max(Ytrain); 

% Initialization
predicted = zeros(num_test,1);
memberships = zeros(num_test, max_class, 1);
numhits = 0;


% For each test pattern, do:

for i=1:num_test
    % Computer the Euclidean distances from test patters to each train
    % pattern, (for efficiency, no need to take sqrt since it is a
    % non-decreasing function)

    distances        = (repmat(Xtest(i,:), num_train,1) - Xtrain).^2;
    distances        = sum(distances,2)';

    [~, indices]     = sort(distances);     % Sort the distances
    neighbor_index   = indices(1:K);  % Find the indices of nearest neighbors

    Xneighbors       = Xtrain(neighbor_index,:); % Set of nearest neighbors
    Yneighbors       = Ytrain(neighbor_index);   % Class labels of the nearest neighbors
    Yneighbors_class = unique(Yneighbors);  % Find the inclused classes in the set of nearest neighbors

    % Initialization of vectors for Bonferroni mean vectors and corresponding class
    % labels
    bm_vectors       = zeros(length(Yneighbors_class),size(Xneighbors,2));
    Xneighbors_class = zeros(length(Yneighbors_class),1);
    
    % Compute the Bonferroni mean (ĺocal-mean) vectors for the neighbors in
    % each class
    for jj = 1:length(Yneighbors_class)  % Go through each class
        % Take the set of nearest neighbors from class i
        Xtrain_i        = Xneighbors(find(Yneighbors == Yneighbors_class(jj)), :); 
        bm_vectors(jj,:) = Bonferroni_mean(Xtrain_i, p, q); % Bonferroni mean vector
        Xneighbors_class(jj,1) = Yneighbors_class(jj); % Corresponding class for Bonferroni mean vector
     
    end

    [n1, ~] = size(bm_vectors); % Take the number of Bonferroni mean vectors (n1)
    
    % Compute the Euclidean distances from test pattern to local-mean (Bonferroni) vectors found for each class
    distances = (repmat(Xtest(i,:), n1, 1) - bm_vectors).^2;
    distances = sum(distances,2)';    
    
    % Compute fuzzy weights:
 	% Though this weight calculation should be: 
    % weight = distances(neighbor_index).^(-2/(m-1)), 
    % but since we did not take sqrt above and the inverse 
    % 2th power the weights are: weight = sqrt(distances(neighbor_index)).^(-2/(m-1));
	% which is equaliavent to:
    weight = distances.^(-1/(m-1));
    
 	% Set the Inf (infite) weights, if there are any, to  1.
        if max(isinf(weight))
           warning(['Some of the weights are Inf for sample: ' num2str(i) '. These weights are set to 1.']);
           weight(isinf(weight)) = 1;
        end
    
    % Convert class Ytrain to unary membership vectors (of 1s and 0s)
    Ytrain_iter = zeros(length(Xneighbors_class),max_class);
    for ii=1:n1
        Ytrain_iter(ii,:) = [zeros(1, Xneighbors_class(ii)-1) 1 zeros(1,max_class - Xneighbors_class(ii))];
    end    
    
	test_out = weight*Ytrain_iter/(sum(weight));
    memberships(i,:,1) = test_out; 
    
	% Find the predicted class (the one with the max. fuzzy vote)
	[~, index_of_max] = max(test_out');
    predicted(i) = index_of_max;

	% Compute current hit rate, if test labels are given
    if ~isempty(Ytest) && predicted(i)==Ytest(i)
        numhits = numhits + 1;
    end

end
