function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%
[m n] = size(X);

% Create a matrix repeating every example k times and subtract the set of 
% centroids from each k-sized set of repeated examples
diff = reshape(repmat(X(:)',K,1)(:),m*K,n) - repmat(centroids,m,1); 

diff = norm(diff,'rows').^2; % Find norm for every k for every m Should be (k*m) x 1

% Reshape into a matrix with m rows and k columns containing norms for k centroids
diff = reshape(diff,K,m)';
% Get indices of minima
[trash idx] = min(diff');
idx = idx';


% Unvectorized implementation
% for i=1:size(X,1)
%     diff = X(i,:) - centroids; % Should give a Kxn matrix
%     [trash idx(i)] = min(norm(diff,'rows').^2);
% end




% =============================================================

end

