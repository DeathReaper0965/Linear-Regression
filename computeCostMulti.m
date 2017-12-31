function J = computeCostMulti(X, y, theta)

m = length(y); % number of training examples


J = (1/(2 * m) * (X*theta - y)' * (X*theta-y));


end
