function J = computeCost(X, y, theta)


m = length(y); % number of training examples

predictions = X * theta;
error = predictions - y;
squareErrors = error .^ 2;

J = (1 / (2 * m)) * sum(squareErrors);


end
