function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

h = @(x) sigmoid(x*theta);
J = (-y' * log(h(X)) - (1 - y)' * log(1 - h(X)))/m + ...
    lambda * sum(theta(2:end) .^ 2)/(2 * m);
grad = (1/m) * ((h(X) - y)' * X)' + (1:size(theta) ~= 1)' .* ...
        (lambda / m) .* theta;

% =============================================================

grad = grad(:);

end
