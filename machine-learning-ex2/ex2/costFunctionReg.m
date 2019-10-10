function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
h = @(x) sigmoid(x*theta);
J = (-y' * log(h(X)) - (1 - y)' * log(1 - h(X)))/m + ...
    lambda * sum(theta(2:end) .^ 2)/(2 * m);
grad = (1/m) * ((h(X) - y)' * X)' + (1:size(theta) ~= 1)' .* ...
        (lambda / m) .* theta;

% =============================================================

end
