function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

theta_temp = theta;

theta_temp(1) = 0;

z = X * theta;

g = sigmoid(z);

log_term_1 = (-y .* log(g));

log_term_2 = ((1 - y) .* log(1-g));

reg_term = (lambda / (2*m)) * sum(theta_temp.^2);

%for i = 2:size(theta)(1);
  
%  reg_term = reg_term + (lambda / (2*m)) .*(theta(i))**2;
  

J = (1/m) * sum(log_term_1 - log_term_2) + reg_term;


grad = ( (1/m) * sum((g - y) .* X )) + (lambda/m) .* theta_temp';


%grad(1) = (1/m) * (g - y)' .* X(1);

% =============================================================

end
