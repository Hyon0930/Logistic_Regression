function [J, grad] = gradDes(theta, X, y, lambda)

m = length(y); % number of patient

J = 0;
grad = zeros(size(theta));

J = (1/m) * sum(-y' * log(sigmoid(X * theta)) ...
    - (1-y)' * log(1-sigmoid(X * theta))) ...
    + lambda * sum(theta(2:end).^2)/ (2*m);
%regularization term to prevent overfit.

grad(1) = (1/m) * X(:,1)' * (sigmoid(X * theta) - y );


grad(2:end) = (1/m) * X(:,2:end)' * (sigmoid(X * theta) - y ) ...
               + lambda * theta(2:end) / m;  

grad = grad(:);

end
