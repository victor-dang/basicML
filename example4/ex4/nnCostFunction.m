function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
I = eye(num_labels);
Y = zeros(m,num_labels);
for i=1:m
	Y(i,:)=I(y(i),:);
end
%X = [ones(m, 1) X];
activations1 = [ones(m, 1) X];
z2 = activations1*Theta1';
activations2 = sigmoid(z2);
activations2 =[ones(m,1) activations2];
z3 = activations2*Theta2';
activations3 = sigmoid(z3);

J = (1/m)*sum(sum((-Y).*log(activations3)-(1-Y).*log(1-activations3),2));
regularization = lambda/(2*m)*(sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));
J = J + regularization;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
delta3 = activations3 - Y;
delta2 = (delta3*Theta2.*sigmoidGradient([ones(size(z2,1),1) z2]))(:,2:end);
%delta2 = delta2(2:end);
%delta2 = 1/m.*delta2;
%Delta_1 = zeros(hidden_layer_size,1)';
%Delta_2 = zeros(num_labels,1)';
%for t = 1:m 
	% fprintf('\n iter: %d',t);
	% a1(t,:) = [1 X(t,:)];
	% %fprintf('\n a1: %d \n',a1);
	% z2(t,:) = a1(t,:)*Theta1';
	% a2(t,:) = [1 sigmoid(z2(t,:))];
	% z3(t,:) = a2(t,:)*Theta2';
	% a3(t,:) = sigmoid(z3(t,:));
	% deta3(t,:) = a3(t,:) - Y(t,:);
	% deta2(t,:) = deta3(t,:)*Theta2.*sigmoidGradient([1 z2(t,:)]);
	% deta2 = deta2(t,2:end);
	% fprintf('\n Delta_1: %d deta2: %d, a1: %d\nwhos',size(Delta_1,1),size(delta2,1),size(a1,1));
	% Delta_1(t,:) = Delta_1(t,:)+deta2(t,:)*a1(t,:);
	% Delta_2(t,:) = Delta_2(t,:)+deta3(t,:)*a2(t,:);
%end
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Delta_1 = delta2'*activations1;
Delta_2 = delta3'*activations2;


Theta1_grad = Delta_1./m + (lambda/m)*[zeros(size(Theta1,1), 1) Theta1(:, 2:end)];
Theta2_grad = Delta_2./m + (lambda/m)*[zeros(size(Theta2,1), 1) Theta2(:, 2:end)];
grad = [Theta1_grad(:) ; Theta2_grad(:)];

















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
