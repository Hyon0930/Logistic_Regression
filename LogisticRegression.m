clear ; close all; clc

Data = dlmread('finalCleanedTrainData.csv'); % training data
[DataR,DataC] = size(Data);
disp("DataR = ");disp(DataR);
disp("DataC = ");disp(DataC);

trainData = Data(:, 1:end-1); %exclude the results in the original training data
[trainR,trainC] = size(trainData);
disp("trainR = ");disp(trainR);
disp("trainC = ");disp(trainC);

%parameter set up--------------------------------------------------------------------------------
 
label_vec = Data(:,DataC); %result of trainign set
num_labels = 3;   %final number of classification
%1 => <30
%2 => >30
%3 => no

lambda = 0.01; %for regularization to prevent overfit
all_theta = zeros(num_labels, trainC + 1);


%check if mean normalization works
%meanTrain = mean(trainData);
%maxTrain = max(trainData);
%minTrain = min(trainData);
%rangeTrain = (maxTrain-minTrain);
%
%preMat = bsxfun(@minus, trainData, meanTrain);
%trainData = bsxfun(@rdivide, preMat, rangeTrain);

%create matrix traindata with theta for gradient descent
temp = [ones(trainR, 1) trainData];
initial_theta = zeros(trainC+1, 1);


%Gradient descent computation to find the minimun value cost function.----------------------------------
options = optimset('GradObj', 'on', 'MaxIter', 100);
disp("Training multi classification logistic regression...");

%traininf for each label for readmission
for c = 1:num_labels 
[all_theta(c,:)] = ...
         fminunc (@(t)(gradDes(t, temp, (label_vec == c), lambda)), initial_theta, options);
end
[thetaR, thetaC] = size(all_theta)


%check accuracy of the cost function agaiinst the training set.-----------------------------------------
%m = size(X, 1);

p = zeros(size(trainData , 1), 1);

checkTrainData = [ones(trainR, 1) trainData];

P = zeros(size(trainR,num_labels));
P = sigmoid(checkTrainData * all_theta');
%find max value for each cost function for classification.
[maxValue, rowIdx] = max(P, [], 2);
%disp("rowIdx = ");disp(rowIdx);
exportData = [maxValue, rowIdx,label_vec];

csvwrite("checkResult.csv", exportData);
p = rowIdx;

%print overall probability to get the answer
fprintf('\nTraining Set Accuracy: %f\n', mean(double(p == label_vec)) * 100);

%Prediction based on the test set.-----------------------------------------------------------------------

testData = dlmread('cleanedTest.csv');
[testR, testC] = size(testData);
disp("testR = ");disp(testR);
disp("testC = ");disp(testC);

p = zeros(size(testData , 1), 1);

checkTestData = [ones(testR, 1) testData];

P = zeros(size(testR,num_labels));
P = sigmoid(checkTestData * all_theta');
%find max value for each cost function for classification.
[testMaxValue, testRowIdx] = max(P, [], 2);
testExportData = testRowIdx;

csvwrite("MSD_FT_HyonKim.csv", testExportData);









