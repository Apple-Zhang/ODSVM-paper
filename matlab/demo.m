clc;
clear;
close;

rng(2025) % random seed for reproducibility

addpath(genpath('utils')); % The utils codes are from Deng Cai.

%% init
trainRate = 0.6;
load('data/dna.scale.mat'); % load data
data = full(data);
n    = size(data,1);

% randomly partition the data into training and test sets
train_ind = false(n,1);
train_ind(randperm(n, round(n*trainRate))) = true;

Xtrain = data(train_ind, :);
Xtest  = data(~train_ind, :);
Ytrain = label(train_ind, :);
Ytest  = label(~train_ind, :);

% hyperparameters
Wopt = struct('NeighborMode', 'Supervised', ...
              'WeightMode', 'heatkernel', ...
              'gnd', Ytrain);
param.G = constructW(Xtrain, Wopt);
param.d = 16;  % dimension
param.c = 0.25; % SVM penalty
param.lambda = 1e3;
param.gamma  = 1e4;

model = mODSVM_train(Ytrain, Xtrain, param);
[~, accu] = mODSVM_predict(Ytest, Xtest, model);
disp(accu(1));

% %% Figures
figure("Position", [500, 500, 1600, 400])

% % data in the original space
yy = tsne(Xtest, 'NumDimensions', 2);
subplot(1,3,1)
gscatter(yy(:,1), yy(:,2), Ytest, [], '.sx', [8, 4, 4], 'off');

% data in the subspace
yy = tsne(Xtest * model.P,'NumDimensions', 2);
subplot(1,3,2)
gscatter(yy(:,1), yy(:,2), Ytest, [], '.sx', [8, 4, 4], 'off');

% reconstruction data
yy = tsne(Xtest * model.P * model.Q','NumDimensions', 2);
subplot(1,3,3)
gscatter(yy(:,1), yy(:,2), Ytest, [], '.sx', [8, 4, 4], 'off');