%%  Code for "A deep learning approach to predict the number of k-barriers for intrusion detection over a circular region using wireless sensor networks" 
%%  By Abhilash Singh (Email: abhilash.iiserb@gmail.com) 
%%  Data from https://www.kaggle.com/datasets/abhilashdata/ffannid-intrusion-detection-in-wsns
%%  IF you are using this code then please cite the following paper;
%%  Singh, A., Amutha, J., Nagar, J., & Sharma, S. (2022). A deep learning approach to predict the number of k-barriers for intrusion detection over a circular region using wireless sensor networks. Expert Systems with Applications, 118588.

clc
clear all
data=xlsread('circ_bsm_gu.xlsx');
rng(0) %seed for reproducibility
rand_pos = (randperm(length(data)));
for k = 1:length(data)
    data(k) = data(rand_pos(k));
end
Area=data(:,1);
SensingRange=data(:,2);
Transmissionrange=data(:,3);
No_of_sensor=data(:,4);
X=[Area,SensingRange,Transmissionrange,No_of_sensor];
Y=data(:,end);

x = X';
t = Y';
ts=tic;
% Choose a Training Function
% For a list of all training functions type: help nntrain
% 'trainlm' is usually fastest.
% 'trainbr' takes longer but may be better for challenging problems.
% 'trainscg' uses less memory. Suitable in low memory situations.
trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.

% Create a Fitting Network
hiddenLayerSize = [20 20];
net = fitnet(hiddenLayerSize,trainFcn);

% Setup Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 55/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 30/100;
net.layers{1}.transferFcn = 'tansig'; % hidden layer
net.layers{2}.transferFcn = 'tansig'; % hidden layer
net.layers{3}.transferFcn = 'tansig'; % output layer
% Train the Network
[net,tr] = train(net,x,t);

% Test the Network
y = net(x);
tend=toc(ts);