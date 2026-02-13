clear all
close all
close(findall(groot, "Type", "figure"));

embedDim =7;
mylayers = [
    image3dInputLayer([8 8 2 embedDim],'Name','input','Mean',zeros(8,8,2,embedDim))
    flattenLayer('Name', 'flatten')
    selfAttentionLayer(1,embedDim,'Name','selfAttn','QueryWeights',rand(embedDim,896), 'KeyWeights',rand(embedDim,896), 'ValueWeights',rand(embedDim,896), 'OutputWeights',rand(896,embedDim), 'QueryBias',rand(embedDim,1), 'KeyBias',rand(embedDim,1), 'ValueBias',rand(embedDim,1), 'OutputBias',rand(896,1))
    fullyConnectedLayer(2,'Name','fc','Weights',rand(2,896),'Bias',zeros(2,1))
    softmaxLayer
    classificationLayer('Name', 'output')];
assembledNet = assembleNetwork(mylayers)
%analyzeNetwork(assembledNet)

myinput = rand(8,8,2,embedDim);
X = activations(assembledNet ,myinput,'flatten' );
outputattn = activations(assembledNet,myinput,'selfAttn');

Q = assembledNet.Layers(3).QueryWeights * X + assembledNet.Layers(3).QueryBias;
K = assembledNet.Layers(3).KeyWeights   * X + assembledNet.Layers(3).KeyBias  ;
V = assembledNet.Layers(3).ValueWeights * X + assembledNet.Layers(3).ValueBias;
scores = 1/sqrt(embedDim)*Q*K';              
%weights = rowwise_softmax(scores);
%weights = softmax(scores);
max_vals = max(scores, [], 2);
z = scores- max_vals; 
exp_z = exp(z);
sum_exp_z = sum(exp_z, 2);
weights = exp_z ./ sum_exp_z;

Y = weights * V;            
% TODO - this should be zero
myattn = assembledNet.Layers(3).OutputWeights * Y + assembledNet.Layers(3).OutputBias;
norm(myattn - outputattn)

% TODO - this is approximately zero
myattn = assembledNet.Layers(3).OutputWeights * V + assembledNet.Layers(3).OutputBias;
norm(myattn - outputattn)
