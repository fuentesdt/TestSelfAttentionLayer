clear all
close all
close(findall(groot, "Type", "figure"));

layers = [
    sequenceInputLayer(12,'Name', 'input')
    selfAttentionLayer(4,12,'Name', 'selfAttn','QueryWeights',rand(12,12), 'KeyWeights',rand(12,12), 'ValueWeights',rand(12,12), 'OutputWeights',rand(12,12), 'QueryBias',rand(12,1), 'KeyBias',rand(12,1), 'ValueBias',rand(12,1), 'OutputBias',rand(12,1))
    layerNormalizationLayer('Offset',rand(12,1),'Scale',zeros(12,1))
    fullyConnectedLayer(9,'Weights',rand(9,12),'Bias',zeros(9,1))
    softmaxLayer
    classificationLayer('Name', 'output')];
multitoken = assembleNetwork(layers)
%analyzeNetwork(multitoken )


myinput = rand(12,1);
X = activations(multitoken ,myinput,'input' );
outputattn = activations(multitoken,myinput,'selfAttn');

Q = multitoken.Layers(2).QueryWeights * X{1} + multitoken.Layers(2).QueryBias;
K = multitoken.Layers(2).KeyWeights   * X{1} + multitoken.Layers(2).KeyBias  ;
V = multitoken.Layers(2).ValueWeights * X{1} + multitoken.Layers(2).ValueBias;
scores = 1/sqrt(12)*Q*K';              
%weights = rowwise_softmax(scores);
%weights = softmax(scores);
max_vals = max(scores, [], 2);
z = scores- max_vals; 
exp_z = exp(z);
sum_exp_z = sum(exp_z, 2);
weights = exp_z ./ sum_exp_z;

Y = weights * V;            
% TODO - this should be zero
myattn = multitoken.Layers(2).OutputWeights * Y + multitoken.Layers(2).OutputBias;
norm(myattn - outputattn{1})

% TODO - this is approximately zero
myattn = multitoken.Layers(2).OutputWeights * V + multitoken.Layers(2).OutputBias;
norm(myattn - outputattn{1})
