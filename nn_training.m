function Status = nn_training()
File=load('speech');
%==================Speech Disorder======================================%
%----(1)----------Frequency--------------

[NNStruct_categories,NNTr_categories] = neural_network_trainingcat1(File.features,File.categories_nn);
save('nnmodels.mat','NNStruct_categories','-append');
save('nnmodels.mat','NNTr_categories','-append');
%figure, plotperform(NNTr_categories);
%figure, plottrainstate(NNTr_categories);
%==================Speech Emotions======================================%
%----(1)----------Frequency--------------
[NNStruct_emotions,NNTr_emotions]  = neural_network_training(File.features,File.emotions_nn);
save('nnmodels.mat','NNStruct_emotions','-append');
save('nnmodels.mat','NNTr_emotions','-append');
%figure, plotperform(NNTr_emotions);
%figure, plottrainstate(NNTr_emotions);
Status= 'NN Model has been Trained';