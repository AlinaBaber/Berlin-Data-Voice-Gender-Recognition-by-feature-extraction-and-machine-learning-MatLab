function [Category,Category_cost,Category_output,Category_error,Emotion,Emotion_cost,Emotions_output,Emotion_error]=neural_network_classification(wav_file)
Features= allfeatures_extraction( wav_file );
MFile= load('nnmodels'); File= load('speech');
%Tr_categories=MFile.NNTr_categories;
%Tr_emotions=MFile.NNTr_emotions;
[rn,col]=find(strcmp(File.FilePath,wav_file));
targets1=transpose(File.categories_nn(col,:));
targets2=transpose(File.emotions_nn(col,:));
%==================Speech Disorder======================================%
% Test the Network
Features=transpose(Features);
Category_output=MFile.NNStruct_categories(Features);
Category_errors = gsubtract(targets1,Category_output);
%performance1 = perform(MFile.NNStruct_categories,targets1,Category_output);
%performance2 = perform(MFile.NNStruct_categories,targets1,Category_output);
%Category_performance= max([performance1,performance2]);
    Category_error=Category_errors;
A=max(Category_output);
if A==Category_output(1)
    Category={'Male'};
    Category_cost=[1,0];

else
    Category={'Female'}; 
    Category_cost=[0,1];
end
% Plots
% Uncomment these lines to enable various plots.
% figure, plotperform(tr)
% figure, plottrainstate(tr)
%figure, plotconfusion(targets1,Category_output)
%figure, ploterrhist(Category_errors)
%Categories_errors = gsubtract(targets,frequency_categories_outputs);
%Categories_performance = perform(MFile.NNStruct_frequency_categories,targets,frequency_categories_outputs);
%==================Speech Emotions======================================%
Emotions_output=MFile.NNStruct_emotions(Features);
%frequency_errors = gsubtract(targets,frequency_emotions_outputs);
%frequency_performance = perform(MFile.NNStruct_frequency_emotions,targets,frequency_emotions_outputs);
Emotion_errors = gsubtract(targets2,Emotions_output);
%Emotion_performance= perform(MFile.NNStruct_emotions,targets2,Emotions_output);
%Emotion_performance= max([performance1,performance2,performance3,performance4]);
Emotion_error=Emotion_errors;

% Plots
% Uncomment these lines to enable various plots.
% figure, plotperform(tr)
%figure, plottrainstate(tr)
%figure, plotconfusion(targets2,Emotions_output)
%figure, ploterrhist(Emotion_errors)
A=max(Emotions_output);
if A==Emotions_output(1)
    Emotion={'Angry'};
    Emotion_cost=[1,0,0,0];
end
if A==Emotions_output(2)
   Emotion={'Happy'}; 
   Emotion_cost=[0,1,0,0];
end
if A==Emotions_output(3)
   Emotion={'Neutral'}; 
   Emotion_cost=[0,0,1,0];
end
if A==Emotions_output(4)
   Emotion={'Sad'};  
   Emotion_cost=[0,0,0,1];
end
%figure, ploterrhist(errors);