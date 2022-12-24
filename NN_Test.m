function [Category_Error_Rate,Category_Accuracy_Rate,Emotion_Error_Rate,Emotion_Accuracy_Rate,Categories,Emotions,Category_SampleDistribution,Category_ErrorDistribution,Emotion_SampleDistribution,Emotion_ErrorDistribution]= NN_Test()
File=load('speech');
[row,col]=size(File.features); Categories={}; Emotions={};
for i=1:row
[Category,Category_cost,Category_output,Emotion,Emotion_cost,Emotions_output]=neural_network_classification_Test(File.features(i,:));
Categories(i)=Category;
Emotions(i)=Emotion;
end
NN_Category_Analysis = classperf(File.categories,Categories);
Category_Error_Rate=NN_Category_Analysis.ErrorRate;
Category_Accuracy_Rate=NN_Category_Analysis.CorrectRate;
NN_Emotion_Analysis = classperf(File.emotions,Emotions);
Emotion_Error_Rate=NN_Emotion_Analysis.ErrorRate;
Emotion_Accuracy_Rate=NN_Emotion_Analysis.CorrectRate;
Category_SampleDistribution=NN_Category_Analysis.SampleDistribution;
Category_ErrorDistribution=NN_Category_Analysis.ErrorDistribution;
Emotion_SampleDistribution=NN_Emotion_Analysis.SampleDistribution;
Emotion_ErrorDistribution=NN_Emotion_Analysis.ErrorDistribution;
figure(1),subplot(2,1,1);
bar(Category_SampleDistribution-Category_ErrorDistribution); xlabel('Samples');ylabel('Accuracy Rate'); title('Category Performance');
subplot(2,1,2);
bar(Emotion_SampleDistribution-Emotion_ErrorDistribution); xlabel('Samples');ylabel('Accuracy Rate'); title('Emotion Performance');
figure(2),subplot(2,1,1);
bar(Category_ErrorDistribution); xlabel('Samples');ylabel('Error Rate'); title('Category Error');
subplot(2,1,2);
bar(Emotion_ErrorDistribution); xlabel('Samples');ylabel('Error Rate'); title('Emotion Error');