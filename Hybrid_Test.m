function [Category_Error_Rate,Category_Accuracy_Rate,Emotion_Error_Rate,Emotion_Accuracy_Rate,Categories,Emotions,Category_SampleDistribution,Category_ErrorDistribution,Emotion_SampleDistribution,Emotion_ErrorDistribution]= Hybrid_Test()
File=load('speech');
[row,col]=size(File.hybrid_features); Categories={}; Emotions={};
for i=1:row
[Category,Emotion]=hybrid_classification_Test(File.hybrid_features(i,:));
Categories(i)=Category;
Emotions(i)=Emotion;
end
Hybrid_Category_Analysis = classperf(File.categories,Categories);
Category_Error_Rate=Hybrid_Category_Analysis.ErrorRate;
Category_Accuracy_Rate=Hybrid_Category_Analysis.CorrectRate;
Hybrid_Emotion_Analysis = classperf(File.emotions,Emotions);
Emotion_Error_Rate=Hybrid_Emotion_Analysis.ErrorRate;
Emotion_Accuracy_Rate=Hybrid_Emotion_Analysis.CorrectRate;
Category_SampleDistribution=Hybrid_Category_Analysis.SampleDistribution;
Category_ErrorDistribution=Hybrid_Category_Analysis.ErrorDistribution;
Emotion_SampleDistribution=Hybrid_Emotion_Analysis.SampleDistribution;
Emotion_ErrorDistribution=Hybrid_Emotion_Analysis.ErrorDistribution;
figure(1),subplot(2,1,1);
bar(Category_SampleDistribution-Category_ErrorDistribution); xlabel('Samples');ylabel('Accuracy Rate'); title('Category Performance');
subplot(2,1,2);
bar(Emotion_SampleDistribution-Emotion_ErrorDistribution); xlabel('Samples');ylabel('Accuracy Rate'); title('Emotion Performance');
figure(2),subplot(2,1,1);
bar(Category_ErrorDistribution); xlabel('Samples');ylabel('Error Rate'); title('Category Error');
subplot(2,1,2);
bar(Emotion_ErrorDistribution); xlabel('Samples');ylabel('Error Rate'); title('Emotion Error');