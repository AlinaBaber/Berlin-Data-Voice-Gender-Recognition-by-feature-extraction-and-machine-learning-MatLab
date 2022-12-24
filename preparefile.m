function [status]= preparefile(FilePath,FileName,wav_file)

[pathstr,filename,ext] = fileparts(FilePath);
[pathstr,emotion,ext] = fileparts(pathstr) ;
[pathstr,categories,ext] = fileparts(pathstr) ;
File=load('speech');
File_Name={FileName}; 
File.FileName(end+1)=File_Name; 
FileName=File.FileName;
save('speech.mat','FileName','-append');
File_emotions={emotion};
%File.emotions(end+1,:)=File_emotions; 
emotions=[File.emotions;File_emotions]; 
save('speech.mat','emotions','-append');
aa=strcmp(File_emotions,'Angry');
hh=strcmp(File_emotions,'Happy');
nn=strcmp(File_emotions,'Neutral');
ss=strcmp(File_emotions,'Sad');
emotion_number=[aa,hh,nn,ss];
emotions_nn=[File.emotions_nn;emotion_number];
save('speech.mat','emotions_nn','-append');
a= strcmp(emotions,'Angry');
if (a==1)
    emotion={'Angry'};
else
    emotion={'Others'}; 
end
angry=[File.angry; emotion];
save('speech.mat','angry','-append');
h= strcmp(emotions,'Happy');
if (h==1)
    emotion={'Happy'};
else
    emotion={'Others'};
end
happy=[File.happy; emotion];
save('speech.mat','happy','-append');
n= strcmp(emotions,'Neutral');
if (n==1)
  emotion={'Neutral'};
else
  emotion={'Others'}; 
end
neutral=[File.neutral; emotion];
save('speech.mat','neutral','-append');
s= strcmp(emotions,'Sad'); 
if (s==1)
emotion='Sad';
else
emotion='Others'; 
end
sad=[File.sad; emotion];
save('speech.mat','sad','-append');
File_Path={FilePath};
File.FilePath(end+1)=File_Path; 
FilePath=File.FilePath;
save('speech.mat','FilePath','-append');
File_catergory={categories};
categories=[File.categories; File_catergory];
save('speech.mat','categories','-append');
nn=strcmp(File_catergory,'Male');
ss=strcmp(File_catergory,'Female');
categories_number=[nn,ss];
categories_nn=[File.categories_nn;categories_number];
save('speech.mat','categories_nn','-append');
Features= allfeatures_extraction( wav_file );
[SVMCategory,SVMCategory_score,SVMCategories_error,SVMCategory_cost,SVMEmotion,SVMEmotion_Score,SVMEmotion_Cost,SVMEmotion_Error] = svm_classification(wav_file);
[KNNCategory,KNNCategory_score,KNNCategories_error,KNNCategory_cost,KNNEmotion,KNNEmotion_score,KNNEmotion_error,KNNEmotion_cost] = knn_classification(wav_file);
[NNCategory,NNCategory_cost,NNCategory_output,NNCategory_error,NNEmotion,NNEmotion_cost,NNEmotions_output,NNEmotion_error]=neural_network_classification(wav_file);
[DTWCategory,DTWCategory_Cost,DTWCategory_Score,DTWCategory_Error,DTW_Categories,DTWEmotion,DTWEmotion_Cost,DTWEmotion_Score,DTWEmotion_Error,DTW_Emotions] = comparison(wav_file);
 
%------(1)--------- Features-----------
File_features=Features; 
%File.frequency(end+1,end+1)=File_Frequency; 
features=[File.features;File_features;];
save('speech.mat','features','-append');
File_hybrid_features=[SVMCategory_cost,SVMCategory_score,SVMEmotion_Cost,SVMEmotion_Score,KNNCategory_cost,KNNCategory_score,KNNEmotion_cost,KNNEmotion_score,NNCategory_cost,transpose(NNCategory_output),NNEmotion_cost,transpose(NNEmotions_output),DTWCategory_Cost,DTWCategory_Score,DTWEmotion_Cost,DTWEmotion_Score];
hybrid_features=[File.hybrid_features;File_hybrid_features];
save('speech.mat','hybrid_features','-append');
status=' All Features are Extracted ,File is Prepared for Training';