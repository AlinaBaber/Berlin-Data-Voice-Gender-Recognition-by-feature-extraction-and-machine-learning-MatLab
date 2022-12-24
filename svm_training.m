function [Status] = svm_training()
File=load('speech');
%==================Speech Disorder======================================%
rng(10);
SVMStruct_categories = fitcsvm(File.features,File.categories,'KernelScale','auto','Standardize',true,'OutlierFraction',0.5);
save('svmmodels.mat','SVMStruct_categories','-append');
%==================Speech Emotions======================================%
%------------------Angry----------------------------------------
rng(10);
Features=[File.features(1:14,:);File.features(15:21,:)];
Angry_Happy=[File.emotions(1:14,:);File.emotions(15:21,:)];
SVMStruct_angry_happy = fitcsvm(Features,Angry_Happy,'KernelScale','auto','Standardize',true,'OutlierFraction',0.5);
save('svmmodels.mat','SVMStruct_angry_happy','-append');
rng(10);
Features=[File.features(1:14,:);File.features(22:29,:)];
Angry_Neutral=[File.emotions(1:14,:);File.emotions(22:29,:)];
SVMStruct_angry_neutral = fitcsvm(Features,Angry_Neutral,'KernelScale','auto','Standardize',true,'OutlierFraction',0.5);
save('svmmodels.mat','SVMStruct_angry_neutral','-append');
rng(10);
Features=[File.features(1:14,:);File.features(30:36,:)];
Angry_Sad=[File.emotions(1:14,:);File.emotions(30:36,:)];
SVMStruct_angry_sad = fitcsvm(Features,Angry_Sad,'KernelScale','auto','Standardize',true,'OutlierFraction',0.5);
save('svmmodels.mat','SVMStruct_angry_sad','-append');
%-----------------Happy-----------------------------------------
rng(10);
Features=[File.features(15:21,:);File.features(1:14,:)];
Happy_Angry=[File.emotions(15:21,:);File.emotions(1:14,:)];
SVMStruct_happy_angry = fitcsvm(Features,Happy_Angry,'KernelScale','auto','Standardize',true,'OutlierFraction',0.5);
save('svmmodels.mat','SVMStruct_happy_angry','-append');
rng(10);
Features=[File.features(15:21,:);File.features(22:29,:)];
Happy_Neutral=[File.emotions(15:21,:);File.emotions(22:29,:)];
SVMStruct_happy_neutral = fitcsvm(Features,Happy_Neutral,'KernelScale','auto','Standardize',true,'OutlierFraction',0.5);
save('svmmodels.mat','SVMStruct_happy_neutral','-append');
rng(10);
Features=[File.features(15:21,:);File.features(30:36,:)];
Happy_Sad=[File.emotions(15:21,:);File.emotions(30:36,:)];
SVMStruct_happy_sad = fitcsvm(Features,Happy_Sad,'KernelScale','auto','Standardize',true,'OutlierFraction',0.5);
save('svmmodels.mat','SVMStruct_happy_sad','-append');
%-----------------Neutral---------------------------------------
rng(10);
Features=[File.features(22:29,:);File.features(15:21,:)];
Neutral_Happy=[File.emotions(22:29,:);File.emotions(15:21,:)];
SVMStruct_neutral_happy = fitcsvm(Features,Neutral_Happy,'KernelScale','auto','Standardize',true,'OutlierFraction',0.5);
save('svmmodels.mat','SVMStruct_neutral_happy','-append');
rng(10);
Features=[File.features(22:29,:);File.features(1:14,:)];
Neutral_Angry=[File.emotions(22:29,:);File.emotions(1:14,:)];
SVMStruct_neutral_angry = fitcsvm(Features,Neutral_Angry,'KernelScale','auto','Standardize',true,'OutlierFraction',0.5);
save('svmmodels.mat','SVMStruct_neutral_angry','-append');
rng(10);
Features=[File.features(22:29,:);File.features(30:36,:)];
Neutral_Sad=[File.emotions(22:29,:);File.emotions(30:36,:)];
SVMStruct_neutral_sad = fitcsvm(Features,Neutral_Sad,'KernelScale','auto','Standardize',true,'OutlierFraction',0.5);
save('svmmodels.mat','SVMStruct_neutral_sad','-append');

%-----------------Sad--------------------------------------
rng(10);
Features=[File.features(30:36,:);File.features(1:14,:)];
Sad_Angry=[;File.emotions(30:36,:);File.emotions(1:14,:)];
SVMStruct_sad_angry = fitcsvm(Features,Sad_Angry,'KernelScale','auto','Standardize',true,'OutlierFraction',0.5);
save('svmmodels.mat','SVMStruct_sad_angry','-append');
rng(10);
Features=[File.features(30:36,:);File.features(22:29,:);];
Sad_Neutral=[;File.emotions(30:36,:);File.emotions(22:29,:)];
SVMStruct_sad_neutral = fitcsvm(Features,Sad_Neutral,'KernelScale','auto','Standardize',true,'OutlierFraction',0.5);
save('svmmodels.mat','SVMStruct_sad_neutral','-append');
rng(10);
Features=[File.features(30:36,:);File.features(15:21,:)];
Sad_Happy=[;File.emotions(30:36,:);File.emotions(15:21,:)];
SVMStruct_sad_happy = fitcsvm(Features,Sad_Happy,'KernelScale','auto','Standardize',true,'OutlierFraction',0.5);
save('svmmodels.mat','SVMStruct_sad_happy','-append');

Status= 'SVM Model has been Trained';