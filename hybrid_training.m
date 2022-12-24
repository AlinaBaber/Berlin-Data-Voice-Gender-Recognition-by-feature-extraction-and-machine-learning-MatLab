function Status = hybrid_training()
File=load('speech');
%=====================SVM==================================%
%==================Speech Disorder======================================%
rng(10);
HybridSVMStruct_categories = fitcsvm(File.hybrid_features,File.categories,'KernelScale','auto','Standardize',true,'OutlierFraction',0.5);
save('hybridmodels.mat','HybridSVMStruct_categories','-append');
%==================Speech Emotions======================================%
%------------------Angry----------------------------------------
rng(10);
Features=[File.hybrid_features(1:14,:);File.hybrid_features(15:21,:)];
Angry_Happy=[File.emotions(1:14,:);File.emotions(15:21,:)];
HybridSVMStruct_angry_happy = fitcsvm(Features,Angry_Happy,'KernelScale','auto','Standardize',true,'OutlierFraction',0.5);
save('hybridmodels.mat','HybridSVMStruct_angry_happy','-append');
rng(10);
Features=[File.hybrid_features(1:14,:);File.hybrid_features(22:29,:)];
Angry_Neutral=[File.emotions(1:14,:);File.emotions(22:29,:)];
HybridSVMStruct_angry_neutral = fitcsvm(Features,Angry_Neutral,'KernelScale','auto','Standardize',true,'OutlierFraction',0.5);
save('hybridmodels.mat','HybridSVMStruct_angry_neutral','-append');
rng(10);
Features=[File.hybrid_features(1:14,:);File.hybrid_features(30:36,:)];
Angry_Sad=[File.emotions(1:14,:);File.emotions(30:36,:)];
HybridSVMStruct_angry_sad = fitcsvm(Features,Angry_Sad,'KernelScale','auto','Standardize',true,'OutlierFraction',0.5);
save('hybridmodels.mat','HybridSVMStruct_angry_sad','-append');
%-----------------Happy-----------------------------------------
rng(10);
Features=[File.hybrid_features(15:21,:);File.hybrid_features(1:14,:)];
Happy_Angry=[File.emotions(15:21,:);File.emotions(1:14,:)];
HybridSVMStruct_happy_angry = fitcsvm(Features,Happy_Angry,'KernelScale','auto','Standardize',true,'OutlierFraction',0.5);
save('hybridmodels.mat','HybridSVMStruct_happy_angry','-append');
rng(10);
Features=[File.hybrid_features(15:21,:);File.hybrid_features(22:29,:)];
Happy_Neutral=[File.emotions(15:21,:);File.emotions(22:29,:)];
HybridSVMStruct_happy_neutral = fitcsvm(Features,Happy_Neutral,'KernelScale','auto','Standardize',true,'OutlierFraction',0.5);
save('hybridmodels.mat','HybridSVMStruct_happy_neutral','-append');
rng(10);
Features=[File.hybrid_features(15:21,:);File.hybrid_features(30:36,:)];
Happy_Sad=[File.emotions(15:21,:);File.emotions(30:36,:)];
HybridSVMStruct_happy_sad = fitcsvm(Features,Happy_Sad,'KernelScale','auto','Standardize',true,'OutlierFraction',0.5);
save('hybridmodels.mat','HybridSVMStruct_happy_sad','-append');
%-----------------Neutral---------------------------------------
rng(10);
Features=[File.hybrid_features(22:29,:);File.hybrid_features(15:21,:)];
Neutral_Happy=[File.emotions(22:29,:);File.emotions(15:21,:)];
HybridSVMStruct_neutral_happy = fitcsvm(Features,Neutral_Happy,'KernelScale','auto','Standardize',true,'OutlierFraction',0.5);
save('hybridmodels.mat','HybridSVMStruct_neutral_happy','-append');
rng(10);
Features=[File.hybrid_features(22:29,:);File.hybrid_features(1:14,:)];
Neutral_Angry=[File.emotions(22:29,:);File.emotions(1:14,:)];
HybridSVMStruct_neutral_angry = fitcsvm(Features,Neutral_Angry,'KernelScale','auto','Standardize',true,'OutlierFraction',0.5);
save('hybridmodels.mat','HybridSVMStruct_neutral_angry','-append');
rng(10);
Features=[File.hybrid_features(22:29,:);File.hybrid_features(30:36,:)];
Neutral_Sad=[File.emotions(22:29,:);File.emotions(30:36,:)];
HybridSVMStruct_neutral_sad = fitcsvm(Features,Neutral_Sad,'KernelScale','auto','Standardize',true,'OutlierFraction',0.5);
save('hybridmodels.mat','HybridSVMStruct_neutral_sad','-append');

%-----------------Sad--------------------------------------
rng(10);
Features=[File.hybrid_features(30:36,:);File.hybrid_features(1:14,:)];
Sad_Angry=[File.emotions(30:36,:);File.emotions(1:14,:)];
HybridSVMStruct_sad_angry = fitcsvm(Features,Sad_Angry,'KernelScale','auto','Standardize',true,'OutlierFraction',0.5);
save('hybridmodels.mat','HybridSVMStruct_sad_angry','-append');
rng(10);
Features=[File.hybrid_features(30:36,:);File.hybrid_features(22:29,:)];
Sad_Neutral=[File.emotions(30:36,:);File.emotions(22:29,:)];
HybridSVMStruct_sad_neutral = fitcsvm(Features,Sad_Neutral,'KernelScale','auto','Standardize',true,'OutlierFraction',0.5);
save('hybridmodels.mat','HybridSVMStruct_sad_neutral','-append');
rng(10);
Features=[File.hybrid_features(30:36,:);File.hybrid_features(15:21,:)];
Sad_Happy=[File.emotions(30:36,:);File.emotions(15:21,:)];
HybridSVMStruct_sad_happy = fitcsvm(Features,Sad_Happy,'KernelScale','auto','Standardize',true,'OutlierFraction',0.5);
save('hybridmodels.mat','HybridSVMStruct_sad_happy','-append');
%======================KNN=============================================%
%==================Speech Disorder======================================%

rng(100);
HybridKNNStruct_categories = fitcknn(File.hybrid_features,File.categories,'NumNeighbors',1);
save('hybridmodels.mat','HybridKNNStruct_categories','-append');

%==================Speech Emotions======================================%

rng(100);
HybridKNNStruct_emotions = fitcknn(File.hybrid_features,File.emotions,'NumNeighbors',1);
save('hybridmodels.mat','HybridKNNStruct_emotions','-append');
%==================NN======================================%
%----(1)----------Frequency--------------

[HybridNNStruct_categories,HybridNNTr_categories] = neural_network_trainingcat(File.hybrid_features,File.categories_nn);
save('hybridmodels.mat','HybridNNStruct_categories','-append');
save('hybridmodels.mat','HybridNNTr_categories','-append');
figure, plotperform(HybridNNTr_categories);
figure, plottrainstate(HybridNNTr_categories);
%==================Speech Emotions======================================%
%----(1)----------Frequency--------------
[HybridNNStruct_emotions,HybridNNTr_emotions]  = neural_network_trainingem(File.hybrid_features,File.emotions_nn);
save('hybridmodels.mat','HybridNNStruct_emotions','-append');
save('hybridmodels.mat','HybridNNTr_emotions','-append');
figure, plotperform(HybridNNTr_emotions);
figure, plottrainstate(HybridNNTr_emotions);
%======================DTW================================================%
 %--------------Normal------------
 [rn,cn]=find(strcmp(File.categories,'Male'));
 HybridDTWNormal_features=mean(File.hybrid_features(rn,:));
 save('hybridmodels.mat','HybridDTWNormal_features','-append');
 %---------------Special-------------------
 [rn,cn]=find(strcmp(File.categories,'Female'));
 HybridDTWSpecial_features=mean(File.hybrid_features(rn,:));
 save('hybridmodels.mat','HybridDTWSpecial_features','-append');
 %----------------Angry--------------------------------
  [rn,cn]=find(strcmp(File.emotions,'Angry'));
 HybridDTWAngry_features=mean(File.hybrid_features(rn,:));
 save('hybridmodels.mat','HybridDTWAngry_features','-append');
 %--------------Happy----------------------------------
  [rn,cn]=find(strcmp(File.emotions,'Happy'));
 HybridDTWHappy_features=mean(File.hybrid_features(rn,:));
 save('hybridmodels.mat','HybridDTWHappy_features','-append');
 %-------------Neutral-------------------
  [rn,cn]=find(strcmp(File.emotions,'Neutral'));
 HybridDTWNeutral_features=mean(File.hybrid_features(rn,:));
 save('hybridmodels.mat','HybridDTWNeutral_features','-append');
 %------------------Sad-------------------------------
  [rn,cn]=find(strcmp(File.emotions,'Sad'));
 HybridDTWSad_features=mean(File.hybrid_features(rn,:));
 save('hybridmodels.mat','HybridDTWSad_features','-append');
Status= 'Hybrid Model has been Trained';