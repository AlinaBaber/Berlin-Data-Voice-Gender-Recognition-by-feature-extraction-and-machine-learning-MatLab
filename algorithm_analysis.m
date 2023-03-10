function [Normal_Error_Rate,Special_Error_Rate,Normal_Accuracy_Rate,Special_Accuracy_Rate]= algorithm_analysis(Categories,Emotions)
%File=load('speech');
   Categories=transpose(Categories); Emotions=transpose(Emotions);
   NAC=strcmp(Categories(1:8,1),'Male'); 
   NAE=strcmp(Emotions(1:8,1),'Angry');
   SAC=strcmp(Categories(9:14,1),'Female'); 
   SAE=strcmp(Emotions(9:14,1),'Angry');
   NHC=strcmp(Categories(15:17,1),'Male'); 
   NHE=strcmp(Emotions(15:17,1),'Happy');
   SHC=strcmp(Categories(18:21,1),'Female'); 
   SHE=strcmp(Emotions(18:21,1),'Happy');
   NNC=strcmp(Categories(22:25,1),'Male'); 
   NNE=strcmp(Emotions(22:25,1),'Neutral');
   SNC=strcmp(Categories(26:29,1),'Female'); 
   SNE=strcmp(Emotions(26:29,1),'Neutral');
   NSC=strcmp(Categories(30:32,1),'Male'); 
   NSE=strcmp(Emotions(30:32,1),'Sad');
   SSC=strcmp(Categories(33:36,1),'Female'); 
   SSE=strcmp(Emotions(33:36,1),'Sad');
   Normal_Angry=[sum(NAC),sum(NAE)];
   Special_Angry=[sum(SAC),sum(SAE)];
   Normal_Happy=[sum(NHC),sum(NHE)];
   Special_Happy=[sum(SHC),sum(SHE)];
   Normal_Neutral=[sum(NNC),sum(NNE)];
   Special_Neutral=[sum(SNC),sum(SNE)];
   Normal_Sad=[sum(NSC),sum(NSE)];
   Special_Sad=[sum(SSC),sum(SSE)];
   Normal_Angry_Error_Rate= max([8-Normal_Angry(1,1),8-Normal_Angry(1,2)])/8;
   Special_Angry_Error_Rate= max([6-Special_Angry(1,1),6-Special_Angry(1,2)])/6;
   Normal_Happy_Error_Rate= max([3-Normal_Happy(1,1),3-Normal_Happy(1,2)])/3;
   Special_Happy_Error_Rate= max([4-Special_Happy(1,1),4-Special_Happy(1,2)])/4;
   Normal_Neutral_Error_Rate= max([4-Normal_Neutral(1,1),4-Normal_Neutral(1,2)])/4;
   Special_Neutral_Error_Rate= max([4-Special_Neutral(1,1),4-Special_Neutral(1,2)])/4;
   Normal_Sad_Error_Rate= max([3-Normal_Sad(1,1),3-Normal_Sad(1,2)])/3;
   Special_Sad_Error_Rate= max([4-Special_Sad(1,1),4-Special_Sad(1,2)])/4;
   Normal_Error_Rate=[Normal_Angry_Error_Rate,Normal_Happy_Error_Rate,Normal_Neutral_Error_Rate,Normal_Sad_Error_Rate];
   Special_Error_Rate=[Special_Angry_Error_Rate,Special_Happy_Error_Rate,Special_Neutral_Error_Rate,Special_Sad_Error_Rate];
   
   Normal_Angry_Accuracy_Rate= min([Normal_Angry(1,1),Normal_Angry(1,2)])/8;
   Special_Angry_Accuracy_Rate= min([Special_Angry(1,1),Special_Angry(1,2)])/6;
   Normal_Happy_Accuracy_Rate= min([Normal_Happy(1,1),Normal_Happy(1,2)])/3;
   Special_Happy_Accuracy_Rate= min([Special_Happy(1,1),Special_Happy(1,2)])/4;
   Normal_Neutral_Accuracy_Rate= min([Normal_Neutral(1,1),Normal_Neutral(1,2)])/4;
   Special_Neutral_Accuracy_Rate= min([Special_Neutral(1,1),Special_Neutral(1,2)])/4;
   Normal_Sad_Accuracy_Rate= min([Normal_Sad(1,1),Normal_Sad(1,2)])/3;
   Special_Sad_Accuracy_Rate= min([Special_Sad(1,1),Special_Sad(1,2)])/4;
   Normal_Accuracy_Rate=[Normal_Angry_Accuracy_Rate,Normal_Happy_Accuracy_Rate,Normal_Neutral_Accuracy_Rate,Normal_Sad_Accuracy_Rate];
   Special_Accuracy_Rate=[Special_Angry_Accuracy_Rate,Special_Happy_Accuracy_Rate,Special_Neutral_Accuracy_Rate,Special_Sad_Accuracy_Rate];
   
   