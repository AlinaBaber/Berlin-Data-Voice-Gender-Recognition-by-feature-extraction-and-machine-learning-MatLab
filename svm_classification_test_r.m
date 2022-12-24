function [Category,Category_score,Categories_error,Category_cost,Emotion,Emotion_Score,Emotion_Cost,Emotion_Error] = svm_classification_test_r(Features)
%Features= allfeatures_extraction( wav_file );
MFile= load('svmmodel');
%File= load('speech');
%==================Speech Disorder======================================%
[Category,Category_score,Category_cost] = predict(MFile.SVMStruct_categories,Features);
Categories_error = loss(MFile.SVMStruct_categories,Features,Category);
DS=max(Category_score);
if DS==Category_score(1)
   Category_cost= [1,0,0,0];
end
if DS==Category_score(2)
   Category_cost= [0,1,0,0];
end
%==================Speech Emotions======================================%
%====================Angry===================================%
[Emotion_angry_happy,Emotion_angry_happy_score,Emotion_angry_happy_cost] = predict(MFile.SVMStruct_angry_happy,Features);
Angry_Happy_error = loss(MFile.SVMStruct_angry_happy,Features,Emotion_angry_happy);
[Emotion_angry_neutral,Emotion_angry_neutral_score,Emotion_angry_neutral_cost] = predict(MFile.SVMStruct_angry_neutral,Features);
Angry_Neutral_error = loss(MFile.SVMStruct_angry_neutral,Features,Emotion_angry_neutral);
[Emotion_angry_sad,Emotion_angry_sad_score,Emotion_angry_sad_cost] = predict(MFile.SVMStruct_angry_sad,Features);
Angry_Sad_error = loss(MFile.SVMStruct_angry_sad,Features,Emotion_angry_sad);
Angry=[Emotion_angry_happy,Emotion_angry_neutral,Emotion_angry_sad];
Angry_Score=[Emotion_angry_happy_score,Emotion_angry_neutral_score,Emotion_angry_sad_score];
Angry_Cost=[Emotion_angry_happy_cost,Emotion_angry_neutral_cost,Emotion_angry_sad_cost];
Angry_Error=[Angry_Happy_error,Angry_Neutral_error,Angry_Sad_error];
%======================Happy===================================================
[Emotion_happy_angry,Emotion_happy_angry_score,Emotion_happy_angry_cost] = predict(MFile.SVMStruct_happy_angry,Features);
Happy__Angry_error = loss(MFile.SVMStruct_happy_angry,Features,Emotion_happy_angry);
[Emotion_happy_neutral,Emotion_happy_neutral_score,Emotion_happy_neutral_cost] = predict(MFile.SVMStruct_happy_neutral,Features);
Happy_Neutral_error = loss(MFile.SVMStruct_happy_neutral,Features,Emotion_happy_neutral);
[Emotion_happy_sad,Emotion_happy_sad_score,Emotion_happy_sad_cost] = predict(MFile.SVMStruct_happy_sad,Features);
Happy_Sad_error = loss(MFile.SVMStruct_happy_sad,Features,Emotion_happy_sad);
Happy=[Emotion_happy_angry,Emotion_happy_neutral,Emotion_happy_sad];
Happy_Score=[Emotion_happy_angry_score,Emotion_happy_neutral_score,Emotion_happy_sad_score];
Happy_Cost=[Emotion_happy_angry_cost,Emotion_happy_neutral_cost,Emotion_happy_sad_cost];
Happy_Error=[Happy__Angry_error,Happy_Neutral_error,Happy_Sad_error];
    %====================Neutral===================================%
[Emotion_neutral_angry,Emotion_neutral_angry_score,Emotion_neutral_angry_cost] = predict(MFile.SVMStruct_neutral_angry,Features);
Neutral_Angry_error = loss(MFile.SVMStruct_neutral_angry,Features,Emotion_neutral_angry);
[Emotion_neutral_happy,Emotion_neutral_happy_score,Emotion_happy_neutral_cost] = predict(MFile.SVMStruct_neutral_happy,Features);
Neutral_Happy_error = loss(MFile.SVMStruct_neutral_happy,Features,Emotion_neutral_happy);
[Emotion_neutral_sad,Emotion_neutral_sad_score,Emotion_neutral_sad_cost] = predict(MFile.SVMStruct_neutral_sad,Features);
Neutral_Sad_error = loss(MFile.SVMStruct_neutral_sad,Features,Emotion_neutral_sad);
Neutral=[Emotion_neutral_angry,Emotion_neutral_happy,Emotion_neutral_sad];
Neutral_Score=[Emotion_neutral_angry_score,Emotion_neutral_happy_score,Emotion_neutral_sad_score];
Neutral_Cost=[Emotion_neutral_angry_cost,Emotion_happy_neutral_cost,Emotion_neutral_sad_cost];
Neutral_Error=[Neutral_Angry_error,Neutral_Happy_error,Neutral_Sad_error];
%=====================Sad=============================================
[Emotion_sad_angry,Emotion_sad_angry_score,Emotion_sad_angry_cost] = predict(MFile.SVMStruct_sad_angry,Features);
Sad_Angry_error = loss(MFile.SVMStruct_sad_angry,Features,Emotion_sad_angry);
[Emotion_sad_happy,Emotion_sad_happy_score,Emotion_sad_neutral_cost] = predict(MFile.SVMStruct_sad_happy,Features);
Sad_Happy_error = loss(MFile.SVMStruct_sad_happy,Features,Emotion_sad_happy);
[Emotion_sad_neutral,Emotion_sad_neutral_score,Emotion_neutral_sad_cost] = predict(MFile.SVMStruct_sad_neutral,Features);
Sad_Neutral_error = loss(MFile.SVMStruct_sad_neutral,Features,Emotion_sad_neutral);
Sad=[Emotion_sad_angry,Emotion_sad_happy,Emotion_sad_neutral];
Sad_Score=[Emotion_sad_angry_score,Emotion_sad_happy_score,Emotion_sad_neutral_score];
Sad_Cost=[Emotion_sad_angry_cost,Emotion_sad_neutral_cost,Emotion_neutral_sad_cost];
Sad_Error=[Sad_Angry_error,Sad_Happy_error,Sad_Neutral_error];
%--------------------------------------------------------------------------------------
Emotion=[Angry;Happy;Neutral;Sad];
Emotion_Score=[Angry_Score;Happy_Score;Neutral_Score;Sad_Score];
Emotion_Cost=[Angry_Cost,Happy_Cost,Neutral_Cost,Sad_Cost];
Emotion_error=[Angry_Error;Happy_Error;Neutral_Error;Sad_Error];
A=sum(sum(strcmp(Emotion,'Angry')));
H=sum(sum(strcmp(Emotion,'Happy')));
N=sum(sum(strcmp(Emotion,'Neutral')));
S=sum(sum(strcmp(Emotion,'Sad')));
Em=max([A,H,N,S]);
if Em==A
   Emotion ={'Angry'};
   Emotion_Score= [A,H,N,S];
   Emotion_Cost= [1,0,0,0];
   Emotion_Error= Angry_Error;
end
if Em==H
   Emotion ={'Happy'};
   Emotion_Score= [A,H,N,S];
   Emotion_Cost= [0,1,0,0];
   Emotion_Error= Happy_Error;
end
if Em==N
   Emotion={'Neutral'};
   Emotion_Score= [A,H,N,S];
   Emotion_Cost= [0,0,1,0];
   Emotion_Error= Neutral_Error;
end
if Em==S
    Emotion={'Sad'}; 
   Emotion_Score= [A,H,N,S];
   Emotion_Cost= [0,0,0,1];
   Emotion_Error= Sad_Error;
end
