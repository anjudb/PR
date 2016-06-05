clc
close all
clear all

num_of_clsses=5;
model=cell(num_of_clsses,5);
num_of_clusters=30;
number_of_gaussians=5;
feature_directory=dir('./digit_data/eight/*');
no_of_data_in_class1=length(feature_directory)-2;


no_feature_vector_class1=zeros(no_of_data_in_class1,1);
creating_training_data1=cell(1,no_feature_vector_class1);
for i = 3:length(feature_directory)
   name = sprintf('/home/priyanka/Documents/MATLAB/ps3_ver2_fun_preprocessing/digit_data/eight/%s',feature_directory(i).name);
    fid = load(name);
  
   [fid_lenght fid_width]=size(fid);
   no_feature_vector_class1(i-2,1)=fid_lenght;
  creating_training_data1{i-2}= load(name);
end

   
feature_directory=dir('./digit_data/seven/*');

no_of_data_in_class2=length(feature_directory)-2;
no_feature_vector_class2=zeros(no_of_data_in_class2,1);
creating_training_data2=cell(1,no_feature_vector_class2);



for i = 3:length(feature_directory)
   name = sprintf('/home/priyanka/Documents/MATLAB/ps3_ver2_fun_preprocessing/digit_data/seven/%s',feature_directory(i).name);
    fid = load(name);
  
   [fid_lenght fid_width]=size(fid);
      no_feature_vector_class2(i-2,1)=fid_lenght;
   creating_training_data2{i-2}=load(name);
end


feature_directory=dir('./digit_data/six/*');

no_of_data_in_class3=length(feature_directory)-2;
no_feature_vector_class3=zeros(no_of_data_in_class3,1);
creating_training_data3=cell(1,no_feature_vector_class3);
for i = 3:length(feature_directory)
 name = sprintf('/home/priyanka/Documents/MATLAB/ps3_ver2_fun_preprocessing/digit_data/six/%s',feature_directory(i).name);
    fid = load(name);
  
   [fid_lenght fid_width]=size(fid);
    no_feature_vector_class3(i-2,1)=fid_lenght;
    
  creating_training_data3{i-2}=load(name);
end



feature_directory=dir('./digit_data/two/*');

no_of_data_in_class4=length(feature_directory)-2;
no_feature_vector_class4=zeros(no_of_data_in_class4,1);
creating_training_data4=cell(1,no_feature_vector_class4);
for i = 3:length(feature_directory)
   name = sprintf('/home/priyanka/Documents/MATLAB/ps3_ver2_fun_preprocessing/digit_data/two/%s',feature_directory(i).name);
    fid = load(name);
  
   [fid_lenght fid_width]=size(fid);
   no_feature_vector_class4(i-2,1)=fid_lenght;
  creating_training_data4{i-2}=load(name);
   end
no_of_data_in_class4=length(feature_directory)-2;


feature_directory=dir('./digit_data/three/*');

no_of_data_in_class5=length(feature_directory)-2;
no_feature_vector_class5=zeros(no_of_data_in_class5,1);
creating_training_data5=cell(1,no_feature_vector_class5);
for i = 3:length(feature_directory)
  name = sprintf('/home/priyanka/Documents/MATLAB/ps3_ver2_fun_preprocessing/digit_data/three/%s',feature_directory(i).name);
    fid = load(name);
  
   [fid_lenght fid_width]=size(fid);
   no_feature_vector_class5(i-2,1)=fid_lenght;
  creating_training_data5{i-2}=load(name);
   end
no_of_data_in_class5=length(feature_directory)-2;


%creating_training_data=[creating_training_data1;creating_training_data2;creating_training_data3;creating_training_data4;creating_training_data5];


class1_training_data=creating_training_data1(1:round(0.7*no_of_data_in_class1));
class1_testing_data=creating_training_data1(round(0.7*no_of_data_in_class1)+1:no_of_data_in_class1);
num_of_one_labeles=no_of_data_in_class1-round(0.7*no_of_data_in_class1);
labels_class1(1:num_of_one_labeles,1)=ones(1:num_of_one_labeles,1);


class2_training_data=creating_training_data2(1:round(0.7*no_of_data_in_class2));
class2_testing_data=creating_training_data2(round(0.7*no_of_data_in_class2)+1:no_of_data_in_class2);
num_of_one_labeles=no_of_data_in_class2-round(0.7*no_of_data_in_class2);
labels_class2(1:num_of_one_labeles,1)=2*ones(1:num_of_one_labeles,1);

class3_training_data=creating_training_data3(1:round(0.7*no_of_data_in_class3));
class3_testing_data=creating_training_data3(round(0.7*no_of_data_in_class3)+1:no_of_data_in_class3);
num_of_one_labeles=no_of_data_in_class3-round(0.7*no_of_data_in_class3);
labels_class3(1:num_of_one_labeles,1)=3*ones(1:num_of_one_labeles,1);

class4_training_data=creating_training_data4(1:round(0.7*no_of_data_in_class4));
class4_testing_data=creating_training_data4(round(0.7*no_of_data_in_class4)+1:no_of_data_in_class4);
num_of_one_labeles=no_of_data_in_class4-round(0.7*no_of_data_in_class4);
labels_class4(1:num_of_one_labeles,1)=4*ones(1:num_of_one_labeles,1);

class5_training_data=creating_training_data5(1:round(0.7*no_of_data_in_class5));
creating_training_data5=creating_training_data5(round(0.7*no_of_data_in_class5)+1:no_of_data_in_class5);
num_of_one_labeles=no_of_data_in_class5-round(0.7*no_of_data_in_class5);
labels_class5(1:num_of_one_labeles,1)=5*ones(1:num_of_one_labeles,1);



%%%%%%%%%%%%%%%%%%%%%testing class1%%%%%%% %%%%%%%%%%%

%score=zeros(length(class5_training_data),num_of_clsses);

scores_1=get_scores_DTW(class1_testing_data,class1_training_data);
scores_2=get_scores_DTW(class1_testing_data,class2_training_data);
scores_3=get_scores_DTW(class1_testing_data,class3_training_data);
scores_4=get_scores_DTW(class1_testing_data,class4_training_data);
scores_5=get_scores_DTW(class1_testing_data,class5_training_data);

%scores_temp=zeros(length(class1_testing_data),length(class1_training_data));
%     
%     for i=1:length(class1_testing_data)
%         for j=1:length(class1_training_data)
%             temp1=class1_testing_data{1,i};
%             temp2=class1_training_data{1,j};
%             [Dist,D,k,w,rw,tw]=dtw_fun(temp2,temp1);
% 
%             scores_temp(i,j)=Dist;
%         end
%         score=min(scores_temp');
% 
%     end

scores_class1=[scores_1 scores_2 scores_3 scores_4 scores_5];

for i=1:length(scores_class1)
    temp=1./scores_class1(i,:) ;
    [ value testing_labels_class1(i)] =max(temp);
     proab_class1(i,:)=value./sum(temp);
    
end



scores_1=get_scores_DTW(class2_testing_data,class1_training_data);
scores_2=get_scores_DTW(class2_testing_data,class2_training_data);
scores_3=get_scores_DTW(class2_testing_data,class3_training_data);
scores_4=get_scores_DTW(class2_testing_data,class4_training_data);
scores_5=get_scores_DTW(class2_testing_data,class5_training_data);

scores_class2=[scores_1 scores_2 scores_3 scores_4 scores_5];


for i=1:length(scores_class2)
    temp=1./scores_class2(i,:) ;
    [ value testing_labels_class2(i)] =max(temp);
     proab_class2(i,:)=value./sum(temp);
    
end


scores_1=get_scores_DTW(class3_testing_data,class1_training_data);
scores_2=get_scores_DTW(class3_testing_data,class2_training_data);
scores_3=get_scores_DTW(class3_testing_data,class3_training_data);
scores_4=get_scores_DTW(class3_testing_data,class4_training_data);
scores_5=get_scores_DTW(class3_testing_data,class5_training_data);

scores_class3=[scores_1 scores_2 scores_3 scores_4 scores_5];

for i=1:length(scores_class3)
    temp=1./scores_class3(i,:) ;
    [ value testing_labels_class3(i)] =max(temp);
     proab_class3(i,:)=value./sum(temp);
    
end



scores_1=get_scores_DTW(class4_testing_data,class1_training_data);
scores_2=get_scores_DTW(class4_testing_data,class2_training_data);
scores_3=get_scores_DTW(class4_testing_data,class3_training_data);
scores_4=get_scores_DTW(class4_testing_data,class4_training_data);
scores_5=get_scores_DTW(class4_testing_data,class5_training_data);

scores_class4=[scores_1 scores_2 scores_3 scores_4 scores_5];


for i=1:length(scores_class4)
    temp=1./scores_class4(i,:) ;
    [ value testing_labels_class4(i)] =max(temp);
     proab_class4(i,:)=value./sum(temp);
    
end



scores_1=get_scores_DTW(creating_training_data5,class1_training_data);
scores_2=get_scores_DTW(creating_training_data5,class2_training_data);
scores_3=get_scores_DTW(creating_training_data5,class3_training_data);
scores_4=get_scores_DTW(creating_training_data5,class4_training_data);
scores_5=get_scores_DTW(creating_training_data5,class5_training_data);

scores_class5=[scores_1 scores_2 scores_3 scores_4 scores_5];

for i=1:length(scores_class5)
    temp=1./scores_class5(i,:) ;
    [ value testing_labels_class5(i)] =max(temp);
     proab_class5(i,:)=value./sum(temp);
end



%%%%%%%%%%%%%%%%%confusion matrix%%%%%%%%%%%%
confusion_matrix=zeros(num_of_clsses,num_of_clsses);

confusion_matrix(1,1)=sum((testing_labels_class1)==1);
confusion_matrix(1,2)=sum((testing_labels_class1)==2);
confusion_matrix(1,3)=sum((testing_labels_class1)==3);
confusion_matrix(1,4)=sum((testing_labels_class1)==4);
confusion_matrix(1,5)=sum((testing_labels_class1)==5);

confusion_matrix(2,1)=sum((testing_labels_class2)==1);
confusion_matrix(2,2)=sum((testing_labels_class2)==2);
confusion_matrix(2,3)=sum((testing_labels_class2)==3);
confusion_matrix(2,4)=sum((testing_labels_class2)==4);
confusion_matrix(2,5)=sum((testing_labels_class2)==5);



confusion_matrix(3,1)=sum((testing_labels_class3)==1);
confusion_matrix(3,2)=sum((testing_labels_class3)==2);
confusion_matrix(3,3)=sum((testing_labels_class3)==3);
confusion_matrix(3,4)=sum((testing_labels_class3)==4);
confusion_matrix(3,5)=sum((testing_labels_class3)==5);


confusion_matrix(4,1)=sum((testing_labels_class4)==1);
confusion_matrix(4,2)=sum((testing_labels_class4)==2);
confusion_matrix(4,3)=sum((testing_labels_class4)==3);
confusion_matrix(4,4)=sum((testing_labels_class4)==4);
confusion_matrix(4,5)=sum((testing_labels_class4)==5);

confusion_matrix(5,1)=sum((testing_labels_class5)==1);
confusion_matrix(5,2)=sum((testing_labels_class5)==2);
confusion_matrix(5,3)=sum((testing_labels_class5)==3);
confusion_matrix(5,4)=sum((testing_labels_class5)==4);
confusion_matrix(5,5)=sum((testing_labels_class5)==5);






%%%%%%%%%%%%%%%%%%%%roc curve%%%%%%%%%%%%%%%
prob_all=[proab_class1;proab_class2;proab_class3;proab_class4;proab_class5];
labels=[labels_class1;labels_class2;labels_class3;labels_class4;labels_class5];
classified_labels=[testing_labels_class1;testing_labels_class2;testing_labels_class3;testing_labels_class4;testing_labels_class5];


[X,Y] = perfcurve(labels,prob_all(:,1),1);
plot(X,Y);

hold on
[X,Y] = perfcurve(labels,prob_all(:,2),2);
plot(X,Y,'r');

hold on
[X,Y] = perfcurve(labels,prob_all(:,3),3);
plot(X,Y,'g');


[X,Y] = perfcurve(labels,prob_all(:,4),4);
plot(X,Y,'y');

hold on
[X,Y] = perfcurve(labels,prob_all(:,5),5);
plot(X,Y,'c');



precision=zeros(num_of_clsses,1);
recall=zeros(num_of_clsses,1);
f1score=zeros(num_of_clsses,1);
accuracy=0;
for i=1:num_of_clsses
   precision(i,1)=confusion_matrix(i,i)/sum(confusion_matrix(:,i));
   recall(i,1)=confusion_matrix(i,i)/sum(confusion_matrix(i,:));
   f1score(i,1)=(2*precision(i,1)*recall(i,1))/(precision(i,1)+recall(i,1));
   accuracy=accuracy+confusion_matrix(i,i);
   
end
num_data = sum(sum(confusion_matrix));
accuracy=(accuracy/num_data)*100

%%%%%%%%%%%%%%det curves%%%%%%%%%%%%%%%

index_class1_tp=find(testing_labels_class1==1);
true_postive_score=proab_class1(index_class1_tp,1);

index_class1_fp_class2=find(testing_labels_class2==1);
false_postive_score_class2=proab_class2(index_class1_fp_class2,1);

index_class1_fp_class3=find(testing_labels_class3==1);
false_postive_score_class3=proab_class3(index_class1_fp_class3,1);

index_class1_fp_class4=find(testing_labels_class4==1);
false_postive_score_class4=proab_class4(index_class1_fp_class4,1);

index_class1_fp_class5=find(testing_labels_class5==1);
false_postive_score_class5=proab_class5(index_class1_fp_class5,1);



false_postive_score=[false_postive_score_class2;false_postive_score_class3;false_postive_score_class4;false_postive_score_class5];

[Pmiss, Pfa] = Compute_DET(true_postive_score, false_postive_score);

% Plot_DET (Pmiss,Pfa,'r');
plot(Pmiss,Pfa,'r');
 

 hold on
 %%%%%%%%%%%%%%%%%class2%%%%%%%%%%%%%%%%
 
index_class2_tp=find(testing_labels_class2==2);
true_postive_score=proab_class2(index_class2_tp,2);

index_class2_fp_class1=find(testing_labels_class1==2);
false_postive_score_class1=proab_class1(index_class2_fp_class1,2);

index_class2_fp_class3=find(testing_labels_class3==2);
false_postive_score_class3=proab_class3(index_class2_fp_class3,2);

index_class2_fp_class4=find(testing_labels_class4==2);
false_postive_score_class4=proab_class4(index_class2_fp_class4,2);

index_class2_fp_class5=find(testing_labels_class5==2);
false_postive_score_class5=proab_class5(index_class2_fp_class5,2);



false_postive_score=[false_postive_score_class1;false_postive_score_class3;false_postive_score_class4;false_postive_score_class5];

[Pmiss, Pfa] = Compute_DET(true_postive_score, false_postive_score);

 %Plot_DET (Pmiss,Pfa,'g');
plot(Pmiss,Pfa,'g');
 hold on

 %%%%%%%%%%%%%%%%%class 3%%%%%%%%%%%%%%%%%%
 index_class3_tp=find(testing_labels_class3==3);
true_postive_score=proab_class3(index_class3_tp,3);

index_class3_fp_class1=find(testing_labels_class1==3);
false_postive_score_class1=proab_class1(index_class3_fp_class1,3);

index_class3_fp_class2=find(testing_labels_class3==3);
false_postive_score_class2=proab_class2(index_class3_fp_class2,3);

index_class3_fp_class4=find(testing_labels_class4==3);
false_postive_score_class4=proab_class4(index_class3_fp_class4,3);

index_class3_fp_class5=find(testing_labels_class5==3);
false_postive_score_class5=proab_class5(index_class3_fp_class5,3);



false_postive_score=[false_postive_score_class1;false_postive_score_class2;false_postive_score_class4;false_postive_score_class5];

[Pmiss, Pfa] = Compute_DET(true_postive_score, false_postive_score);

% Plot_DET (Pmiss,Pfa);
 plot(Pmiss,Pfa);
 hold on

 %%%%%%%%%%%%%%%%%%%%%class4 %%%%%%%%%%%%%%%
 index_class4_tp=find(testing_labels_class4==4);
true_postive_score=proab_class4(index_class4_tp,4);

index_class4_fp_class1=find(testing_labels_class2==4);
false_postive_score_class1=proab_class1(index_class4_fp_class1,4);

index_class4_fp_class2=find(testing_labels_class3==4);
false_postive_score_class2=proab_class2(index_class4_fp_class2,4);

index_class4_fp_class3=find(testing_labels_class4==4);
false_postive_score_class4=proab_class3(index_class4_fp_class3,4);

index_class4_fp_class5=find(testing_labels_class5==4);
false_postive_score_class5=proab_class5(index_class4_fp_class5,4);



false_postive_score=[false_postive_score_class1;false_postive_score_class2;false_postive_score_class3;false_postive_score_class5];

[Pmiss, Pfa] = Compute_DET(true_postive_score, false_postive_score);

 %Plot_DET (Pmiss,Pfa,'m');
  plot(Pmiss,Pfa,'m');
 hold on

 %%%%%%%%%%%%%%%%%%class5%%%%%%%%%%%%%%%
 index_class5_tp=find(testing_labels_class5==5);
true_postive_score=proab_class5(index_class5_tp,5);

index_class5_fp_class1=find(testing_labels_class1==5);
false_postive_score_class2=proab_class1(index_class5_fp_class1,5);

index_class5_fp_class2=find(testing_labels_class2==5);
false_postive_score_class3=proab_class2(index_class5_fp_class2,5);

index_class5_fp_class3=find(testing_labels_class3==5);
false_postive_score_class4=proab_class3(index_class5_fp_class3,5);

index_class5_fp_class4=find(testing_labels_class5==5);
false_postive_score_class5=proab_class4(index_class5_fp_class4,5);



false_postive_score=[false_postive_score_class1;false_postive_score_class2;false_postive_score_class3;false_postive_score_class4];

[Pmiss, Pfa] = Compute_DET(true_postive_score, false_postive_score);

 %Plot_DET (Pmiss,Pfa,'y');

 plot(Pmiss,Pfa,'y');


