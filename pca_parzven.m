clc
close all
clear all
num_of_clsses=8;


dim = 30;

%h =0.0000000000001;
h=5.75;
feature_directory=dir('/home/priyanka/Documents/MATLAB/ps3_ver3/GMM/forest/forest_features/*');
no_of_data_in_class1=length(feature_directory)-2;

creating_training_data1=[];
no_feature_vector_class1=zeros(no_of_data_in_class1,1);
for i = 3:length(feature_directory)
  name = sprintf('/home/priyanka/Documents/MATLAB/ps3_ver3/GMM/forest/forest_features/%s',feature_directory(i).name);
    fid = load(name);
  vec_fid=reshape( fid.' ,1,numel(fid));
  creating_training_data1=[ creating_training_data1 ; vec_fid];
end
[class1_train_data,class1_test_data]=divide_data(creating_training_data1);

creating_training_data2=[];

feature_directory=dir('/home/priyanka/Documents/MATLAB/ps3_ver3/GMM/coast/coast_features/*');
no_of_data_in_class2=length(feature_directory)-2;
no_feature_vector_class2=zeros(no_of_data_in_class2,1);

creating_training_data2=[];
for i = 3:length(feature_directory)
    name = sprintf('/home/priyanka/Documents/MATLAB/ps3_ver3/GMM/coast/coast_features/%s',feature_directory(i).name);
    fid = load(name);
  
vec_fid=reshape( fid.' ,1,numel(fid));
creating_training_data2=[ creating_training_data2 ; vec_fid];
end
[class2_train_data,class2_test_data]=divide_data(creating_training_data2);

feature_directory=dir('/home/priyanka/Documents/MATLAB/ps3_ver3/GMM/street/street_features/*');
creating_training_data3=[];
no_of_data_in_class3=length(feature_directory)-2;
no_feature_vector_class3=zeros(no_of_data_in_class3,1);

for i = 3:length(feature_directory)
  name = sprintf('/home/priyanka/Documents/MATLAB/ps3_ver3/GMM/street/street_features/%s',feature_directory(i).name);
    fid = load(name);
  vec_fid=reshape( fid.' ,1,numel(fid));  
  creating_training_data3=[ creating_training_data3 ; vec_fid];
end
[class3_train_data,class3_test_data]=divide_data(creating_training_data3);
%%%%% other 5 classes%%%%
feature_directory=dir('/home/priyanka/Documents/MATLAB/ps3_ver3/GMM/highway/highway_features/*');
creating_training_data4=[];
no_of_data_in_class4=length(feature_directory)-2;
no_feature_vector_class4=zeros(no_of_data_in_class4,1);


for i = 3:length(feature_directory)
  name = sprintf('/home/priyanka/Documents/MATLAB/ps3_ver3/GMM/highway/highway_features/%s',feature_directory(i).name);
    fid = load(name);
  vec_fid=reshape( fid.' ,1,numel(fid));  
  creating_training_data4=[ creating_training_data4 ; vec_fid];
end
[class4_train_data,class4_test_data]=divide_data(creating_training_data4);

feature_directory=dir('/home/priyanka/Documents/MATLAB/ps3_ver3/GMM/insidecity/insidecity_features/*');
creating_training_data5=[];
no_of_data_in_class5=length(feature_directory)-2;
no_feature_vector_class5=zeros(no_of_data_in_class5,1);

for i = 3:length(feature_directory)
  name = sprintf('/home/priyanka/Documents/MATLAB/ps3_ver3/GMM/insidecity/insidecity_features/%s',feature_directory(i).name);
    fid = load(name);
  vec_fid=reshape( fid.' ,1,numel(fid));  
  creating_training_data5=[ creating_training_data5 ; vec_fid];
end
[class5_train_data,class5_test_data]=divide_data(creating_training_data5);
feature_directory=dir('/home/priyanka/Documents/MATLAB/ps3_ver3/GMM/mountain/mountain_features/*');
creating_training_data6=[];
no_of_data_in_class6=length(feature_directory)-2;
no_feature_vector_class6=zeros(no_of_data_in_class6,1);

for i = 3:length(feature_directory)
  name = sprintf('/home/priyanka/Documents/MATLAB/ps3_ver3/GMM/mountain/mountain_features/%s',feature_directory(i).name);
    fid = load(name);
  vec_fid=reshape( fid.' ,1,numel(fid));  
  creating_training_data6=[ creating_training_data6 ; vec_fid];
end
[class6_train_data,class6_test_data]=divide_data(creating_training_data6);
feature_directory=dir('/home/priyanka/Documents/MATLAB/ps3_ver3/GMM/opencountry/opencountry_features/*');
creating_training_data7=[];
no_of_data_in_class7=length(feature_directory)-2;
no_feature_vector_class7=zeros(no_of_data_in_class7,1);

for i = 3:length(feature_directory)
  name = sprintf('/home/priyanka/Documents/MATLAB/ps3_ver3/GMM/opencountry/opencountry_features/%s',feature_directory(i).name);
    fid = load(name);
 vec_fid=reshape( fid.' ,1,numel(fid));   
  creating_training_data7=[ creating_training_data7 ; vec_fid];
end
[class7_train_data,class7_test_data]=divide_data(creating_training_data7);
feature_directory=dir('/home/priyanka/Documents/MATLAB/ps3_ver3/GMM/tallbuilding/tallbuilding_features/*');
creating_training_data8=[];
no_of_data_in_class8=length(feature_directory)-2;
no_feature_vector_class8=zeros(no_of_data_in_class8,1);

for i = 3:length(feature_directory)
  name = sprintf('/home/priyanka/Documents/MATLAB/ps3_ver3/GMM/tallbuilding/tallbuilding_features/%s',feature_directory(i).name);
    fid = load(name);
  vec_fid=reshape( fid.' ,1,numel(fid));  
  creating_training_data8=[ creating_training_data8 ; vec_fid];
end

[class8_train_data,class8_test_data]=divide_data(creating_training_data8);

train_data=[class1_train_data ;class2_train_data;class3_train_data;class4_train_data;class5_train_data;class6_train_data;class7_train_data;class8_train_data];


test_data=[class1_test_data ;class2_test_data;class3_test_data;class4_test_data;class5_test_data;class6_test_data;class7_test_data;class8_test_data];

[mu,sigma,training_data] = norm_data(train_data);
train_labels = [ones(size(class1_train_data,1),1);(2*ones(size(class2_train_data,1),1));(3*ones(size(class3_train_data,1),1));(4*ones(size(class4_train_data,1),1));(5*ones(size(class5_train_data,1),1));(6*ones(size(class6_train_data,1),1));(7*ones(size(class7_train_data,1),1));(8*ones(size(class8_train_data,1),1))];
test_labels = [ones(size(class1_test_data,1),1);(2*ones(size(class2_test_data,1),1));(3*ones(size(class3_test_data,1),1));(4*ones(size(class4_test_data,1),1));(5*ones(size(class5_test_data,1),1));(6*ones(size(class6_test_data,1),1));(7*ones(size(class7_test_data,1),1));(8*ones(size(class8_test_data,1),1))];
% 
%  training_data = norm_data_max_min(train_data);
%  testing_data = norm_data_max_min(test_data);


%%% pca %%%%%%%%%%% 
%  250 dimension 

covar=cov(training_data);for i = 1:size(class1_test_data,1)
    class1_test_data(i,:) = (class1_test_data(i,:) - mu) ./ sigma ;
end




for i = 1:size(class2_test_data,1)
    class2_test_data(i,:) = (class2_test_data(i,:) - mu) ./ sigma ;
end



for i = 1:size(class3_test_data,1)
    class3_test_data(i,:) = (class3_test_data(i,:) - mu) ./ sigma ;
end



for i = 1:size(class4_test_data,1)
    class4_test_data(i,:) = (class4_test_data(i,:) - mu) ./ sigma ;
end


for i = 1:size(class5_test_data,1)
    class5_test_data(i,:) = (class5_test_data(i,:) - mu) ./ sigma ;
end

for i = 1:size(class6_test_data,1)
    class6_test_data(i,:) = (class6_test_data(i,:) - mu) ./ sigma ;
end


for i = 1:size(class7_test_data,1)
    class7_test_data(i,:) = (class7_test_data(i,:) - mu) ./ sigma ;
end


for i = 1:size(class8_test_data,1)
    class8_test_data(i,:) = (class8_test_data(i,:) - mu) ./ sigma ;
end

testing_data = [ class1_test_data; class2_test_data; class3_test_data; class4_test_data; class5_test_data; class6_test_data; class7_test_data; class8_test_data];


[cov_length covar_width]=size(covar);
[u v]=eig(covar);
%u = pca(training_data);

reduced_data=training_data*u(:,cov_length-dim:cov_length);

reduced_testing_data=testing_data*u(:,cov_length-dim:cov_length);


%%%%%%%% getting  training and testing data for each class%%%%%%%%%%%
no_of_training_data_class1 = size(class1_train_data,1);
no_of_training_data_class2 = size(class2_train_data,1);
no_of_training_data_class3 = size(class3_train_data,1);
no_of_training_data_class4 = size(class4_train_data,1);
no_of_training_data_class5 = size(class5_train_data,1);
no_of_training_data_class6 = size(class6_train_data,1);
no_of_training_data_class7 = size(class7_train_data,1);
no_of_training_data_class8 = size(class8_train_data,1);


no_of_testing_data_class1 = size(class1_test_data,1);
no_of_testing_data_class2 = size(class2_test_data,1);
no_of_testing_data_class3 = size(class3_test_data,1);
no_of_testing_data_class4 = size(class4_test_data,1);
no_of_testing_data_class5 = size(class5_test_data,1);
no_of_testing_data_class6 = size(class6_test_data,1);
no_of_testing_data_class7 = size(class7_test_data,1);
no_of_testing_data_class8 = size(class8_test_data,1);
training_class1=reduced_data(1:no_of_training_data_class1,:);
count=no_of_training_data_class1;
last=count+no_of_training_data_class2;
training_class2=reduced_data(count+1:last,:);
count=last;
last=count+no_of_training_data_class3;
training_class3=reduced_data(count+1:last,:);
count=last;
last=count+no_of_training_data_class4;
training_class4=reduced_data(count+1:last,:);
count=last;
last=count+no_of_training_data_class5;
training_class5=reduced_data(count+1:last,:);
count=last;
last=count+no_of_training_data_class6;
training_class6=reduced_data(count+1:last,:);
count=last;
last=count+no_of_training_data_class7;
training_class7=reduced_data(count+1:last,:);
count=last;
last=count+no_of_training_data_class8;
training_class8=reduced_data(count+1:last,:);




testing_class1=reduced_testing_data(1:no_of_testing_data_class1,:);
count=no_of_testing_data_class1;
last=count+no_of_testing_data_class2;
testing_class2=reduced_testing_data(count+1:last,:);
count=last;
last=count+no_of_testing_data_class3;
testing_class3=reduced_testing_data(count+1:last,:);

count=last;
last=count+no_of_testing_data_class4;
testing_class4=reduced_testing_data(count+1:last,:);

count=last;
last=count+no_of_testing_data_class5;
testing_class5=reduced_testing_data(count+1:last,:);

count=last;
last=count+no_of_testing_data_class6;
testing_class6=reduced_testing_data(count+1:last,:);
count=last;
last=count+no_of_testing_data_class7;
testing_class7=reduced_testing_data(count+1:last,:);
count=last;
last=count+no_of_testing_data_class8;
testing_class8=reduced_testing_data(count+1:last,:);

train_data = [training_class1;training_class2;training_class3;training_class4;training_class5;training_class6;training_class7;training_class8];
test_data = [testing_class1;testing_class2;testing_class3;testing_class4;testing_class5;testing_class6;testing_class7;testing_class8];

[prob_density_class1]=parzen_fun(training_class1,test_data,h);
[prob_density_class2]=parzen_fun(training_class2,test_data,h);
[prob_density_class3]=parzen_fun(training_class3,test_data,h);
[prob_density_class4]=parzen_fun(training_class4,test_data,h);
[prob_density_class5]=parzen_fun(training_class5,test_data,h);
[prob_density_class6]=parzen_fun(training_class6,test_data,h);
[prob_density_class7]=parzen_fun(training_class7,test_data,h);
[prob_density_class8]=parzen_fun(training_class8,test_data,h);

prob_density_stacked = [prob_density_class1 prob_density_class2 prob_density_class3 prob_density_class4 prob_density_class5 prob_density_class6 prob_density_class7 prob_density_class8];
[assigned_classes proab_all]=parzen_classifier_fun(prob_density_stacked,num_of_clsses);



confusion_matrix = confusionmat(test_labels,assigned_classes);


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


no_of_test_class1 = size(testing_class1,1);
no_of_test_class2 = size(testing_class2,1);
no_of_test_class3 = size(testing_class3,1);
no_of_test_class4 = size(testing_class4,1);
no_of_test_class5 = size(testing_class5,1);
no_of_test_class6 = size(testing_class6,1);
no_of_test_class7 = size(testing_class7,1);
no_of_test_class8 = size(testing_class8,1);
testing_labels_class1 = ones(no_of_test_class1,1);
testing_labels_class2 = 2*ones(no_of_test_class2,1);
testing_labels_class3 = 3*ones(no_of_test_class3,1);
testing_labels_class4 = 4*ones(no_of_test_class4,1);
testing_labels_class5 = 5*ones(no_of_test_class5,1);
testing_labels_class6 = 6*ones(no_of_test_class6,1);
testing_labels_class7 = 7*ones(no_of_test_class7,1);
testing_labels_class8 = 8*ones(no_of_test_class8,1);
count =1;
proab_class1 = proab_all(1:no_of_test_class1)';
testing_labels_class1 = assigned_classes(1:no_of_test_class1);
count = count+no_of_test_class1;
last = count + no_of_test_class2-1;

proab_class2 = proab_all(count:last)';
testing_labels_class2 = assigned_classes(count:last);
count = last+1;
last = count + no_of_test_class3-1;
proab_class3 = proab_all(count:last)';
testing_labels_class3 = assigned_classes(count:last);
count = last+1;
last = count + no_of_test_class4-1;
proab_class4 = proab_all(count:last)';
testing_labels_class4 = assigned_classes(count:last);
count = last+1;
last = count + no_of_test_class5-1;
proab_class5 = proab_all(count:last)';
testing_labels_class5 = assigned_classes(count:last);
count = last+1;
last = count + no_of_test_class6-1;
proab_class6 = proab_all(count:last)';
testing_labels_class6 = assigned_classes(count:last);
count = last+1;
last = count + no_of_test_class7-1;
proab_class7 = proab_all(count:last)';
testing_labels_class7 = assigned_classes(count:last);
count = last+1;
last = count + no_of_test_class8-1;
proab_class8 = proab_all(count:last)';
testing_labels_class8 = assigned_classes(count:last);
% %%%%%%%%%%%%%%%%%%%% roc curves%%%%%%%%%%%%%


labels=test_labels;
classified_labels=[testing_labels_class1;testing_labels_class2;testing_labels_class3;testing_labels_class4;testing_labels_class5;testing_labels_class6;testing_labels_class7;testing_labels_class8];
save './ImageData/PCAparzvenGK_Labels.mat' labels
save './ImageData/PCAparzvenGK_Probability.mat' proab_all


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

index_class1_fp_class6=find(testing_labels_class6==1);
false_postive_score_class6=proab_class6(index_class1_fp_class6,1);

index_class1_fp_class7=find(testing_labels_class7==1);
false_postive_score_class7=proab_class7(index_class1_fp_class7,1);

index_class1_fp_class8=find(testing_labels_class8==1);
false_postive_score_class8=proab_class8(index_class1_fp_class8,1);

false_postive_score=[false_postive_score_class2;false_postive_score_class3;false_postive_score_class4;false_postive_score_class5;false_postive_score_class6;false_postive_score_class7;false_postive_score_class8];
save './ImageData/PCAparvenGK_TPScores.mat' true_postive_score
save './ImageData/PCAparzvenGK_FPScores.mat' false_postive_score
