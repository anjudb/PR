
%%%%%%%%%class 1%%%%%%%%
clc;
clear all;
close all;

num_of_clsses=8;
model=cell(num_of_clsses,3);
num_of_clusters=100;
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
% training_data = norm_data_max_min(train_data);
% testing_data = norm_data_max_min(train_data);

no_of_training_data_class1 = size(class1_train_data,1);
no_of_training_data_class2 = size(class2_train_data,1);
no_of_training_data_class3 = size(class3_train_data,1);
no_of_training_data_class4 = size(class4_train_data,1);
no_of_training_data_class5 = size(class5_train_data,1);
no_of_training_data_class6 = size(class6_train_data,1);
no_of_training_data_class7 = size(class7_train_data,1);
no_of_training_data_class8 = size(class8_train_data,1);


class1_train_data=training_data(1:no_of_training_data_class1,:);
count=no_of_training_data_class1;
last=count+no_of_training_data_class2;
class2_train_data=training_data(count+1:last,:);
count=last;
last=count+no_of_training_data_class3;
class3_train_data=training_data(count+1:last,:);
count=last;
last=count+no_of_training_data_class4;
class4_train_data=training_data(count+1:last,:);
count=last;
last=count+no_of_training_data_class5;
class5_train_data=training_data(count+1:last,:);
count=last;
last=count+no_of_training_data_class6;
class6_train_data=training_data(count+1:last,:);
count=last;
last=count+no_of_training_data_class7;
class7_train_data=training_data(count+1:last,:);
count=last;
last=count+no_of_training_data_class8;
class8_train_data=training_data(count+1:last,:);


for i = 1:size(class1_test_data,1)
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
LocalMu = cell(1,num_of_clsses);
CovVal = cell(1,num_of_clsses);
sizeC=zeros(1,num_of_clsses);
 % Compute local Mu, cov matrix for each class data         
        LocalMu(1) = {mean(class1_train_data,1)};
        CovVal(1) = {cov(class1_train_data)};
        sizeC(1)=size(class1_train_data,1);
       
        
         LocalMu(2) = {mean(class2_train_data,1)};
        CovVal(2) = {cov(class2_train_data)};
        sizeC(2)=size(class2_train_data,1);
        
     LocalMu(3) = {mean(class3_train_data,1)};
        CovVal(3) = {cov(class3_train_data)};
        sizeC(3)=size(class3_train_data,1);
       
         LocalMu(4) = {mean(class4_train_data,1)};
        CovVal(4) = {cov(class4_train_data)};
        sizeC(4)=size(class4_train_data,1);
       
         LocalMu(5) = {mean(class5_train_data,1)};
        CovVal(5) = {cov(class5_train_data)};
        sizeC(5)=size(class5_train_data,1);
       
         LocalMu(6) = {mean(class6_train_data,1)};
        CovVal(6) = {cov(class6_train_data)};
        sizeC(6)=size(class6_train_data,1);
       
         LocalMu(7) = {mean(class7_train_data,1)};
        CovVal(7) = {cov(class7_train_data)};
        sizeC(7)=size(class7_train_data,1);
       
       
         LocalMu(8) = {mean(class8_train_data,1)};
        CovVal(8) = {cov(class8_train_data)};
        sizeC(8)=size(class8_train_data,1);
       
%Computing  the Global Mu
    Global_Mu = zeros(1,size(class1_train_data,2));
    
    for i = 1:num_of_clsses
        Global_Mu = Global_Mu+LocalMu{i};
    end
    Global_Mu=Global_Mu./num_of_clsses;
    
    
    
    SB = zeros(size(class1_train_data,2),size(class1_train_data,2));
    SW = zeros(size(class1_train_data,2),size(class1_train_data,2));
    
    for i = 1:num_of_clsses
        SB = SB + sizeC(i).*(LocalMu{i}-Global_Mu)*(LocalMu{i}-Global_Mu)';
        SW = SW+CovVal{i};
         %SW = SW+(((PriorProb(i)*size(train_data,1)) - 1) / (size(train_data,1) - num_of_clsses) ).*CovVal{i};
    end
    invSw = inv(SW);
    invSw_by_SB = invSw * SB;
    PriorProb = zeros(num_of_clsses,1);
    PriorProb(1) = size(class1_train_data,1) / size(train_data,1);
    PriorProb(2) = size(class2_train_data,1) / size(train_data,1);
    PriorProb(3) = size(class3_train_data,1) / size(train_data,1);
    PriorProb(4) = size(class4_train_data,1) / size(train_data,1);
    PriorProb(5) = size(class5_train_data,1) / size(train_data,1);
    PriorProb(6) = size(class6_train_data,1) / size(train_data,1);
    PriorProb(7) = size(class7_train_data,1) / size(train_data,1);
    PriorProb(8) = size(class8_train_data,1) / size(train_data,1);
    
    W = zeros(num_of_clsses,size(train_data,2)+1);
    for i = 1:num_of_clsses,
        Temp = cell2mat(LocalMu(i)) * invSw;
        W(i,1) = -0.5 * Temp * cell2mat(LocalMu(i))' + log(PriorProb(i));
        W(i,2:end) = Temp;
    end
    test_data_addb = [ones(size(testing_data),1) testing_data];
    fda_test_data = test_data_addb * W';
    Prob = exp(fda_test_data);
    for i=1:size(Prob,1)
        Prob(i,:) = Prob(i,:)/sum(Prob(i,:));
    end
    testing_labels_class = zeros(size(Prob,1),1);
    assigned_classes =[];
     for i=1:size(test_data,1)
     [temp testing_labels_class(i,1)]=max((Prob(i,:))');
     assigned_classes = [assigned_classes;testing_labels_class(i,1)];
     end
 
  confusion_matrix = confusionmat(test_labels,testing_labels_class);  
    
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
no_of_test_class1 = size(class1_test_data,1);
no_of_test_class2 = size(class2_test_data,1);
no_of_test_class3 = size(class3_test_data,1);
no_of_test_class4 = size(class4_test_data,1);
no_of_test_class5 = size(class5_test_data,1);
no_of_test_class6 = size(class6_test_data,1);
no_of_test_class7 = size(class7_test_data,1);
no_of_test_class8 = size(class8_test_data,1);

%%%% Help this prob part how to divide something wrong below code. i have Prob array of size 806x8 no_of_test_samples x num_of_classes
count =1;
proab_class1 = Prob(1:no_of_test_class1,1);
testing_labels_class1 = assigned_classes(1:no_of_test_class1);
count = count+no_of_test_class1;
last = count + no_of_test_class2-1;

proab_class2 = Prob(count:last,2);
testing_labels_class2 = assigned_classes(count:last);
count = last+1;
last = count + no_of_test_class3-1;
proab_class3 = Prob(count:last,3);
testing_labels_class3 = assigned_classes(count:last);
count = last+1;
last = count + no_of_test_class4-1;
proab_class4 = Prob(count:last,4);
testing_labels_class4 = assigned_classes(count:last);
count = last+1;
last = count + no_of_test_class5-1;
proab_class5 = Prob(count:last,5);
testing_labels_class5 = assigned_classes(count:last);
count = last+1;
last = count + no_of_test_class6-1;
proab_class6 = Prob(count:last,6);
testing_labels_class6 = assigned_classes(count:last);
count = last+1;
last = count + no_of_test_class7-1;
proab_class7 = Prob(count:last,7);
testing_labels_class7 = assigned_classes(count:last);
count = last+1;
last = count + no_of_test_class8-1;
proab_class8 = Prob(count:last,8);
testing_labels_class8 = assigned_classes(count:last);

%%%%%%%%%%%%%%%%%%%% roc curves%%%%%%%%%%%%%
prob_all=[proab_class1;proab_class2;proab_class3;proab_class4;proab_class5;proab_class6;proab_class7;proab_class8];
labels=test_labels;
classified_labels=[testing_labels_class1;testing_labels_class2;testing_labels_class3;testing_labels_class4;testing_labels_class5;testing_labels_class6;testing_labels_class7;testing_labels_class8];
save './ImageData/FDA_Labels_g7.mat' labels
save './ImageData/FDA_Probability_g7.mat' prob_all


index_class1_tp=find(testing_labels_class(1)==1);
true_postive_score=proab_class1(index_class1_tp,1);
[X,Y] = perfcurve(labels,prob_all(:,1),1);
plot(X,Y);
index_class1_fp_class2=find(testing_labels_class(2)==1);
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
save './ImageData/FDA_TPScores_g7.mat' true_postive_score
save './ImageData/FDA_FPScores_g7.mat' false_postive_score

%%%%%%%%%%%%%%%%%%det%%%%%%%%%%%%%%%%

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

% save './roc_det_different_case/PCA_TPScores_g4.mat' true_postive_score
% save './roc_det_different_case/PCA_FPScores_g4.mat' false_postive_score
% 


[Pmiss, Pfa] = Compute_DET(true_postive_score, false_postive_score);

figure
Plot_DET(Pmiss,Pfa,'r');

