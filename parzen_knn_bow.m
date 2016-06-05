clc
clear  all
close all
num_of_clsses=8;
model=cell(num_of_clsses,3);
num_of_clusters=15;
number_of_gaussians=4;
h=0.5;

class8_training_data=load('class8_training_data.mat');
class7_training_data=load('class7_training_data.mat');
class6_training_data=load('class6_training_data.mat');
class5_training_data=load('class5_training_data.mat');
class4_training_data=load('class4_training_data.mat');
class3_training_data=load('class3_training_data.mat');
class2_training_data=load('class2_training_data.mat');
class1_training_data=load('class1_training_data.mat');

class1_testing_data = load('class1_testing_data');
class2_testing_data = load('class2_testing_data');
class3_testing_data = load('class3_testing_data');
class4_testing_data = load('class4_testing_data');
class5_testing_data = load('class5_testing_data');
class6_testing_data = load('class6_testing_data');
class7_testing_data = load('class7_testing_data');
class8_testing_data = load('class8_testing_data');

train_labels = [ones(size(class1_training_data.class1_training_data,1),1);(2*ones(size(class2_training_data.class2_training_data,1),1));(3*ones(size(class3_training_data.class3_training_data,1),1));(4*ones(size(class4_training_data.class4_training_data,1),1));(5*ones(size(class5_training_data.class5_training_data,1),1));(6*ones(size(class6_training_data.class6_training_data,1),1));(7*ones(size(class7_training_data.class7_training_data,1),1));(8*ones(size(class8_training_data.class8_training_data,1),1))];
test_labels = [ones(size(class1_testing_data.class1_testing_data,1),1);(2*ones(size(class2_testing_data.class2_testing_data,1),1));(3*ones(size(class3_testing_data.class3_testing_data,1),1));(4*ones(size(class4_testing_data.class4_testing_data,1),1));(5*ones(size(class5_testing_data.class5_testing_data,1),1));(6*ones(size(class6_testing_data.class6_testing_data,1),1));(7*ones(size(class7_testing_data.class7_testing_data,1),1));(8*ones(size(class8_testing_data.class8_testing_data,1),1))];

train_data=[class1_training_data.class1_training_data ;class2_training_data.class2_training_data;class3_training_data.class3_training_data;class4_training_data.class4_training_data;class5_training_data.class5_training_data;class6_training_data.class6_training_data;class7_training_data.class7_training_data;class8_training_data.class8_training_data];


test_data=[class1_testing_data.class1_testing_data ;class2_testing_data.class2_testing_data;class3_testing_data.class3_testing_data;class4_testing_data.class4_testing_data;class5_testing_data.class5_testing_data;class6_testing_data.class6_testing_data;class7_testing_data.class7_testing_data;class8_testing_data.class8_testing_data];


[assigned_classes proab_all]= knn_function(train_data,test_data,train_labels,num_of_clsses,k);
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
save './BOWknn_Labels_g7.mat' labels
save './BOWknn_Probability_g7.mat' proab_all


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
save './BOWknn_TPScores_g7.mat' true_postive_score
save './BOWknn_FPScores_g7.mat' false_postive_score

