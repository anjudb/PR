clc
close all
clear all

%%%%%%%%%%%%%%%% Seperating the data from given file %%%%%%%%%%%%%

num_of_clsses = 5;
num_classes=5;
no_of_data_in_class = zeros(num_classes,1);

%%%%%%%%%%%% Class1
feature_directory=dir('/home/anju/Desktop/PR4/mandi_data_set/Ariyalur/*');
creating_training_data1=[];
no_of_data_in_class(1,1)=length(feature_directory)-2;
no_feature_vector_class1=zeros(no_of_data_in_class(1,1),1);
for i = 3:length(feature_directory)
  name = sprintf('/home/anju/Desktop/PR4/mandi_data_set/Ariyalur/%s',feature_directory(i).name);
    data=dlmread(name,' ',1,1);
    [data_lenght data_width]=size(data);
   no_feature_vector_class1(i-2,1)=data_lenght;
  creating_training_data1=[ creating_training_data1 ; data];
   end
no_of_data_in_class1=length(feature_directory)-2;


%%%%%%%%%%%%% class 2
feature_directory=dir('/home/anju/Desktop/PR4/mandi_data_set/Chennai/*');
creating_training_data2=[];
no_of_data_in_class(2,1)=length(feature_directory)-2;
no_feature_vector_class2=zeros(no_of_data_in_class(2,1),1);
for i = 3:length(feature_directory)
  name = sprintf('/home/anju/Desktop/PR4/mandi_data_set/Chennai/%s',feature_directory(i).name);
    data=dlmread(name,' ',1,1);
    [data_lenght data_width]=size(data);
   no_feature_vector_class2(i-2,1)=data_lenght;
  creating_training_data2=[ creating_training_data2 ; data];
   end
no_of_data_in_class2=length(feature_directory)-2;

%%%%%%%%%%%%% class 3
feature_directory=dir('/home/anju/Desktop/PR4/mandi_data_set/Coimbatore/*');
creating_training_data3=[];
no_of_data_in_class(3,1)=length(feature_directory)-2;
no_feature_vector_class3=zeros(no_of_data_in_class(3,1),1);
for i = 3:length(feature_directory)
  name = sprintf('/home/anju/Desktop/PR4/mandi_data_set/Coimbatore/%s',feature_directory(i).name);
    data=dlmread(name,' ',1,1);
    [data_lenght data_width]=size(data);
   no_feature_vector_class3(i-2,1)=data_lenght;
  creating_training_data3=[ creating_training_data3 ; data];
   end
no_of_data_in_class3=length(feature_directory)-2;

%%%%%%%%%%%%% class 4
feature_directory=dir('/home/anju/Desktop/PR4/mandi_data_set/Cuddalore/*');
creating_training_data4=[];
no_of_data_in_class(4,1)=length(feature_directory)-2;
no_feature_vector_class4=zeros(no_of_data_in_class(4,1),1);
for i = 3:length(feature_directory)
  name = sprintf('/home/anju/Desktop/PR4/mandi_data_set/Cuddalore/%s',feature_directory(i).name);
    data=dlmread(name,' ',1,1);
    [data_lenght data_width]=size(data);
   no_feature_vector_class4(i-2,1)=data_lenght;
  creating_training_data4=[ creating_training_data4 ; data];
   end
no_of_data_in_class4=length(feature_directory)-2;

%%%%%%%%%%%%% class 5
feature_directory=dir('/home/anju/Desktop/PR4/mandi_data_set/Dharmapuri/*');
creating_training_data5=[];
no_of_data_in_class(5,1)=length(feature_directory)-2;
no_feature_vector_class5=zeros(no_of_data_in_class(5,1),1);
for i = 3:length(feature_directory)
  name = sprintf('/home/anju/Desktop/PR4/mandi_data_set/Dharmapuri/%s',feature_directory(i).name);
    data=dlmread(name,' ',1,1);
    [data_lenght data_width]=size(data);
   no_feature_vector_class5(i-2,1)=data_lenght;
  creating_training_data5=[ creating_training_data5 ; data];
   end
no_of_data_in_class5=length(feature_directory)-2;

%%%%%%%%%%%%%%%%% K Means %%%%%%%%%%%%%%%
num_of_clusters= 40;

creating_training_data=[creating_training_data1;creating_training_data2;creating_training_data3;creating_training_data4;creating_training_data5];

[labels,cluster_centers,sumd,D]=kmeans(creating_training_data,num_of_clusters,'MaxIter',500);





%%%%%%%%%%%% creating bag of words%%%%%%%
    
%%%%%%class1%%%%%%%%%

data_class1=cell(1,no_of_data_in_class1);
count=0;

for i=1:no_of_data_in_class1
    temp=labels(count+1:count+no_feature_vector_class1(i,1),:);
    count=count+no_feature_vector_class1(i,1);
    data_class1{i}=temp';
    
  end

% mu = mean(data_class1);
% sigma = std(data_class1);
% 
% for i = 1:size(data_class1,1)
%     norm_data_class1(i,:) = (data_class1(i,:)) - mu ./ sigma ;
% end


count
data_class2=cell(1,no_of_data_in_class2);


for i=1:no_of_data_in_class2
    temp=labels(count+1:count+no_feature_vector_class2(i,1),:);
    count=count+no_feature_vector_class2(i,1);
    
    data_class2{i}=temp';
    
    
end


% mu = mean(data_class2);
% sigma = std(data_class2);
% 
% for i = 1:size(data_class2,1)
%     norm_data_class2(i,:) = (data_class2(i,:)) - mu ./ sigma ;
% end

count

data_class3=cell(1,no_of_data_in_class3);


for i=1:no_of_data_in_class3
    temp=labels(count+1:count+no_feature_vector_class3(i,1),:);
    count=count+no_feature_vector_class3(i,1);
   
    data_class3{i}=temp';
    
    
end


% mu = mean(data_class3);
% sigma = std(data_class3);
% 
% for i = 1:size(data_class3,1)
%     norm_data_class3(i,:) = (data_class3(i,:)) - mu ./ sigma ;
% end
 
count

data_class4=cell(1,no_of_data_in_class4);
for i=1:no_of_data_in_class4
    temp=labels(count+1:count+no_feature_vector_class4(i,1),:);
    count=count+no_feature_vector_class4(i,1);
    
    data_class4{i}=temp';
    
    
end


% mu = mean(data_class4);
% sigma = std(data_class4);
% 
% for i = 1:size(data_class4,1)
%     norm_data_class4(i,:) = (data_class4(i,:)) - mu ./ sigma ;
% end
 

count

data_class5=cell(no_of_data_in_class5,1);
for i=1:no_of_data_in_class5
    temp=labels(count+1:count+no_feature_vector_class5(i,1),:);
    count=count+no_feature_vector_class5(i,1);
    
    data_class5{i}=temp';
    
end

% 
% mu = mean(data_class5);
% sigma = std(data_class5);
% 
% for i = 1:size(data_class5,1)
%     norm_data_class5(i,:) = (data_class5(i,:)) - mu ./ sigma ;
% end


count


%creating_training_data=[creating_training_data1;creating_training_data2;creating_training_data3;creating_training_data4;creating_training_data5];


class1_training_data=data_class1(1:round(0.7*no_of_data_in_class1));
class1_testing_data=data_class1(round(0.7*no_of_data_in_class1)+1:no_of_data_in_class1);
num_of_one_labeles=no_of_data_in_class1-round(0.7*no_of_data_in_class1);
labels_class1(1:num_of_one_labeles,1)=ones(num_of_one_labeles,1);


class2_training_data=data_class2(1:round(0.7*no_of_data_in_class2));
class2_testing_data=data_class2(round(0.7*no_of_data_in_class2)+1:no_of_data_in_class2);
num_of_one_labeles=no_of_data_in_class2-round(0.7*no_of_data_in_class2);
labels_class2(1:num_of_one_labeles,1)=2*ones(num_of_one_labeles,1);

class3_training_data=data_class3(1:round(0.7*no_of_data_in_class3));
class3_testing_data=data_class3(round(0.7*no_of_data_in_class3)+1:no_of_data_in_class3);
num_of_one_labeles=no_of_data_in_class3-round(0.7*no_of_data_in_class3);
labels_class3(1:num_of_one_labeles,1)=3*ones(num_of_one_labeles,1);

class4_training_data=data_class4(1:round(0.7*no_of_data_in_class4));
class4_testing_data=data_class4(round(0.7*no_of_data_in_class4)+1:no_of_data_in_class4);
num_of_one_labeles=no_of_data_in_class4-round(0.7*no_of_data_in_class4);
labels_class4(1:num_of_one_labeles,1)=4*ones(num_of_one_labeles,1);

class5_training_data=data_class5(1:round(0.7*no_of_data_in_class5));
class5_testing_data=data_class5(round(0.7*no_of_data_in_class5)+1:no_of_data_in_class5);
num_of_one_labeles=no_of_data_in_class5-round(0.7*no_of_data_in_class5);
labels_class5(1:num_of_one_labeles,1)=5*ones(num_of_one_labeles,1);



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
     proab_class1(i,:)= temp./sum(temp);
    
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
     proab_class2(i,:)=temp./sum(temp);
    
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
     proab_class3(i,:)=temp./sum(temp);
    
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
     proab_class4(i,:)=temp./sum(temp);
    
end



scores_1=get_scores_DTW(class5_testing_data,class1_training_data);
scores_2=get_scores_DTW(class5_testing_data,class2_training_data);
scores_3=get_scores_DTW(class5_testing_data,class3_training_data);
scores_4=get_scores_DTW(class5_testing_data,class4_training_data);
scores_5=get_scores_DTW(class5_testing_data,class5_training_data);

scores_class5=[scores_1 scores_2 scores_3 scores_4 scores_5];

for i=1:length(scores_class5)-2
    temp=1./scores_class5(i,:) ;
    [ value testing_labels_class5(i)] =max(temp);
     proab_class5(i,:)=temp./sum(temp);
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
n_samples= size(labels_class1,1)+size(labels_class2,1)+size(labels_class3,1)+size(labels_class4,1)+size(labels_class5,1);
accuracy=(accuracy/n_samples)*100



precision
