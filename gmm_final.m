clc
close all
clear all
num_of_clsses=8;
model=cell(num_of_clsses,3);
num_of_clusters=30;
dim = 30;
number_of_gaussians=4;
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


% fid = fopen('Image_SVM_Train.txt','w+');
%   for i = 1 : size(training_data,1)
%      
%       fprintf(fid,'%d ', train_labels(i));    
%      
%       for j = 1: size(training_data,2)
%      
%           fprintf(fid,'%d:%f ', j,training_data(i,j));
%           %fprintf(fid,'%f ', j,training_data(i,j));
%       end
%       
%       fprintf(fid,'\n');
%      
%   end
%   fclose(fid);
%  
%    fid = fopen('Image_SVM_Test.txt','w+');
%      
%   for i = 1 : size(testing_data,1)
%      
%       fprintf(fid,'%d ', test_labels(i));    
%      
%       for j = 1: size(testing_data,2)
%      
%           fprintf(fid,'%d:%f ', j,testing_data(i,j));
%           
%       end
%       
%       fprintf(fid,'\n');
%      
%   end
%   fclose(fid);
 



%%% pca %%%%%%%%%%% 
%  250 dimension 

covar=cov(training_data);
[cov_length covar_width]=size(covar);
[u v]=eig(covar);
%u = pca(training_data);

reduced_data=training_data*u(:,cov_length-dim:cov_length);

reduced_testing_data=testing_data*u(:,cov_length-dim:cov_length);

fid = fopen('Image_PCASVMSV_Train.txt','w+');
  for i = 1 : size(reduced_data,1)
     
      fprintf(fid,'%d ', train_labels(i));    
     
      for j = 1: size(reduced_data,2)
     
          fprintf(fid,'%d:%f ', j,reduced_data(i,j));
          %fprintf(fid,'%f ', j,training_data(i,j));
      end
      
      fprintf(fid,'\n');
     
  end
  fclose(fid);
 
   fid = fopen('Image_PCASVMSV_Test.txt','w+');
     
  for i = 1 : size(reduced_testing_data,1)
     
      fprintf(fid,'%d ', test_labels(i));    
     
      for j = 1: size(reduced_testing_data,2)
     
          fprintf(fid,'%d:%f ', j,reduced_testing_data(i,j));
          
      end
      
      fprintf(fid,'\n');
     
  end
  fclose(fid);


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


%%%%%%%%%%%GMMM training%%%%%%%%%%%%

[mean,sigma,err,priors]=em_gaussian(number_of_gaussians,training_class1);
model(1,1)={[mean]};
model(1,2)={[sigma]};
model(1,3)={[priors]};


[mean,sigma,err,priors]=em_gaussian(number_of_gaussians,training_class2);
model(2,1)={[mean]};
model(2,2)={[sigma]};
model(2,3)={[priors]};



[mean,sigma,err,priors]=em_gaussian(number_of_gaussians,training_class3);
model(3,1)={[mean]};
model(3,2)={[sigma]};
model(3,3)={[priors]};
[mean,sigma,err,priors]=em_gaussian(number_of_gaussians,training_class4);
model(4,1)={[mean]};
model(4,2)={[sigma]};
model(4,3)={[priors]};

[mean,sigma,err,priors]=em_gaussian(number_of_gaussians,training_class5);
model(5,1)={[mean]};
model(5,2)={[sigma]};
model(5,3)={[priors]};

[mean,sigma,err,priors]=em_gaussian(number_of_gaussians,training_class6);
model(6,1)={[mean]};
model(6,2)={[sigma]};
model(6,3)={[priors]};

[mean,sigma,err,priors]=em_gaussian(number_of_gaussians,training_class7);
model(7,1)={[mean]};
model(7,2)={[sigma]};
model(7,3)={[priors]};

[mean,sigma,err,priors]=em_gaussian(number_of_gaussians,training_class8);
model(8,1)={[mean]};
model(8,2)={[sigma]};
model(8,3)={[priors]};


%%% testing class 1




[class1_testing_length  class1_testing_width]=size(testing_class1);
testing_labels_class1=zeros(class1_testing_length,1);
proab_class1=zeros(class1_testing_length,num_of_clsses);
for iter=1:(class1_testing_length)
prob_each_class=zeros(1,num_of_clsses);
estimate=0;
for k=1:num_of_clsses
    mean=model{k,1};
    variance=model{k,2};
    prior=model{k,3};
    temp=0;
    for i=1:number_of_gaussians    
    temp=temp+prior(i)*mvnpdf(testing_class1(iter,:),mean(i,:),variance(:,:,i));
    end
    prob_each_class(k)=temp;
    estimate=estimate+temp;
end
[ value testing_labels_class1(iter)] =max(prob_each_class);
proab_class1(iter,:)=prob_each_class./estimate;
end


%%%%%testing data2

[class2_testing_length  class2_testing_width]=size(testing_class2);
testing_labels_class2=zeros(class2_testing_length,1);
proab_class2=zeros(class2_testing_length,num_of_clsses);

for iter=1:class2_testing_length
prob_each_class=zeros(1,num_of_clsses);
estimate=0;
for k=1:num_of_clsses
    mean=model{k,1};
    variance=model{k,2};
    prior=model{k,3};
    temp=0;
    for i=1:number_of_gaussians    
    temp=temp+prior(i)*mvnpdf(testing_class2(iter,:),mean(i,:),variance(:,:,i));
    end
    prob_each_class(k)=temp;
    estimate=estimate+temp;
end
[ value testing_labels_class2(iter)] =max(prob_each_class);
proab_class2(iter,:)=prob_each_class./estimate;
end

[class3_testing_length  class3_testing_width]=size(testing_class3);
testing_labels_class3=zeros(class3_testing_length,1);
proab_class3=zeros(class3_testing_length,num_of_clsses);

for iter=1:class3_testing_length
prob_each_class=zeros(1,num_of_clsses);
estimate=0;

for k=1:num_of_clsses
    mean=model{k,1};
    variance=model{k,2};
    prior=model{k,3};
    temp=0;
    for i=1:number_of_gaussians    
    temp=temp+prior(i)*mvnpdf(testing_class3(iter,:),mean(i,:),variance(:,:,i));
    end
    prob_each_class(k)=temp;
    estimate=estimate+temp;

end
[ value testing_labels_class3(iter)] =max(prob_each_class);
proab_class3(iter,:)=prob_each_class./estimate;

end

%% testing4


[class4_testing_length  class4_testing_width]=size(testing_class4);
testing_labels_class4=zeros(class4_testing_length,1);
proab_class3=zeros(class4_testing_length,num_of_clsses);

for iter=1:class4_testing_length
prob_each_class=zeros(1,num_of_clsses);
estimate=0;

for k=1:num_of_clsses
    mean=model{k,1};
    variance=model{k,2};
    prior=model{k,3};
    temp=0;
    for i=1:number_of_gaussians    
    temp=temp+prior(i)*mvnpdf(testing_class4(iter,:),mean(i,:),variance(:,:,i));
    end
    prob_each_class(k)=temp;
    estimate=estimate+temp;

end
[ value testing_labels_class4(iter)] =max(prob_each_class);
proab_class4(iter,:)=prob_each_class./estimate;

end
%% testing5

[class5_testing_length  class5_testing_width]=size(testing_class5);
testing_labels_class5=zeros(class5_testing_length,1);
proab_class5=zeros(class5_testing_length,num_of_clsses);

for iter=1:class5_testing_length
prob_each_class=zeros(1,num_of_clsses);
estimate=0;

for k=1:num_of_clsses
    mean=model{k,1};
    variance=model{k,2};
    prior=model{k,3};
    temp=0;
    for i=1:number_of_gaussians    
    temp=temp+prior(i)*mvnpdf(testing_class5(iter,:),mean(i,:),variance(:,:,i));
    end
    prob_each_class(k)=temp;
    estimate=estimate+temp;

end
[ value testing_labels_class5(iter)] =max(prob_each_class);
proab_class5(iter,:)=prob_each_class./estimate;

end
%% testing 6

[class6_testing_length  class6_testing_width]=size(testing_class6);
testing_labels_class6=zeros(class6_testing_length,1);
proab_class6=zeros(class6_testing_length,num_of_clsses);

for iter=1:class6_testing_length
prob_each_class=zeros(1,num_of_clsses);
estimate=0;

for k=1:num_of_clsses
    mean=model{k,1};
    variance=model{k,2};
    prior=model{k,3};
    temp=0;
    for i=1:number_of_gaussians    
    temp=temp+prior(i)*mvnpdf(testing_class6(iter,:),mean(i,:),variance(:,:,i));
    end
    prob_each_class(k)=temp;
    estimate=estimate+temp;

end
[ value testing_labels_class6(iter)] =max(prob_each_class);
proab_class6(iter,:)=prob_each_class./estimate;

end
%%% testing 7

[class7_testing_length  class7_testing_width]=size(testing_class7);
testing_labels_class7=zeros(class7_testing_length,1);
proab_class7=zeros(class7_testing_length,num_of_clsses);

for iter=1:class7_testing_length
prob_each_class=zeros(1,num_of_clsses);
estimate=0;

for k=1:num_of_clsses
    mean=model{k,1};
    variance=model{k,2};
    prior=model{k,3};
    temp=0;
    for i=1:number_of_gaussians    
    temp=temp+prior(i)*mvnpdf(testing_class7(iter,:),mean(i,:),variance(:,:,i));
    end
    prob_each_class(k)=temp;
    estimate=estimate+temp;

end
[ value testing_labels_class7(iter)] =max(prob_each_class);
proab_class7(iter,:)=prob_each_class./estimate;

end
%%%% testing 8

[class8_testing_length  class8_testing_width]=size(testing_class8);
testing_labels_class8=zeros(class8_testing_length,1);
proab_class8=zeros(class8_testing_length,num_of_clsses);

for iter=1:class8_testing_length
prob_each_class=zeros(1,num_of_clsses);
estimate=0;

for k=1:num_of_clsses
    mean=model{k,1};
    variance=model{k,2};
    prior=model{k,3};
    temp=0;
    for i=1:number_of_gaussians    
    temp=temp+prior(i)*mvnpdf(testing_class8(iter,:),mean(i,:),variance(:,:,i));
    end
    prob_each_class(k)=temp;
    estimate=estimate+temp;

end
[ value testing_labels_class8(iter)] =max(prob_each_class);
proab_class8(iter,:)=prob_each_class./estimate;

end

%%%%%%calculating performance matrix
num_of_clsses
clear confusion_matrix;
confusion_matrix=zeros(num_of_clsses,num_of_clsses);

confusion_matrix(1,1)=sum((testing_labels_class1)==1);
confusion_matrix(1,2)=sum((testing_labels_class1)==2);
confusion_matrix(1,3)=sum((testing_labels_class1)==3);
confusion_matrix(1,4)=sum((testing_labels_class1)==4);
confusion_matrix(1,5)=sum((testing_labels_class1)==5);
confusion_matrix(1,6)=sum((testing_labels_class1)==6);
confusion_matrix(1,7)=sum((testing_labels_class1)==7);
confusion_matrix(1,8)=sum((testing_labels_class1)==8);


confusion_matrix(2,1)=sum((testing_labels_class2)==1);
confusion_matrix(2,2)=sum((testing_labels_class2)==2);
confusion_matrix(2,3)=sum((testing_labels_class2)==3);
confusion_matrix(2,4)=sum((testing_labels_class2)==4);
confusion_matrix(2,5)=sum((testing_labels_class2)==5);
confusion_matrix(2,6)=sum((testing_labels_class2)==6);
confusion_matrix(2,7)=sum((testing_labels_class2)==7);
confusion_matrix(2,8)=sum((testing_labels_class2)==8);


confusion_matrix(3,1)=sum((testing_labels_class3)==1);
confusion_matrix(3,2)=sum((testing_labels_class3)==2);
confusion_matrix(3,3)=sum((testing_labels_class3)==3);
confusion_matrix(3,4)=sum((testing_labels_class3)==4);
confusion_matrix(3,5)=sum((testing_labels_class3)==5);
confusion_matrix(3,6)=sum((testing_labels_class3)==6);
confusion_matrix(3,7)=sum((testing_labels_class3)==7);
confusion_matrix(3,8)=sum((testing_labels_class3)==8);

confusion_matrix(4,1)=sum((testing_labels_class4)==1);
confusion_matrix(4,2)=sum((testing_labels_class4)==2);
confusion_matrix(4,3)=sum((testing_labels_class4)==3);
confusion_matrix(4,4)=sum((testing_labels_class4)==4);
confusion_matrix(4,5)=sum((testing_labels_class4)==5);
confusion_matrix(4,6)=sum((testing_labels_class4)==6);
confusion_matrix(4,7)=sum((testing_labels_class4)==7);
confusion_matrix(4,8)=sum((testing_labels_class4)==8);

confusion_matrix(5,1)=sum((testing_labels_class5)==1);
confusion_matrix(5,2)=sum((testing_labels_class5)==2);
confusion_matrix(5,3)=sum((testing_labels_class5)==3);
confusion_matrix(5,4)=sum((testing_labels_class5)==4);
confusion_matrix(5,5)=sum((testing_labels_class5)==5);
confusion_matrix(5,6)=sum((testing_labels_class5)==6);
confusion_matrix(5,7)=sum((testing_labels_class5)==7);
confusion_matrix(5,8)=sum((testing_labels_class5)==8);

confusion_matrix(6,1)=sum((testing_labels_class6)==1);
confusion_matrix(6,2)=sum((testing_labels_class6)==2);
confusion_matrix(6,3)=sum((testing_labels_class6)==3);
confusion_matrix(6,4)=sum((testing_labels_class6)==4);
confusion_matrix(6,5)=sum((testing_labels_class6)==5);
confusion_matrix(6,6)=sum((testing_labels_class6)==6);
confusion_matrix(6,7)=sum((testing_labels_class6)==7);
confusion_matrix(6,8)=sum((testing_labels_class6)==8);

confusion_matrix(7,1)=sum((testing_labels_class7)==1);
confusion_matrix(7,2)=sum((testing_labels_class7)==2);
confusion_matrix(7,3)=sum((testing_labels_class7)==3);
confusion_matrix(7,4)=sum((testing_labels_class7)==4);
confusion_matrix(7,5)=sum((testing_labels_class7)==5);
confusion_matrix(7,6)=sum((testing_labels_class7)==6);
confusion_matrix(7,7)=sum((testing_labels_class7)==7);
confusion_matrix(7,8)=sum((testing_labels_class7)==8);

confusion_matrix(8,1)=sum((testing_labels_class8)==1);
confusion_matrix(8,2)=sum((testing_labels_class8)==2);
confusion_matrix(8,3)=sum((testing_labels_class8)==3);
confusion_matrix(8,4)=sum((testing_labels_class8)==4);
confusion_matrix(8,5)=sum((testing_labels_class8)==5);
confusion_matrix(8,6)=sum((testing_labels_class8)==6);
confusion_matrix(8,7)=sum((testing_labels_class8)==7);
confusion_matrix(8,8)=sum((testing_labels_class8)==8);




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

%%%%%%%%%%%%%%%%%%%% roc curves%%%%%%%%%%%%%
prob_all=[proab_class1;proab_class2;proab_class3;proab_class4;proab_class5;proab_class6;proab_class7;proab_class8];
labels=test_labels;
classified_labels=[testing_labels_class1;testing_labels_class2;testing_labels_class3;testing_labels_class4;testing_labels_class5;testing_labels_class6;testing_labels_class7;testing_labels_class8];
save './ImageData/PCASV_Labels_g4.mat' labels
save './ImageData/PCASV_Probability_g4.mat' prob_all


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
save './ImageData/PCASV_TPScores_g4.mat' true_postive_score
save './ImageData/PCASV_FPScores_g4.mat' false_postive_score
