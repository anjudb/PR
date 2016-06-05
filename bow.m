function [accuracy]=bow(num_of_clusters)


num_of_clsses=8;
model=cell(num_of_clsses,3);
%num_of_clusters=50;
number_of_gaussians=5;
feature_directory=dir('/home/priyanka/Documents/MATLAB/ps3_ver3/GMM/forest/forest_features/*');
no_of_data_in_class1=length(feature_directory)-2;

creating_training_data1=[];
no_feature_vector_class1=zeros(no_of_data_in_class1,1);
for i = 3:length(feature_directory)
  name = sprintf('/home/priyanka/Documents/MATLAB/ps3_ver3/GMM/forest/forest_features/%s',feature_directory(i).name);
    fid = load(name);
  
   [fid_lenght fid_width]=size(fid);
   no_feature_vector_class1(i-2,1)=fid_lenght;
  creating_training_data1=[ creating_training_data1 ; fid];
end



creating_training_data2=[];

feature_directory=dir('/home/priyanka/Documents/MATLAB/ps3_ver3/GMM/coast/coast_features/*');
no_of_data_in_class2=length(feature_directory)-2;
no_feature_vector_class2=zeros(no_of_data_in_class2,1);

creating_training_data2=[];
for i = 3:length(feature_directory)
    name = sprintf('/home/priyanka/Documents/MATLAB/ps3_ver3/GMM/coast/coast_features/%s',feature_directory(i).name);
    fid = load(name);
  
   [fid_lenght fid_width]=size(fid);
      no_feature_vector_class2(i-2,1)=fid_lenght;
   creating_training_data2=[ creating_training_data2 ; fid];
end


feature_directory=dir('/home/priyanka/Documents/MATLAB/ps3_ver3/GMM/street/street_features/*');
creating_training_data3=[];
no_of_data_in_class3=length(feature_directory)-2;
no_feature_vector_class3=zeros(no_of_data_in_class3,1);

for i = 3:length(feature_directory)
  name = sprintf('/home/priyanka/Documents/MATLAB/ps3_ver3/GMM/street/street_features/%s',feature_directory(i).name);
    fid = load(name);
  
   [fid_lenght fid_width]=size(fid);
    no_feature_vector_class3(i-2,1)=fid_lenght;
    
  creating_training_data3=[ creating_training_data3 ; fid];
end

%%%%% other 5 classes%%%%
feature_directory=dir('/home/priyanka/Documents/MATLAB/ps3_ver3/GMM/highway/highway_features/*');
creating_training_data4=[];
no_of_data_in_class4=length(feature_directory)-2;
no_feature_vector_class4=zeros(no_of_data_in_class4,1);

for i = 3:length(feature_directory)
  name = sprintf('/home/priyanka/Documents/MATLAB/ps3_ver3/GMM/highway/highway_features/%s',feature_directory(i).name);
    fid = load(name);
  
   [fid_lenght fid_width]=size(fid);
    no_feature_vector_class4(i-2,1)=fid_lenght;
    
  creating_training_data4=[ creating_training_data4 ; fid];
end

feature_directory=dir('/home/priyanka/Documents/MATLAB/ps3_ver3/GMM/insidecity/insidecity_features/*');
creating_training_data5=[];
no_of_data_in_class5=length(feature_directory)-2;
no_feature_vector_class5=zeros(no_of_data_in_class5,1);

for i = 3:length(feature_directory)
  name = sprintf('/home/priyanka/Documents/MATLAB/ps3_ver3/GMM/insidecity/insidecity_features/%s',feature_directory(i).name);
    fid = load(name);
  
   [fid_lenght fid_width]=size(fid);
    no_feature_vector_class5(i-2,1)=fid_lenght;
    
  creating_training_data5=[ creating_training_data5 ; fid];
end

feature_directory=dir('/home/priyanka/Documents/MATLAB/ps3_ver3/GMM/mountain/mountain_features/*');
creating_training_data6=[];
no_of_data_in_class6=length(feature_directory)-2;
no_feature_vector_class6=zeros(no_of_data_in_class6,1);

for i = 3:length(feature_directory)
  name = sprintf('/home/priyanka/Documents/MATLAB/ps3_ver3/GMM/mountain/mountain_features/%s',feature_directory(i).name);
    fid = load(name);
  
   [fid_lenght fid_width]=size(fid);
    no_feature_vector_class6(i-2,1)=fid_lenght;
    
  creating_training_data6=[ creating_training_data6 ; fid];
end

feature_directory=dir('/home/priyanka/Documents/MATLAB/ps3_ver3/GMM/opencountry/opencountry_features/*');
creating_training_data7=[];
no_of_data_in_class7=length(feature_directory)-2;
no_feature_vector_class7=zeros(no_of_data_in_class7,1);

for i = 3:length(feature_directory)
  name = sprintf('/home/priyanka/Documents/MATLAB/ps3_ver3/GMM/opencountry/opencountry_features/%s',feature_directory(i).name);
    fid = load(name);
  
   [fid_lenght fid_width]=size(fid);
    no_feature_vector_class7(i-2,1)=fid_lenght;
    
  creating_training_data7=[ creating_training_data7 ; fid];
end

feature_directory=dir('/home/priyanka/Documents/MATLAB/ps3_ver3/GMM/tallbuilding/tallbuilding_features/*');
creating_training_data8=[];
no_of_data_in_class8=length(feature_directory)-2;
no_feature_vector_class8=zeros(no_of_data_in_class8,1);

for i = 3:length(feature_directory)
  name = sprintf('/home/priyanka/Documents/MATLAB/ps3_ver3/GMM/tallbuilding/tallbuilding_features/%s',feature_directory(i).name);
    fid = load(name);
  
   [fid_lenght fid_width]=size(fid);
    no_feature_vector_class8(i-2,1)=fid_lenght;
    
  creating_training_data8=[ creating_training_data8 ; fid];
end


creating_training_data=[creating_training_data1;creating_training_data2;creating_training_data3;creating_training_data4;creating_training_data5;creating_training_data6;creating_training_data7;creating_training_data8];

[labels,cluster_centers,sumd,D]=kmeans(creating_training_data,num_of_clusters,'MaxIter',500);




count=0;
data_class1=zeros(no_of_data_in_class1,num_of_clusters);
for i=1:no_of_data_in_class1
    temp=labels(count+1:count+no_feature_vector_class1(i,1),:);
    count=count+no_feature_vector_class1(i,1);
    num_element=hist(temp',num_of_clusters);
    data_class1(i,:)=num_element/sum(num_element);
    
    
end

% mu = mean(data_class1);
% sigma = std(data_class1);
% 
% for i = 1:size(data_class1,1)
%     norm_data_class1(i,:) = (data_class1(i,:)) - mu ./ sigma ;
% end


count

data_class2=zeros(no_of_data_in_class2,num_of_clusters);
for i=1:no_of_data_in_class2
    temp=labels(count+1:count+no_feature_vector_class2(i,1),:);
    count=count+no_feature_vector_class2(i,1);
    num_element=hist(temp,num_of_clusters);
    data_class2(i,:)=num_element/sum(num_element);
    
    
end


% mu = mean(data_class2);
% sigma = std(data_class2);
% 
% for i = 1:size(data_class2,1)
%     norm_data_class2(i,:) = (data_class2(i,:)) - mu ./ sigma ;
% end

count

data_class3=zeros(no_of_data_in_class3,num_of_clusters);
for i=1:no_of_data_in_class3
    temp=labels(count+1:count+no_feature_vector_class3(i,1),:);
    count=count+no_feature_vector_class3(i,1);
    num_element=hist(temp,num_of_clusters);
    data_class3(i,:)=num_element/sum(num_element);
    
    
end

count

data_class4=zeros(no_of_data_in_class4,num_of_clusters);
for i=1:no_of_data_in_class4
    temp=labels(count+1:count+no_feature_vector_class4(i,1),:);
    count=count+no_feature_vector_class4(i,1);
    num_element=hist(temp,num_of_clusters);
    data_class4(i,:)=num_element/sum(num_element);
    
    
end

count

data_class5=zeros(no_of_data_in_class5,num_of_clusters);
for i=1:no_of_data_in_class5
    temp=labels(count+1:count+no_feature_vector_class5(i,1),:);
    count=count+no_feature_vector_class5(i,1);
    num_element=hist(temp,num_of_clusters);
    data_class5(i,:)=num_element/sum(num_element);
    
    
end

count

data_class6=zeros(no_of_data_in_class6,num_of_clusters);
for i=1:no_of_data_in_class6
    temp=labels(count+1:count+no_feature_vector_class6(i,1),:);
    count=count+no_feature_vector_class6(i,1);
    num_element=hist(temp,num_of_clusters);
    data_class6(i,:)=num_element/sum(num_element);
    
    
end

count

data_class7=zeros(no_of_data_in_class7,num_of_clusters);
for i=1:no_of_data_in_class7
    temp=labels(count+1:count+no_feature_vector_class7(i,1),:);
    count=count+no_feature_vector_class7(i,1);
    num_element=hist(temp,num_of_clusters);
    data_class7(i,:)=num_element/sum(num_element);
    
    
end

count

data_class8=zeros(no_of_data_in_class8,num_of_clusters);
for i=1:no_of_data_in_class8
    temp=labels(count+1:count+no_feature_vector_class8(i,1),:);
    count=count+no_feature_vector_class8(i,1);
    num_element=hist(temp,num_of_clusters);
    data_class8(i,:)=num_element/sum(num_element);
    
    
end


%%%%%%%%%%%%%%%%%%dividing data into trainnig and testing%%%%%%%%%%%%%%%%

class1_training_data=data_class1(1:round(0.7*no_of_data_in_class1),:);
class1_testing_data=data_class1(round(0.7*no_of_data_in_class1)+1:no_of_data_in_class1,:);
num_of_one_labeles=no_of_data_in_class1-round(0.7*no_of_data_in_class1);
labels_class1(1:num_of_one_labeles,1)=ones(1:num_of_one_labeles,1);


class2_training_data=data_class2(1:round(0.7*no_of_data_in_class2),:);
class2_testing_data=data_class2(round(0.7*no_of_data_in_class2)+1:no_of_data_in_class2,:);
num_of_one_labeles=no_of_data_in_class2-round(0.7*no_of_data_in_class2);
labels_class2(1:num_of_one_labeles,1)=2*ones(1:num_of_one_labeles,1);

class3_training_data=data_class3(1:round(0.7*no_of_data_in_class3),:);
class3_testing_data=data_class3(round(0.7*no_of_data_in_class3)+1:no_of_data_in_class3,:);
num_of_one_labeles=no_of_data_in_class3-round(0.7*no_of_data_in_class3);
labels_class3(1:num_of_one_labeles,1)=3*ones(1:num_of_one_labeles,1);



class4_training_data=data_class4(1:round(0.7*no_of_data_in_class4),:);
class4_testing_data=data_class4(round(0.7*no_of_data_in_class4)+1:no_of_data_in_class4,:);
num_of_one_labeles=no_of_data_in_class4-round(0.7*no_of_data_in_class4);
labels_class4(1:num_of_one_labeles,1)=4*ones(1:num_of_one_labeles,1);

class5_training_data=data_class5(1:round(0.7*no_of_data_in_class5),:);
class5_testing_data=data_class5(round(0.7*no_of_data_in_class5)+1:no_of_data_in_class5,:);
num_of_one_labeles=no_of_data_in_class5-round(0.7*no_of_data_in_class5);
labels_class5(1:num_of_one_labeles,1)=5*ones(1:num_of_one_labeles,1);


class6_training_data=data_class6(1:round(0.7*no_of_data_in_class6),:);
class6_testing_data=data_class3(round(0.7*no_of_data_in_class3)+1:no_of_data_in_class3,:);
num_of_one_labeles=no_of_data_in_class6-round(0.7*no_of_data_in_class6);
labels_class6(1:num_of_one_labeles,1)=6*ones(1:num_of_one_labeles,1);

class7_training_data=data_class7(1:round(0.7*no_of_data_in_class7),:);
class7_testing_data=data_class7(round(0.7*no_of_data_in_class7)+1:no_of_data_in_class7,:);
num_of_one_labeles=no_of_data_in_class7-round(0.7*no_of_data_in_class7);
labels_class7(1:num_of_one_labeles,1)=7*ones(1:num_of_one_labeles,1);

class8_training_data=data_class8(1:round(0.7*no_of_data_in_class8),:);
class8_testing_data=data_class8(round(0.7*no_of_data_in_class8)+1:no_of_data_in_class8,:);
num_of_one_labeles=no_of_data_in_class8-round(0.7*no_of_data_in_class8);
labels_class8(1:num_of_one_labeles,1)=8*ones(1:num_of_one_labeles,1);



%%%%%%%%%%%%%%%%% appling gmm trainig%%%%%%%%%%%%%%%%

[mean,sigma,err,priors]=em_gaussian(number_of_gaussians,class1_training_data);
model(1,1)={[mean]};
model(1,2)={[sigma]};
model(1,3)={[priors]};


[mean,sigma,err,priors]=em_gaussian(number_of_gaussians,class2_training_data);
model(2,1)={[mean]};
model(2,2)={[sigma]};
model(2,3)={[priors]};



[mean,sigma,err,priors]=em_gaussian(number_of_gaussians,class3_training_data);
model(3,1)={[mean]};
model(3,2)={[sigma]};
model(3,3)={[priors]};

[mean,sigma,err,priors]=em_gaussian(number_of_gaussians,class4_training_data);
model(4,1)={[mean]};
model(4,2)={[sigma]};
model(4,3)={[priors]};

[mean,sigma,err,priors]=em_gaussian(number_of_gaussians,class5_training_data);
model(5,1)={[mean]};
model(5,2)={[sigma]};
model(5,3)={[priors]};

[mean,sigma,err,priors]=em_gaussian(number_of_gaussians,class6_training_data);
model(6,1)={[mean]};
model(6,2)={[sigma]};
model(6,3)={[priors]};

[mean,sigma,err,priors]=em_gaussian(number_of_gaussians,class7_training_data);
model(7,1)={[mean]};
model(7,2)={[sigma]};
model(7,3)={[priors]};

[mean,sigma,err,priors]=em_gaussian(number_of_gaussians,class8_training_data);
model(8,1)={[mean]};
model(8,2)={[sigma]};
model(8,3)={[priors]};
%%%%%%%%%%%%%GMM testing%%%%%%%%%%%%%%%%
[class1_testing_length  class1_testing_width]=size(class1_testing_data);
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
    temp=temp+prior(i)*mvnpdf(class1_testing_data(iter,:),mean(i,:),variance(:,:,i));
    end
    prob_each_class(k)=temp;
    estimate=estimate+temp;
end
[ value testing_labels_class1(iter)] =max(prob_each_class);
proab_class1(iter,:)=prob_each_class./estimate;
end


%%%%%testing data2

[class2_testing_length  class2_testing_width]=size(class2_testing_data);
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
    temp=temp+prior(i)*mvnpdf(class2_testing_data(iter,:),mean(i,:),variance(:,:,i));
    end
    prob_each_class(k)=temp;
    estimate=estimate+temp;
end
[ value testing_labels_class2(iter)] =max(prob_each_class);
proab_class2(iter,:)=prob_each_class./estimate;
end



[class3_testing_length  class3_testing_width]=size(class3_testing_data);
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
    temp=temp+prior(i)*mvnpdf(class3_testing_data(iter,:),mean(i,:),variance(:,:,i));
    end
    prob_each_class(k)=temp;
    estimate=estimate+temp;

end
[ value testing_labels_class3(iter)] =max(prob_each_class);
proab_class3(iter,:)=prob_each_class./estimate;

end

[class4_testing_length  class4_testing_width]=size(class4_testing_data);
testing_labels_class4=zeros(class4_testing_length,1);
proab_class4=zeros(class4_testing_length,num_of_clsses);
for iter=1:(class4_testing_length)
prob_each_class=zeros(1,num_of_clsses);
estimate=0;
for k=1:num_of_clsses
    mean=model{k,1};
    variance=model{k,2};
    prior=model{k,3};
    temp=0;
    for i=1:number_of_gaussians    
    temp=temp+prior(i)*mvnpdf(class4_testing_data(iter,:),mean(i,:),variance(:,:,i));
    end
    prob_each_class(k)=temp;
    estimate=estimate+temp;
end
[ value testing_labels_class4(iter)] =max(prob_each_class);
proab_class4(iter,:)=prob_each_class./estimate;
end

[class5_testing_length  class5_testing_width]=size(class5_testing_data);
testing_labels_class5=zeros(class5_testing_length,1);
proab_class5=zeros(class5_testing_length,num_of_clsses);
for iter=1:(class5_testing_length)
prob_each_class=zeros(1,num_of_clsses);
estimate=0;
for k=1:num_of_clsses
    mean=model{k,1};
    variance=model{k,2};
    prior=model{k,3};
    temp=0;
    for i=1:number_of_gaussians    
    temp=temp+prior(i)*mvnpdf(class5_testing_data(iter,:),mean(i,:),variance(:,:,i));
    end
    prob_each_class(k)=temp;
    estimate=estimate+temp;
end
[ value testing_labels_class5(iter)] =max(prob_each_class);
proab_class5(iter,:)=prob_each_class./estimate;
end

[class6_testing_length  class6_testing_width]=size(class6_testing_data);
testing_labels_class6=zeros(class6_testing_length,1);
proab_class6=zeros(class6_testing_length,num_of_clsses);
for iter=1:(class6_testing_length)
prob_each_class=zeros(1,num_of_clsses);
estimate=0;
for k=1:num_of_clsses
    mean=model{k,1};
    variance=model{k,2};
    prior=model{k,3};
    temp=0;
    for i=1:number_of_gaussians    
    temp=temp+prior(i)*mvnpdf(class6_testing_data(iter,:),mean(i,:),variance(:,:,i));
    end
    prob_each_class(k)=temp;
    estimate=estimate+temp;
end
[ value testing_labels_class6(iter)] =max(prob_each_class);
proab_class6(iter,:)=prob_each_class./estimate;
end

[class7_testing_length  class7_testing_width]=size(class7_testing_data);
testing_labels_class7=zeros(class7_testing_length,1);
proab_class7=zeros(class7_testing_length,num_of_clsses);
for iter=1:(class7_testing_length)
prob_each_class=zeros(1,num_of_clsses);
estimate=0;
for k=1:num_of_clsses
    mean=model{k,1};
    variance=model{k,2};
    prior=model{k,3};
    temp=0;
    for i=1:number_of_gaussians    
    temp=temp+prior(i)*mvnpdf(class7_testing_data(iter,:),mean(i,:),variance(:,:,i));
    end
    prob_each_class(k)=temp;
    estimate=estimate+temp;
end
[ value testing_labels_class7(iter)] =max(prob_each_class);
proab_class7(iter,:)=prob_each_class./estimate;
end


[class8_testing_length  class8_testing_width]=size(class8_testing_data);
testing_labels_class8=zeros(class8_testing_length,1);
proab_class8=zeros(class8_testing_length,num_of_clsses);
for iter=1:(class8_testing_length)
prob_each_class=zeros(1,num_of_clsses);
estimate=0;
for k=1:num_of_clsses
    mean=model{k,1};
    variance=model{k,2};
    prior=model{k,3};
    temp=0;
    for i=1:number_of_gaussians    
    temp=temp+prior(i)*mvnpdf(class8_testing_data(iter,:),mean(i,:),variance(:,:,i));
    end
    prob_each_class(k)=temp;
    estimate=estimate+temp;
end
[ value testing_labels_class8(iter)] =max(prob_each_class);
proab_class8(iter,:)=prob_each_class./estimate;
end

%%%%%%%%%%%%%%%%%%%confusion matrix%%%%%%%%%%%%%%%%%%
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
labels=[labels_class1;labels_class2;labels_class3;labels_class4;labels_class5;labels_class6;labels_class7;labels_class8];
classified_labels=[testing_labels_class1;testing_labels_class2;testing_labels_class3;testing_labels_class4;testing_labels_class5;testing_labels_class6;testing_labels_class7;testing_labels_class8];
save './ImageData/BoW_Labels_g7.mat' labels
save './ImageData/BoW_Probability_g7.mat' prob_all


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
save './ImageData/BoW_TPScores_g7.mat' true_postive_score
save './ImageData/BoW_FPScores_g7.mat' false_postive_score
end