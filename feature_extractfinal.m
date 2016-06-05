clc
close all
clear all
number_of_gaussians=4;
num_of_clsses=8;
no_of_patches=36;
model=cell(num_of_clsses,3);
dim=4;


feature_directory=dir('/home/priyanka/Documents/MATLAB/ps3_ver3/GMM/forest/forest_features/*');
no_of_inmages_chosen=round(0.7*length(feature_directory));
creating_training_data1=[];
testing_data1=[];
label1=[];
no_of_features_per_image=36;
for i = 3:no_of_inmages_chosen
   name = sprintf('/home/priyanka/Documents/MATLAB/ps3_ver3/GMM/forest/forest_features/%s',feature_directory(i).name);
    fid = load(name);
    %vec_fid=reshape( fid.' ,1,numel(fid));
   creating_training_data1=[ creating_training_data1 ; fid];
   
end

no_of_training_data_class1=no_of_inmages_chosen-2;


for i = no_of_inmages_chosen+1:length(feature_directory)
   name = sprintf('/home/priyanka/Documents/MATLAB/ps3_ver3/GMM/forest/forest_features/%s',feature_directory(i).name);
    fid = load(name);
  % vec_fid=reshape( fid.' ,1,numel(fid)); 
   testing_data1=[testing_data1 ;fid];
   label1=[label1 ;1];
end
no_of_testing_data_class1=length(feature_directory)-(no_of_inmages_chosen);


feature_directory=dir('/home/priyanka/Documents/MATLAB/ps3_ver3/GMM/coast/coast_features/*');
no_of_inmages_chosen=round(0.7*length(feature_directory));

creating_training_data2=[];
testing_data2=[];
label2=[];

for i = 3:no_of_inmages_chosen
   name = sprintf('/home/priyanka/Documents/MATLAB/ps3_ver3/GMM/coast/coast_features/%s',feature_directory(i).name);
   fid = load(name);
  % vec_fid=reshape( fid.' ,1,numel(fid));
   creating_training_data2=[ creating_training_data2;fid];
end
% [mean,sigma,err,priors]=em_gaussian(number_of_gaussians,creating_training_data2);
% model(2,1)={[mean]};
% model(2,2)={[sigma]};
% model(2,3)={[priors]};

no_of_training_data_class2=no_of_inmages_chosen-2;



for i = no_of_inmages_chosen+1:length(feature_directory)
   name = sprintf('/home/priyanka/Documents/MATLAB/ps3_ver3/GMM/coast/coast_features/%s',feature_directory(i).name);
    fid = load(name);
  % vec_fid=reshape( fid.' ,1,numel(fid));
   testing_data2=[testing_data2 ;fid];
   label2=[label2;2];
end

no_of_testing_data_class2=length(feature_directory)-(no_of_inmages_chosen);

% class 3
feature_directory=dir('/home/priyanka/Documents/MATLAB/ps3_ver3/GMM/street/street_features/*');
no_of_inmages_chosen=round(0.7*length(feature_directory));
creating_training_data3=[];
testing_data3=[];
label3=[];

for i = 3:no_of_inmages_chosen
  name = sprintf('/home/priyanka/Documents/MATLAB/ps3_ver3/GMM/street/street_features/%s',feature_directory(i).name);
    fid = load(name);
  % vec_fid=reshape( fid.' ,1,numel(fid));
   creating_training_data3=[ creating_training_data3;fid];
end
% [mean,sigma,err,priors]=em_gaussian(number_of_gaussians,creating_training_data3);
% model(3,1)={[mean]};
% model(3,2)={[sigma]};
% model(3,3)={[priors]};

no_of_training_data_class3=no_of_inmages_chosen-2;


for i = no_of_inmages_chosen+1:length(feature_directory)
  name = sprintf('/home/priyanka/Documents/MATLAB/ps3_ver3/GMM/street/street_features/%s',feature_directory(i).name);
    fid = load(name);
   %vec_fid=reshape( fid.' ,1,numel(fid));
   testing_data3=[testing_data3 ;fid];
   label3=[label3 ;3];
end

no_of_testing_data_class3=length(feature_directory)-(no_of_inmages_chosen);

%%%% other classes
feature_directory=dir('/home/priyanka/Documents/MATLAB/ps3_ver3/GMM/highway/highway_features/*');
no_of_inmages_chosen=round(0.7*length(feature_directory));
creating_training_data4=[];
testing_data4=[];
label4=[];

for i = 3:no_of_inmages_chosen
 name = sprintf('/home/priyanka/Documents/MATLAB/ps3_ver3/GMM/highway/highway_features/%s',feature_directory(i).name);
    fid = load(name);
  % vec_fid=reshape( fid.' ,1,numel(fid));
   creating_training_data4=[ creating_training_data4;fid];
end
% [mean,sigma,err,priors]=em_gaussian(number_of_gaussians,creating_training_data3);
% model(3,1)={[mean]};
% model(3,2)={[sigma]};
% model(3,3)={[priors]};

no_of_training_data_class4=no_of_inmages_chosen-2;


for i = no_of_inmages_chosen+1:length(feature_directory)
   name = sprintf('/home/priyanka/Documents/MATLAB/ps3_ver3/GMM/highway/highway_features/%s',feature_directory(i).name);
    fid = load(name);
   %vec_fid=reshape( fid.' ,1,numel(fid));
   testing_data4=[testing_data4 ;fid];
   label4=[label4 ;3];
end

no_of_testing_data_class4=length(feature_directory)-(no_of_inmages_chosen);
%%% class 5

feature_directory=dir('/home/priyanka/Documents/MATLAB/ps3_ver3/GMM/insidecity/insidecity_features/*');
no_of_inmages_chosen=round(0.7*length(feature_directory));
creating_training_data5=[];
testing_data5=[];
label5=[];

for i = 3:no_of_inmages_chosen
  name = sprintf('/home/priyanka/Documents/MATLAB/ps3_ver3/GMM/insidecity/insidecity_features/%s',feature_directory(i).name);
    fid = load(name);
  % vec_fid=reshape( fid.' ,1,numel(fid));
   creating_training_data5=[ creating_training_data5;fid];
end
% [mean,sigma,err,priors]=em_gaussian(number_of_gaussians,creating_training_data3);
% model(3,1)={[mean]};
% model(3,2)={[sigma]};
% model(3,3)={[priors]};

no_of_training_data_class5=no_of_inmages_chosen-2;


for i = no_of_inmages_chosen+1:length(feature_directory)
    name = sprintf('/home/priyanka/Documents/MATLAB/ps3_ver3/GMM/insidecity/insidecity_features/%s',feature_directory(i).name);

    fid = load(name);
   %vec_fid=reshape( fid.' ,1,numel(fid));
   testing_data5=[testing_data5 ;fid];
   label5=[label5 ;3];
end

no_of_testing_data_class5=length(feature_directory)-(no_of_inmages_chosen);

%%%% class 6

feature_directory=dir('/home/priyanka/Documents/MATLAB/ps3_ver3/GMM/mountain/mountain_features/*');
no_of_inmages_chosen=round(0.7*length(feature_directory));
creating_training_data6=[];
testing_data6=[];
label6=[];

for i = 3:no_of_inmages_chosen
 name = sprintf('/home/priyanka/Documents/MATLAB/ps3_ver3/GMM/mountain/mountain_features/%s',feature_directory(i).name);
    fid = load(name);
  % vec_fid=reshape( fid.' ,1,numel(fid));
   creating_training_data6=[ creating_training_data6;fid];
end
% [mean,sigma,err,priors]=em_gaussian(number_of_gaussians,creating_training_data3);
% model(3,1)={[mean]};
% model(3,2)={[sigma]};
% model(3,3)={[priors]};

no_of_training_data_class6=no_of_inmages_chosen-2;


for i = no_of_inmages_chosen+1:length(feature_directory)
     name = sprintf('/home/priyanka/Documents/MATLAB/ps3_ver3/GMM/mountain/mountain_features/%s',feature_directory(i).name);

    fid = load(name);
   %vec_fid=reshape( fid.' ,1,numel(fid));
   testing_data6=[testing_data6 ;fid];
   label6=[label6 ;3];
end

no_of_testing_data_class6=length(feature_directory)-(no_of_inmages_chosen);
%%% class 7 

feature_directory=dir('/home/priyanka/Documents/MATLAB/ps3_ver3/GMM/opencountry/opencountry_features/*');
no_of_inmages_chosen=round(0.7*length(feature_directory));
creating_training_data7=[];
testing_data7=[];
label7=[];

for i = 3:no_of_inmages_chosen
 name = sprintf('/home/priyanka/Documents/MATLAB/ps3_ver3/GMM/opencountry/opencountry_features/%s',feature_directory(i).name);
    fid = load(name);
  % vec_fid=reshape( fid.' ,1,numel(fid));
   creating_training_data7=[ creating_training_data7;fid];
end
% [mean,sigma,err,priors]=em_gaussian(number_of_gaussians,creating_training_data3);
% model(3,1)={[mean]};
% model(3,2)={[sigma]};
% model(3,3)={[priors]};

no_of_training_data_class7=no_of_inmages_chosen-2;


for i = no_of_inmages_chosen+1:length(feature_directory)
     name = sprintf('/home/priyanka/Documents/MATLAB/ps3_ver3/GMM/opencountry/opencountry_features/%s',feature_directory(i).name);

    fid = load(name);
   %vec_fid=reshape( fid.' ,1,numel(fid));
   testing_data7=[testing_data7 ;fid];
   label7=[label7 ;3];
end

no_of_testing_data_class7=length(feature_directory)-(no_of_inmages_chosen);

%%% class 8
feature_directory=dir('/home/priyanka/Documents/MATLAB/ps3_ver3/GMM/tallbuilding/tallbuilding_features/*');
no_of_inmages_chosen=round(0.7*length(feature_directory));
creating_training_data8=[];
testing_data8=[];
label8=[];

for i = 3:no_of_inmages_chosen
name = sprintf('/home/priyanka/Documents/MATLAB/ps3_ver3/GMM/tallbuilding/tallbuilding_features/%s',feature_directory(i).name);
    fid = load(name);
  % vec_fid=reshape( fid.' ,1,numel(fid));
   creating_training_data8=[ creating_training_data8;fid];
end
% [mean,sigma,err,priors]=em_gaussian(number_of_gaussians,creating_training_data3);
% model(3,1)={[mean]};
% model(3,2)={[sigma]};
% model(3,3)={[priors]};

no_of_training_data_class8=no_of_inmages_chosen-2;


for i = no_of_inmages_chosen+1:length(feature_directory)
     name = sprintf('/home/priyanka/Documents/MATLAB/ps3_ver3/GMM/tallbuilding/tallbuilding_features/%s',feature_directory(i).name);

    fid = load(name);
   %vec_fid=reshape( fid.' ,1,numel(fid));
   testing_data8=[testing_data8 ;fid];
   label8=[label8 ;3];
end

no_of_testing_data_class8=length(feature_directory)-(no_of_inmages_chosen);


training_data=[creating_training_data1;creating_training_data2;creating_training_data3;creating_training_data4;creating_training_data5;creating_training_data6;creating_training_data7;creating_training_data8];
testing_data=[testing_data1;testing_data2;testing_data3;testing_data4;testing_data5;testing_data6;testing_data7;testing_data8];

norm_traning_data=norm_data_max_min(training_data);
training_data=norm_traning_data;

norm_testing_data=norm_data_max_min(testing_data);
testing_data=norm_testing_data;



%%% pca %%%%%%%%%%% 
%  250 dimension 

covar=cov(training_data);
[cov_length covar_width]=size(covar);
[u v]=eig(covar);
%u = pca(training_data);

reduced_data=training_data*u(:,cov_length-dim:cov_length);

 reduced_testing_data=testing_data*u(:,cov_length-dim:cov_length);

% %%%%%%%% getting  training and testing data for each class%%%%%%%%%%%

temp=1;

for i=1:36:no_of_training_data_class1*no_of_patches;
        temp_matrix=reduced_data(i:i+no_of_patches-1,:);
        training_class1(temp,:)=reshape( temp_matrix.' ,1,numel(temp_matrix));
        
        temp=temp+1;
   
end


temp=1;

count=no_of_training_data_class1*no_of_patches;
last=count+no_of_training_data_class2*no_of_patches;
for i=count+1:36:last;
        temp_matrix=reduced_data(i:i+no_of_patches-1,:);
        training_class2(temp,:)=reshape( temp_matrix.' ,1,numel(temp_matrix));
        
        temp=temp+1;
   
end




temp=1;

count=last;
last=count+no_of_training_data_class3*no_of_patches;
for i=count+1:36:last;
        temp_matrix=reduced_data(i:i+no_of_patches-1,:);
        training_class3(temp,:)=reshape( temp_matrix.' ,1,numel(temp_matrix));
        
        temp=temp+1;
end
%%% other classes
temp=1;

count=last;
last=count+no_of_training_data_class4*no_of_patches;
for i=count+1:36:last;
        temp_matrix=reduced_data(i:i+no_of_patches-1,:);
        training_class4(temp,:)=reshape( temp_matrix.' ,1,numel(temp_matrix));
        
        temp=temp+1;
end

temp=1;

count=last;
last=count+no_of_training_data_class5*no_of_patches;
for i=count+1:36:last
        temp_matrix=reduced_data(i:i+no_of_patches-1,:);
        training_class5(temp,:)=reshape( temp_matrix.' ,1,numel(temp_matrix));
        
        temp=temp+1;
end

temp=1;

count=last;
last=count+no_of_training_data_class6*no_of_patches;
for i=count+1:36:last;
        temp_matrix=reduced_data(i:i+no_of_patches-1,:);
        training_class6(temp,:)=reshape( temp_matrix.' ,1,numel(temp_matrix));
        
        temp=temp+1;
end

temp=1;

count=last;
last=count+no_of_training_data_class7*no_of_patches;
for i=count+1:36:last
        temp_matrix=reduced_data(i:i+no_of_patches-1,:);
        training_class7(temp,:)=reshape( temp_matrix.' ,1,numel(temp_matrix));
        
        temp=temp+1;
end
temp=1;

count=last;
last=count+no_of_training_data_class8*no_of_patches;
for i=count+1:36:last;
        temp_matrix=reduced_data(i:i+no_of_patches-1,:);
        training_class8(temp,:)=reshape( temp_matrix.' ,1,numel(temp_matrix));
        
        temp=temp+1;
end

temp=1;
%% testing
for i=1:36:no_of_testing_data_class1*no_of_patches;
        temp_matrix=reduced_testing_data(i:i+no_of_patches-1,:);
        testing_class1(temp,:)=reshape( temp_matrix.' ,1,numel(temp_matrix));
        
        temp=temp+1;
   
end


temp=1;

count=no_of_testing_data_class1*no_of_patches;
last=count+no_of_testing_data_class2*no_of_patches;
for i=count+1:36:last;
        temp_matrix=reduced_testing_data(i:i+no_of_patches-1,:);
        testing_class2(temp,:)=reshape( temp_matrix.' ,1,numel(temp_matrix));
        
        temp=temp+1;
   
end




temp=1;

count=last;
last=count+no_of_testing_data_class3*no_of_patches;
for i=count+1:36:last;
        temp_matrix=reduced_testing_data(i:i+no_of_patches-1,:);
        testing_class3(temp,:)=reshape( temp_matrix.' ,1,numel(temp_matrix));
        
        temp=temp+1;
   
end
%%% other classes
temp=1;

count=last;
last=count+no_of_testing_data_class4*no_of_patches;
for i=count+1:36:last;
        temp_matrix=reduced_testing_data(i:i+no_of_patches-1,:);
        testing_class4(temp,:)=reshape( temp_matrix.' ,1,numel(temp_matrix));
        
        temp=temp+1;
   
end

temp=1;

count=last;
last=count+no_of_testing_data_class5*no_of_patches;
for i=count+1:36:last;
        temp_matrix=reduced_testing_data(i:i+no_of_patches-1,:);
        testing_class5(temp,:)=reshape( temp_matrix.' ,1,numel(temp_matrix));
        
        temp=temp+1;
   
end

temp=1;

count=last;
last=count+no_of_testing_data_class6*no_of_patches;
for i=count+1:36:last;
        temp_matrix=reduced_testing_data(i:i+no_of_patches-1,:);
        testing_class6(temp,:)=reshape( temp_matrix.' ,1,numel(temp_matrix));
        
        temp=temp+1;
   
end

temp=1;

count=last;
last=count+no_of_testing_data_class7*no_of_patches;
for i=count+1:36:last;
        temp_matrix=reduced_testing_data(i:i+no_of_patches-1,:);
        testing_class7(temp,:)=reshape( temp_matrix.' ,1,numel(temp_matrix));
        
        temp=temp+1;
   
end

temp=1;

count=last;
last=count+no_of_testing_data_class8*no_of_patches;
for i=count+1:36:last;
        temp_matrix=reduced_testing_data(i:i+no_of_patches-1,:);
        testing_class8(temp,:)=reshape( temp_matrix.' ,1,numel(temp_matrix));
        
        temp=temp+1;
   
end

train_data = [training_class1;training_class2;training_class3;training_class4;training_class5;training_class6;training_class7;training_class8];
test_data = [testing_class1;testing_class2;testing_class3;testing_class4;testing_class5;testing_class6;testing_class7;testing_class8];
test_labels = [ones(size(testing_class1,1),1);(2*ones(size(testing_class2,1),1));(3*ones(size(testing_class3,1),1));(4*ones(size(testing_class4,1),1));(5*ones(size(testing_class5,1),1));(6*ones(size(testing_class6,1),1));(7*ones(size(testing_class7,1),1));(8*ones(size(testing_class8,1),1))];

fid = fopen('Image_PCASVMNoSV_Train.txt','w+');
  for i = 1 : size(reduced_data,1)
     
      fprintf(fid,'%d ', train_labels(i));    
     
      for j = 1: size(reduced_data,2)
     
          fprintf(fid,'%d:%f ', j,reduced_data(i,j));
          %fprintf(fid,'%f ', j,training_data(i,j));
      end
      
      fprintf(fid,'\n');
     
  end
  fclose(fid);
 
   fid = fopen('Image_PCASVMSVNoSV_Test.txt','w+');
     
  for i = 1 : size(reduced_testing_data,1)
     
      fprintf(fid,'%d ', test_labels(i));    
     
      for j = 1: size(reduced_testing_data,2)
     
          fprintf(fid,'%d:%f ', j,reduced_testing_data(i,j));
          
      end
      
      fprintf(fid,'\n');
     
  end
  fclose(fid);


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
save './ImageData/PCA_Labels_g7.mat' labels
save './ImageData/PCA_Probability_g7.mat' prob_all


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
save './ImageData/PCA_TPScores_g7.mat' true_postive_score
save './ImageData/PCA_FPScores_g7.mat' false_postive_score
