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
train_labels = [ones(size(training_class1,1),1);(2*ones(size(training_class2,1),1));(3*ones(size(training_class3,1),1));(4*ones(size(training_class4,1),1));(5*ones(size(training_class5,1),1));(6*ones(size(training_class6,1),1));(7*ones(size(training_class7,1),1));(8*ones(size(training_class8,1),1))];
test_labels = [ones(size(testing_class1,1),1);(2*ones(size(testing_class2,1),1));(3*ones(size(testing_class3,1),1));(4*ones(size(testing_class4,1),1));(5*ones(size(testing_class5,1),1));(6*ones(size(testing_class6,1),1));(7*ones(size(testing_class7,1),1));(8*ones(size(testing_class8,1),1))];
fid = fopen('Image_PCASVMNoSV_Train.txt','w+');
  for i = 1 : size(train_data,1)
     
      fprintf(fid,'%d ', train_labels(i));    
     
      for j = 1: size(train_data,2)
     
          fprintf(fid,'%d:%f ', j,train_data(i,j));
          %fprintf(fid,'%f ', j,training_data(i,j));
      end
      
      fprintf(fid,'\n');
     
  end
  fclose(fid);
 
   fid = fopen('Image_PCASVMSVNoSV_Test.txt','w+');
     
  for i = 1 : size(test_data,1)
     
      fprintf(fid,'%d ', test_labels(i));    
     
      for j = 1: size(test_data,2)
     
          fprintf(fid,'%d:%f ', j,test_data(i,j));
          
      end
      
      fprintf(fid,'\n');
     
  end
  fclose(fid);


