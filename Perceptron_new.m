clc
close all
clear all
number_of_gaussians=4;
num_of_clsses=8;
no_of_patches=36;
model=cell(num_of_clsses,3);
dim=4;


feature_directory=dir('/home/anju/Desktop/PR4/ImageDataSet/forest/*');
no_of_inmages_chosen=round(0.7*length(feature_directory));
creating_training_data1=[];
train_label1=[];
testing_data1=[];
label1=[];
no_of_features_per_image=36;
for i = 3:no_of_inmages_chosen
   name = sprintf('/home/anju/Desktop/PR4/ImageDataSet/forest/%s',feature_directory(i).name);
    fid = load(name);
    %vec_fid=reshape( fid.' ,1,numel(fid));
   creating_training_data1=[ creating_training_data1 ; fid];
   
end

no_of_training_data_class1=no_of_inmages_chosen-2;


for i = no_of_inmages_chosen+1:length(feature_directory)
   name = sprintf('/home/anju/Desktop/PR4/ImageDataSet/forest/%s',feature_directory(i).name);
    fid = load(name);
  % vec_fid=reshape( fid.' ,1,numel(fid)); 
   testing_data1=[testing_data1 ;fid];
   label1=[label1 ;1];
end
no_of_testing_data_class1=length(feature_directory)-(no_of_inmages_chosen);


feature_directory=dir('/home/anju/Desktop/PR4/ImageDataSet/coast/*');
no_of_inmages_chosen=round(0.7*length(feature_directory));

creating_training_data2=[];
testing_data2=[];
label2=[];

for i = 3:no_of_inmages_chosen
   name = sprintf('/home/anju/Desktop/PR4/ImageDataSet/coast/%s',feature_directory(i).name);
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
   name = sprintf('/home/anju/Desktop/PR4/ImageDataSet/coast/%s',feature_directory(i).name);
    fid = load(name);
  % vec_fid=reshape( fid.' ,1,numel(fid));
   testing_data2=[testing_data2 ;fid];
   label2=[label2;2];
end

no_of_testing_data_class2=length(feature_directory)-(no_of_inmages_chosen);

% class 3
feature_directory=dir('/home/anju/Desktop/PR4/ImageDataSet/street/*');
no_of_inmages_chosen=round(0.7*length(feature_directory));
creating_training_data3=[];
testing_data3=[];
label3=[];

for i = 3:no_of_inmages_chosen
  name = sprintf('/home/anju/Desktop/PR4/ImageDataSet/street/%s',feature_directory(i).name);
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
  name = sprintf('/home/anju/Desktop/PR4/ImageDataSet/street/%s',feature_directory(i).name);
    fid = load(name);
   %vec_fid=reshape( fid.' ,1,numel(fid));
   testing_data3=[testing_data3 ;fid];
   label3=[label3 ;3];
end

no_of_testing_data_class3=length(feature_directory)-(no_of_inmages_chosen);

%%%% other classes
feature_directory=dir('/home/anju/Desktop/PR4/ImageDataSet/highway/*');
no_of_inmages_chosen=round(0.7*length(feature_directory));
creating_training_data4=[];
testing_data4=[];
label4=[];

for i = 3:no_of_inmages_chosen
 name = sprintf('/home/anju/Desktop/PR4/ImageDataSet/highway/%s',feature_directory(i).name);
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
   name = sprintf('/home/anju/Desktop/PR4/ImageDataSet/highway/%s',feature_directory(i).name);
    fid = load(name);
   %vec_fid=reshape( fid.' ,1,numel(fid));
   testing_data4=[testing_data4 ;fid];
   label4=[label4 ;3];
end

no_of_testing_data_class4=length(feature_directory)-(no_of_inmages_chosen);
%%% class 5

feature_directory=dir('/home/anju/Desktop/PR4/ImageDataSet/insidecity/*');
no_of_inmages_chosen=round(0.7*length(feature_directory));
creating_training_data5=[];
testing_data5=[];
label5=[];

for i = 3:no_of_inmages_chosen
  name = sprintf('/home/anju/Desktop/PR4/ImageDataSet/insidecity/%s',feature_directory(i).name);
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
    name = sprintf('/home/anju/Desktop/PR4/ImageDataSet/insidecity/%s',feature_directory(i).name);

    fid = load(name);
   %vec_fid=reshape( fid.' ,1,numel(fid));
   testing_data5=[testing_data5 ;fid];
   label5=[label5 ;3];
end

no_of_testing_data_class5=length(feature_directory)-(no_of_inmages_chosen);

%%%% class 6

feature_directory=dir('/home/anju/Desktop/PR4/ImageDataSet/mountain/*');
no_of_inmages_chosen=round(0.7*length(feature_directory));
creating_training_data6=[];
testing_data6=[];
label6=[];

for i = 3:no_of_inmages_chosen
 name = sprintf('/home/anju/Desktop/PR4/ImageDataSet/mountain/%s',feature_directory(i).name);
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
     name = sprintf('/home/anju/Desktop/PR4/ImageDataSet/mountain/%s',feature_directory(i).name);

    fid = load(name);
   %vec_fid=reshape( fid.' ,1,numel(fid));
   testing_data6=[testing_data6 ;fid];
   label6=[label6 ;3];
end

no_of_testing_data_class6=length(feature_directory)-(no_of_inmages_chosen);
%%% class 7 

feature_directory=dir('/home/anju/Desktop/PR4/ImageDataSet/opencountry/*');
no_of_inmages_chosen=round(0.7*length(feature_directory));
creating_training_data7=[];
testing_data7=[];
label7=[];

for i = 3:no_of_inmages_chosen
 name = sprintf('/home/anju/Desktop/PR4/ImageDataSet/opencountry/%s',feature_directory(i).name);
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
     name = sprintf('/home/anju/Desktop/PR4/ImageDataSet/opencountry/%s',feature_directory(i).name);

    fid = load(name);
   %vec_fid=reshape( fid.' ,1,numel(fid));
   testing_data7=[testing_data7 ;fid];
   label7=[label7 ;3];
end

no_of_testing_data_class7=length(feature_directory)-(no_of_inmages_chosen);

%%% class 8
feature_directory=dir('/home/anju/Desktop/PR4/ImageDataSet/tallbuilding/*');
no_of_inmages_chosen=round(0.7*length(feature_directory));
creating_training_data8=[];
testing_data8=[];
label8=[];

for i = 3:no_of_inmages_chosen
name = sprintf('/home/anju/Desktop/PR4/ImageDataSet/tallbuilding/%s',feature_directory(i).name);
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
     name = sprintf('/home/anju/Desktop/PR4/ImageDataSet/tallbuilding/%s',feature_directory(i).name);

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

train_labels = [ones(no_of_training_data_class1,1); ones(no_of_training_data_class2,1).*2; ones(no_of_training_data_class3,1).*3 ;ones(no_of_training_data_class4,1).*4; ones(no_of_training_data_class5,1).*5; ones(no_of_training_data_class6,1).*6; ones(no_of_training_data_class7,1).*7; ones(no_of_training_data_class8,1).*8];
train_labels_for_1VR = [ones(no_of_training_data_class1,1);zeros(no_of_training_data_class2,1);zeros(no_of_training_data_class3,1);zeros(no_of_training_data_class4,1);zeros(no_of_training_data_class5,1);zeros(no_of_training_data_class6,1);zeros(no_of_training_data_class7,1);zeros(no_of_training_data_class8,1)];

test_label = [ones(no_of_testing_data_class1,1);zeros(no_of_testing_data_class2,1);zeros(no_of_testing_data_class3,1);zeros(no_of_testing_data_class4,1);zeros(no_of_testing_data_class5,1);zeros(no_of_testing_data_class6,1);zeros(no_of_testing_data_class7,1);zeros(no_of_testing_data_class8,1)];
rows = size(train_data,1);
columns = size(train_data,2);

Train_Data = zeros(rows,columns+1);

Train_Data(:,1) = ones(rows,1);
Train_Data(:,2:columns+1) = train_data;
% Train_Data(:,columns+2) = train_label;

rows = size(test_data,1);
columns = size(test_data,2);
Test_Data = zeros(rows,columns+1);
Test_Data(:,1) = ones(rows,1);
Test_Data(:,2:columns+1) = test_data;

W = zeros(1,columns+1);   % Weight vector
W_old = zeros(1,columns+1);

alpha = 0.05          % Learning parameter

error_threshold = 0.01
Converged = 0;
Y = zeros(size(train_data,1),1);
rows = size(Train_Data,1);
steps=0;
% for steps=10:10:60
%     steps
while steps~=100
%     steps
    temp_1=repmat(W,size(Train_Data,1),1).*Train_Data;
    Y= sum(temp_1,2);
    Predicted_labels = [];
for i=1:length(Y)
    if Y(i,1)<0
        Predicted_labels=[Predicted_labels; 0];
    else
        Predicted_labels=[Predicted_labels;1];
    end
end
%     Y
    diff_lab = (train_labels_for_1VR - Predicted_labels).*alpha;
    W_delta = sum(repmat(diff_lab,1,size(Train_Data,2)).*Train_Data,1);
    W = W + W_delta;
   
%     W
    %for j=1:rows
     %   Y(j,1) = W * (Train_Data(j,:)');
      %  diff_lab = alpha*(train_label -Y
       % for i=1:columns+1
        %    t = W(1,i) + ((alpha*(train_label(j,1)-Y(j,1)))*Train_Data(j,i));
         %   W(1,i) = t;
        %end
    %end
%     error = train_label - Y;
%     error = sum(error);
%     error = error/rows;
%     if error<=error_threshold
%         Converged=1;
%     else
%         Converged=0;
%     end
    steps= steps+1;
%     sum(W)
end
 
%%%%%%%%%%%%%%%%% Testing %%%%%%%%%%%%%%%

temp_1=repmat(W,size(Test_Data,1),1).*Test_Data;
Y_test= sum(temp_1,2);

%%%%%%%%%%%%%% accuracy %%%%%%%%%%%%%%%%%%

%Confusion matrix and other metrics

Predicted_labels = [];
for i=1:length(Y_test)
    if Y_test(i,1)<0
        Predicted_labels=[Predicted_labels; 0];
    else
        Predicted_labels=[Predicted_labels;1];
    end
end
% sum(Predicted_labels~=test_label)
conmat = confusionmat(test_label,Predicted_labels,'order',[0,1]);

precision=zeros(2,1);
recall=zeros(2,1);
f1score=zeros(2,1);
accuracy=0;
for i=1:2
   precision(i,1)=conmat(i,i)/sum(conmat(:,i));
   recall(i,1)=conmat(i,i)/sum(conmat(i,:));
   f1score(i,1)=(2*precision(i,1)*recall(i,1))/(precision(i,1)+recall(i,1));
   accuracy=accuracy+conmat(i,i);
   
end
accuracy=(accuracy/size(test_label,1))*100

conmat
index=[];
for i=1:1:99
    index=[index;i];
end
indexF=[];
for i=100:1:812
    indexF=[indexF;i];
end
TrueScore_forest= findTrueScore(index,Y_test,Predicted_labels,1);
FalseScore_forest=findFalseScore(indexF,Y_test,Predicted_labels,1);
str = sprintf('/home/anju/Desktop/PR4/Perceptron_TrueScore.txt');
fid = fopen(str,'wt');
for i=1:length(TrueScore_forest)
    fprintf(fid,'%g ',TrueScore_forest(i));
    fprintf(fid,'\n');
end
fclose(fid);

str = sprintf('/home/anju/Desktop/PR4/Perceptron_FalseScore.txt');
fid = fopen(str,'wt');
for i=1:length(FalseScore_forest)
    fprintf(fid,'%g ',FalseScore_forest(i));
    fprintf(fid,'\n');
end
fclose(fid);

str = sprintf('/home/anju/Desktop/PR4/PredictedLabels.txt');
fid = fopen(str,'wt');
for i=1:length(Predicted_labels)
    fprintf(fid,'%g ',Predicted_labels(i));
    fprintf(fid,'\n');
end
fclose(fid);