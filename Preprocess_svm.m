   
  
% char_a=load('/home/priyanka/Documents/MATLAB/ps3_ver3/OldFeatures/a.mat');
% char_ai=load('/home/priyanka/Documents/MATLAB/ps3_ver3/OldFeatures/ai.mat');
% char_bA=load('/home/priyanka/Documents/MATLAB/ps3_ver3/OldFeatures/bA.mat');
% char_chA=load('/home/priyanka/Documents/MATLAB/ps3_ver3/OldFeatures/chA.mat');
% char_dA=load('/home/priyanka/Documents/MATLAB/ps3_ver3/OldFeatures/dA.mat');
% 
% char_LA=load('/home/priyanka/Documents/MATLAB/ps3_ver3/OldFeatures/LA.mat');
% char_tA=load('/home/priyanka/Documents/MATLAB/ps3_ver3/OldFeatures/tA.mat');
% 
% char_la=load('/home/priyanka/Documents/MATLAB/ps3_ver3/OldFeatures/la.mat');


char_a=load('/home/priyanka/Documents/MATLAB/ps3_ver3/NewFeatures/a.mat');
char_ai=load('/home/priyanka/Documents/MATLAB/ps3_ver3/NewFeatures/ai.mat');
char_bA=load('/home/priyanka/Documents/MATLAB/ps3_ver3/NewFeatures/bA.mat');
char_chA=load('/home/priyanka/Documents/MATLAB/ps3_ver3/NewFeatures/chA.mat');
char_dA=load('/home/priyanka/Documents/MATLAB/ps3_ver3/NewFeatures/dA.mat');

char_LA=load('/home/priyanka/Documents/MATLAB/ps3_ver3/NewFeatures/LA.mat');
char_tA=load('/home/priyanka/Documents/MATLAB/ps3_ver3/NewFeatures/tA.mat');

char_la=load('/home/priyanka/Documents/MATLAB/ps3_ver3/NewFeatures/lA.mat');

% norm_data_a=norm_data_max_min(char_a.creating_data);
% no_of_data_in_class1=length(norm_data_a);
% 
% norm_data_ai=norm_data_max_min(char_ai.creating_data1);
% no_of_data_in_class2=length(norm_data_ai);
% 
% norm_data_bA=norm_data_max_min(char_bA.creating_data2);
% no_of_data_in_class3=length(norm_data_bA);
% 
% norm_data_chA=norm_data_max_min(char_chA.creating_data3);
% no_of_data_in_class4=length(norm_data_chA);
% 
% norm_data_dA=norm_data_max_min(char_dA.creating_data4);
% no_of_data_in_class5=length(norm_data_dA);
% 
% 
% norm_data_LA=norm_data_max_min(char_LA.creating_data6);
% no_of_data_in_class6=length(norm_data_LA);
% 
% norm_data_tA=norm_data_max_min(char_tA.creating_data7);
% no_of_data_in_class7=length(norm_data_tA);
% 
% 
% norm_data_la=norm_data_max_min(char_la.creating_data5);
% no_of_data_in_class8=length(norm_data_la);

% norm_data_a=(char_a.creating_data);
% no_of_data_in_class1=length(norm_data_a);
% 
% norm_data_ai=(char_ai.creating_data1);
% no_of_data_in_class2=length(norm_data_ai);
% 
% norm_data_bA=(char_bA.creating_data2);
% no_of_data_in_class3=length(norm_data_bA);
% 
% norm_data_chA=(char_chA.creating_data3);
% no_of_data_in_class4=length(norm_data_chA);
% 
% norm_data_dA=(char_dA.creating_data4);
% no_of_data_in_class5=length(norm_data_dA);
% 
% 
% norm_data_LA=(char_LA.creating_data6);
% no_of_data_in_class6=length(norm_data_LA);
% 
% norm_data_tA=(char_tA.creating_data7);
%  no_of_data_in_class7=length(norm_data_tA);
% 
% 
% norm_data_la=(char_la.creating_data5);
% no_of_data_in_class8=length(norm_data_la);

%%% NEW Features
norm_data_a=(char_a.creating_data);
no_of_data_in_class1=length(norm_data_a);

norm_data_ai=(char_ai.creating_data);
no_of_data_in_class2=length(norm_data_ai);

norm_data_bA=(char_bA.creating_data);
no_of_data_in_class3=length(norm_data_bA);

norm_data_chA=(char_chA.creating_data);
no_of_data_in_class4=length(norm_data_chA);

norm_data_dA=(char_dA.creating_data);
no_of_data_in_class5=length(norm_data_dA);


norm_data_LA=(char_LA.creating_data);
no_of_data_in_class6=length(norm_data_LA);

norm_data_tA=(char_tA.creating_data);
no_of_data_in_class7=length(norm_data_tA);



norm_data_la=(char_la.creating_data);
no_of_data_in_class8=length(norm_data_la);


class1_training_data=norm_data_a(1:round(0.7*no_of_data_in_class1),:);
class1_testing_data=norm_data_a(round(0.7*no_of_data_in_class1)+1:no_of_data_in_class1,:);
num_of_one_labeles=no_of_data_in_class1-round(0.7*no_of_data_in_class1);
labels_class1(1:num_of_one_labeles,1)=ones(1:num_of_one_labeles,1);
labels_train_class1(1:(no_of_data_in_class1-num_of_one_labeles),1)=ones(1:(no_of_data_in_class1-num_of_one_labeles),1);




class2_training_data=norm_data_ai(1:round(0.7*no_of_data_in_class2),:);
class2_testing_data=norm_data_ai(round(0.7*no_of_data_in_class2)+1:no_of_data_in_class2,:);
num_of_one_labeles=no_of_data_in_class2-round(0.7*no_of_data_in_class2);
labels_class2(1:num_of_one_labeles,1)=2*ones(1:num_of_one_labeles,1);
labels_train_class2(1:(no_of_data_in_class2-num_of_one_labeles),1)=2*ones(1:(no_of_data_in_class2-num_of_one_labeles),1);

class3_training_data=norm_data_bA(1:round(0.7*no_of_data_in_class3),:);
class3_testing_data=norm_data_bA(round(0.7*no_of_data_in_class3)+1:no_of_data_in_class3,:);
num_of_one_labeles=no_of_data_in_class3-round(0.7*no_of_data_in_class3);
labels_class3(1:num_of_one_labeles,1)=3*ones(1:num_of_one_labeles,1);
labels_train_class3(1:(no_of_data_in_class3-num_of_one_labeles),1)=3*ones(1:(no_of_data_in_class3-num_of_one_labeles),1);

class4_training_data=norm_data_chA(1:round(0.7*no_of_data_in_class4),:);
class4_testing_data=norm_data_chA(round(0.7*no_of_data_in_class4)+1:no_of_data_in_class4,:);
num_of_one_labeles=no_of_data_in_class4-round(0.7*no_of_data_in_class4);
labels_class4(1:num_of_one_labeles,1)=4*ones(1:num_of_one_labeles,1); 
labels_train_class4(1:(no_of_data_in_class4-num_of_one_labeles),1)=4*ones(1:(no_of_data_in_class4-num_of_one_labeles),1);

class5_training_data=norm_data_dA(1:round(0.7*no_of_data_in_class5),:);
class5_testing_data=norm_data_dA(round(0.7*no_of_data_in_class5)+1:no_of_data_in_class5,:);
num_of_one_labeles=no_of_data_in_class5-round(0.7*no_of_data_in_class5);
labels_class5(1:num_of_one_labeles,1)=5*ones(1:num_of_one_labeles,1);
labels_train_class5(1:(no_of_data_in_class5-num_of_one_labeles),1)=5*ones(1:(no_of_data_in_class5-num_of_one_labeles),1);

class6_training_data=norm_data_LA(1:round(0.7*no_of_data_in_class6),:);
class6_testing_data=norm_data_LA(round(0.7*no_of_data_in_class6)+1:no_of_data_in_class6,:);
num_of_one_labeles=no_of_data_in_class6-round(0.7*no_of_data_in_class6);
labels_class6(1:num_of_one_labeles,1)=6*ones(1:num_of_one_labeles,1); 
labels_train_class6(1:(no_of_data_in_class6-num_of_one_labeles),1)=6*ones(1:(no_of_data_in_class6-num_of_one_labeles),1);


class7_training_data=norm_data_tA(1:round(0.7*no_of_data_in_class7),:);
class7_testing_data=norm_data_tA(round(0.7*no_of_data_in_class7)+1:no_of_data_in_class7,:);
num_of_one_labeles=no_of_data_in_class7-round(0.7*no_of_data_in_class7);
labels_class7(1:num_of_one_labeles,1)=7*ones(1:num_of_one_labeles,1);
labels_train_class7(1:(no_of_data_in_class7-num_of_one_labeles),1)=7*ones(1:(no_of_data_in_class7-num_of_one_labeles),1);


class8_training_data=norm_data_la(1:round(0.7*no_of_data_in_class8),:);
class8_testing_data=norm_data_la(round(0.7*no_of_data_in_class8)+1:no_of_data_in_class8,:);
num_of_one_labeles=no_of_data_in_class8-round(0.7*no_of_data_in_class8);
labels_class8(1:num_of_one_labeles,1)=8*ones(1:num_of_one_labeles,1);
labels_train_class8(1:(no_of_data_in_class8-num_of_one_labeles),1)=8*ones(1:(no_of_data_in_class8-num_of_one_labeles),1);

Train = [class1_training_data ; class2_training_data ; class3_training_data;  class4_training_data ; class5_training_data ; class6_training_data ; class7_training_data;class8_training_data];
train_labels = [labels_train_class1;labels_train_class2;labels_train_class3;labels_train_class4;labels_train_class5;labels_train_class6;labels_train_class7; labels_train_class8];
test_labels = [labels_class1;labels_class2;labels_class3;labels_class4;labels_class5;labels_class6;labels_class7;labels_class8];
Test = [class1_testing_data ; class2_testing_data ; class3_testing_data; class4_testing_data ; class5_testing_data; class6_testing_data; class7_testing_data;class8_testing_data];
% Norm_Train = norm_data(Train);
% Norm_Test = norm_data(Test);
% Norm_Train = norm_data_max_min(Train);
% Norm_Test = norm_data_max_min(Test);


fid = fopen('HW_SVM_Train_NewFE.txt','w+');
  for i = 1 : size(Train,1)
     
          
     
      for j = 1: size(Train,2)
     
          %fprintf(fid,'%d:%f ', j,Norm_Train(i,j));
          fprintf(fid,'%f ', j,Train(i,j));
      end
      fprintf(fid,'%d ', train_labels(i));
      fprintf(fid,'\n');
     
  end
  fclose(fid);
 
   fid = fopen('HW_SVM_Test_NewFE.txt','w+');
     
  for i = 1 : size(Test,1)
     
          
     
      for j = 1: size(Test,2)
     
         % fprintf(fid,'%d:%f ', j,Norm_Test(i,j));
          fprintf(fid,'%f ', j,Test(i,j));
      end
      fprintf(fid,'%d ', test_labels(i));
      fprintf(fid,'\n');
     
  end
  fclose(fid);
 
