close all
clear all
clc


feature_directory=dir('./digit_data/eight/*');

no_of_data_in_class1=length(feature_directory)-2;

creating_training_data1=[];
no_feature_vector_class1=zeros(no_of_data_in_class1,1);
for i = 3:length(feature_directory)
   fid = load(feature_directory(i).name);
   [fid_lenght fid_width]=size(fid);
   no_feature_vector_class1(i-2,1)=fid_lenght;
  creating_training_data1=[ creating_training_data1 ; fid];
end
