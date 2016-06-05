clc
close all
clear all

ref1=load('/home/priyanka/Documents/MATLAB/PR4/Music Data/Bhairavi_Ref/Bhairavi_44100.vp1.wav.cent.spline1');
ref2=load('/home/priyanka/Documents/MATLAB/PR4/Music Data/Bhairavi_Ref/Bhairavi_44100.vp2.wav.cent.spline1');
ref3=load('/home/priyanka/Documents/MATLAB/PR4/Music Data/Bhairavi_Ref/Bhairavi_44100.vp3.wav.cent.spline1');


ref1_cell=cell(1,1);
ref1_cell{1}=ref1;


ref2_cell=cell(1,1);
ref2_cell{1}=ref2;

ref3_cell=cell(1,1);
ref3_cell{1}=ref3;

text_read=fopen('test_data.txt','r');
test_data=textscan(text_read,'%s');

labels_read=fopen('labels.txt','r');
labels_data=textscan(labels_read,'%s');
labels=[];
classified_labes=zeros(104,1);
for i=1:16
    fid=labels_data{1, 1}{i, 1} ;
    temp=load(fid);
    labels=[labels;temp.labels];
end
test_cell=cell(106,1);

for i=1:104
    fid=test_data{1, 1}{i, 1} ;
    test_data_temp=load(fid);
    test_cell{i}=test_data_temp.test_data;
    
end

score_stack=zeros(104,3);
for i=1:104
[Dist,D,k,w,rw,tw]=dtw_symbols(ref1,test_cell{i});
score_stack(i,1)=Dist;
end


for i=1:104
[Dist,D,k,w,rw,tw]=dtw_symbols(ref2,test_cell{i});
score_stack(i,2)=Dist;
end


for i=1:104
[Dist,D,k,w,rw,tw]=dtw_symbols(ref3,test_cell{i});
score_stack(i,3)=Dist;
end

score_sum=sum(score_stack,2);

prob_values=zeros(104,3);

for i=1:104
    prob_values(i,1:3)=score_stack(i,1:3)/score_sum(i);
end



for i=1:104
    [temp classified_labels(i)]=min(prob_values(i,:));
end


confusion_matrix=confusionmat(labels,classified_labels)
num_of_clsses = 3;
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
