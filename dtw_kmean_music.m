clc
close all
clear all

big_mat=[];

ref1=load('/home/priyanka/Documents/MATLAB/PR4/Music Data/Bhairavi_Ref/Bhairavi_44100.vp1.wav.cent.spline1');
big_mat=[big_mat;ref1];

ref2=load('/home/priyanka/Documents/MATLAB/PR4/Music Data/Bhairavi_Ref/Bhairavi_44100.vp2.wav.cent.spline1');
big_mat=[big_mat;ref2];

ref3=load('/home/priyanka/Documents/MATLAB/PR4/Music Data/Bhairavi_Ref/Bhairavi_44100.vp3.wav.cent.spline1');
big_mat=[big_mat;ref3];
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


for i=1:104
    fid=test_data{1, 1}{i, 1} ;
    test_data_temp=load(fid);
   big_mat=[big_mat;test_data_temp.test_data];
end
kmean_symbols=kmeans(big_mat,10,'MaxIter',500);



first=1;
last=length(ref1);

ref1_symbol=cell(1,1);
ref1_symbol=kmean_symbols(first:last,1);

first=last+1;
last=first+length(ref2)-1;
ref2_symbol=cell(1,1);
ref2_symbol=kmean_symbols(first:last,1);

first=last+1;
last=first+length(ref3)-1;
ref3_symbol=cell(1,1);
ref3_symbol=kmean_symbols(first:last,1);

test_data_symbol=cell(104,1);



for i=1:104
    first=last+1;
    fid=test_data{1, 1}{i, 1} ;
    test_data_temp=load(fid);
    last=first+length(test_data_temp.test_data)-1;
    test_data_symbol{i}=kmean_symbols(first:last,1);
     
end






%%%%%%%%%%%%%%%test data%%%%%%%%%%%%%





score_stack=zeros(104,3);
for i=1:104
[Dist,D,k,w,rw,tw]=dtw_symbols(ref1_symbol,test_data_symbol{i});
score_stack(i,1)=Dist;
end


for i=1:104
[Dist,D,k,w,rw,tw]=dtw_symbols(ref2_symbol,test_data_symbol{i});
score_stack(i,2)=Dist;
end


for i=1:104
[Dist,D,k,w,rw,tw]=dtw_symbols(ref3_symbol,test_data_symbol{i});
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









