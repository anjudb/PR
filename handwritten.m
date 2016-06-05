clc
clear all
close all



number_of_gaussians=1;
num_of_clsses=8;
model=cell(num_of_clsses,3);


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
% norm_data_ai=norm_data_max_min(char_ai.creating_data);
% no_of_data_in_class2=length(norm_data_ai);
% 
% norm_data_bA=norm_data_max_min(char_bA.creating_data);
% no_of_data_in_class3=length(norm_data_bA);
% 
% norm_data_chA=norm_data_max_min(char_chA.creating_data);
% no_of_data_in_class4=length(norm_data_chA);
% 
% norm_data_dA=norm_data_max_min(char_dA.creating_data);
% no_of_data_in_class5=length(norm_data_dA);
% 
% 
% norm_data_LA=norm_data_max_min(char_LA.creating_data);
% no_of_data_in_class6=length(norm_data_LA);
% 
% norm_data_tA=norm_data_max_min(char_tA.creating_data);
% no_of_data_in_class7=length(norm_data_tA);
% 
% norm_data_la=norm_data_max_min(char_la.creating_data);
% no_of_data_in_class8=length(norm_data_la);

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



class2_training_data=norm_data_ai(1:round(0.7*no_of_data_in_class2),:);
class2_testing_data=norm_data_ai(round(0.7*no_of_data_in_class2)+1:no_of_data_in_class2,:);
num_of_one_labeles=no_of_data_in_class2-round(0.7*no_of_data_in_class2);
labels_class2(1:num_of_one_labeles,1)=2*ones(1:num_of_one_labeles,1);


class3_training_data=norm_data_bA(1:round(0.7*no_of_data_in_class3),:);
class3_testing_data=norm_data_bA(round(0.7*no_of_data_in_class3)+1:no_of_data_in_class3,:);
num_of_one_labeles=no_of_data_in_class3-round(0.7*no_of_data_in_class3);
labels_class3(1:num_of_one_labeles,1)=3*ones(1:num_of_one_labeles,1);


class4_training_data=norm_data_chA(1:round(0.7*no_of_data_in_class4),:);
class4_testing_data=norm_data_chA(round(0.7*no_of_data_in_class4)+1:no_of_data_in_class4,:);
num_of_one_labeles=no_of_data_in_class4-round(0.7*no_of_data_in_class4);
labels_class4(1:num_of_one_labeles,1)=4*ones(1:num_of_one_labeles,1); 


class5_training_data=norm_data_dA(1:round(0.7*no_of_data_in_class5),:);
class5_testing_data=norm_data_dA(round(0.7*no_of_data_in_class5)+1:no_of_data_in_class5,:);
num_of_one_labeles=no_of_data_in_class5-round(0.7*no_of_data_in_class5);
labels_class5(1:num_of_one_labeles,1)=5*ones(1:num_of_one_labeles,1);


class6_training_data=norm_data_LA(1:round(0.7*no_of_data_in_class6),:);
class6_testing_data=norm_data_LA(round(0.7*no_of_data_in_class6)+1:no_of_data_in_class6,:);
num_of_one_labeles=no_of_data_in_class6-round(0.7*no_of_data_in_class6);
labels_class6(1:num_of_one_labeles,1)=6*ones(1:num_of_one_labeles,1); 



class7_training_data=norm_data_tA(1:round(0.7*no_of_data_in_class7),:);
class7_testing_data=norm_data_tA(round(0.7*no_of_data_in_class7)+1:no_of_data_in_class7,:);
num_of_one_labeles=no_of_data_in_class7-round(0.7*no_of_data_in_class7);
labels_class7(1:num_of_one_labeles,1)=7*ones(1:num_of_one_labeles,1);



class8_training_data=norm_data_la(1:round(0.7*no_of_data_in_class8),:);
class8_testing_data=norm_data_la(round(0.7*no_of_data_in_class8)+1:no_of_data_in_class8,:);
num_of_one_labeles=no_of_data_in_class8-round(0.7*no_of_data_in_class8);
labels_class8(1:num_of_one_labeles,1)=8*ones(1:num_of_one_labeles,1);
%%%%%%%%%%%GMMM training%%%%%%%%%%%%

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



%%%%%%%%testing%%%%%%%%%%%

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

for iter=1:class4_testing_length
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

for iter=1:class5_testing_length
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

for iter=1:class6_testing_length
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

for iter=1:class7_testing_length
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

for iter=1:class8_testing_length
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




%%%%%%%%%%%% confusion matrix%%%%%%%%%%%%%
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





confusion_matrix

%%%%%%%%%%roc%%%%%%%%%
prob_all=[proab_class1;proab_class2;proab_class3;proab_class4;proab_class5;proab_class6;proab_class7;proab_class8];
labels=[labels_class1;labels_class2;labels_class3;labels_class4;labels_class5;labels_class6;labels_class7;labels_class8];
classified_labels=[testing_labels_class1;testing_labels_class2;testing_labels_class3;testing_labels_class4;testing_labels_class5;testing_labels_class6;testing_labels_class7;testing_labels_class8];

save './HW_Probability_g4.mat' prob_all
save './HW_Labels_g4.mat' labels

[X,Y] = perfcurve(labels,prob_all(:,1),1);
plot(X,Y,'b','LineWidth',2);

hold on
[X,Y] = perfcurve(labels,prob_all(:,2),2);
plot(X,Y,'r','LineWidth',2);

hold on
[X,Y] = perfcurve(labels,prob_all(:,3),3);
plot(X,Y,'g','LineWidth',2);

hold on
[X,Y] = perfcurve(labels,prob_all(:,4),4);
plot(X,Y,'c','LineWidth',2);
hold on
[X,Y] = perfcurve(labels,prob_all(:,5),5);
plot(X,Y,'m','LineWidth',2);

hold on
[X,Y] = perfcurve(labels,prob_all(:,6),6);
plot(X,Y,'k');

hold on
[X,Y] = perfcurve(labels,prob_all(:,7),7);
plot(X,Y,'y','LineWidth',2);


hold on
[X,Y] = perfcurve(labels,prob_all(:,8),8);
plot(X,Y,'color',[1 0.83 0.23 ],'LineWidth',2); 



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

title('ROC for System')
ylabel('True Positive Rate')
xlabel('False Positive Rate')
legend('Class1','Class2','Class3','Class4','Class5','Class6','Class7','Class8','Location','SouthEast');
screen2jpeg('HW_roc.png');






figure


%%%%%%%% Det curves%%%%%%%%%%%%%%
%%%%%%%class1%%%%%%%%%%%%

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

save '~/HW_TPScores_g4.mat' true_postive_score
save '~/Hw_FPScores_g4.mat' false_postive_score

[Pmiss, Pfa] = Compute_DET(true_postive_score, false_postive_score);

 Plot_DET (Pmiss,Pfa,'b',2);

hold on
 %%%%%%%%%%%%%%%%%class2%%%%%%%%%%%%%%%%
 index_class2_tp=find(testing_labels_class2==2);
true_postive_score=proab_class2(index_class2_tp,2);

index_class2_fp_class1=find(testing_labels_class1==2);
false_postive_score_class1=proab_class1(index_class2_fp_class1,2);

index_class2_fp_class3=find(testing_labels_class3==2);
false_postive_score_class3=proab_class3(index_class2_fp_class3,2);


index_class2_fp_class4=find(testing_labels_class4==2);
false_postive_score_class4=proab_class4(index_class2_fp_class4,2);

index_class2_fp_class5=find(testing_labels_class5==2);
false_postive_score_class5=proab_class5(index_class2_fp_class5,2);


index_class2_fp_class6=find(testing_labels_class6==2);
false_postive_score_class6=proab_class6(index_class2_fp_class6,2);

index_class2_fp_class7=find(testing_labels_class7==2);
false_postive_score_class7=proab_class7(index_class2_fp_class7,2);

index_class2_fp_class8=find(testing_labels_class8==2);
false_postive_score_class8=proab_class8(index_class2_fp_class8,2);



false_postive_score=[false_postive_score_class1;false_postive_score_class3;false_postive_score_class4;false_postive_score_class5;false_postive_score_class6;false_postive_score_class7;false_postive_score_class8];

[Pmiss, Pfa] = Compute_DET(true_postive_score, false_postive_score);

 Plot_DET (Pmiss,Pfa,'r',2);

 
 
 
  %%%%%%%%%%%%%%%%%class3%%%%%%%%%%%%%%%%
  hold on
 
index_class3_tp=find(testing_labels_class3==3);
true_postive_score=proab_class3(index_class3_tp,3);

index_class3_fp_class1=find(testing_labels_class1==3);
false_postive_score_class1=proab_class1(index_class3_fp_class1,3);

index_class3_fp_class2=find(testing_labels_class2==3);
false_postive_score_class2=proab_class2(index_class3_fp_class2,3);


index_class3_fp_class4=find(testing_labels_class4==3);
false_postive_score_class4=proab_class4(index_class3_fp_class4,3);

index_class3_fp_class5=find(testing_labels_class5==3);
false_postive_score_class5=proab_class5(index_class3_fp_class5,3);


index_class3_fp_class6=find(testing_labels_class6==3);
false_postive_score_class6=proab_class6(index_class3_fp_class6,3);

index_class3_fp_class7=find(testing_labels_class7==3);
false_postive_score_class7=proab_class7(index_class3_fp_class7,3);


index_class3_fp_class8=find(testing_labels_class8==3);
false_postive_score_class8=proab_class8(index_class3_fp_class8,1);


false_postive_score=[false_postive_score_class1;false_postive_score_class2;false_postive_score_class4;false_postive_score_class5;false_postive_score_class6;false_postive_score_class7;false_postive_score_class8];

[Pmiss, Pfa] = Compute_DET(true_postive_score, false_postive_score);

 Plot_DET (Pmiss,Pfa,'g',2);



  %%%%%%%%%%%%%%%%%class4%%%%%%%%%%%%%%%%
  hold on
 
index_class4_tp=find(testing_labels_class4==4);
true_postive_score=proab_class4(index_class4_tp,4);

index_class4_fp_class1=find(testing_labels_class1==4);
false_postive_score_class1=proab_class1(index_class4_fp_class1,4);

index_class4_fp_class2=find(testing_labels_class2==4);
false_postive_score_class2=proab_class2(index_class4_fp_class2,4);


index_class4_fp_class3=find(testing_labels_class3==4);
false_postive_score_class3=proab_class3(index_class4_fp_class3,4);

index_class4_fp_class5=find(testing_labels_class5==4);
false_postive_score_class5=proab_class5(index_class4_fp_class5,4);


index_class4_fp_class6=find(testing_labels_class6==4);
false_postive_score_class6=proab_class6(index_class4_fp_class6,4);

index_class4_fp_class7=find(testing_labels_class7==4);
false_postive_score_class7=proab_class7(index_class4_fp_class7,4);


index_class4_fp_class8=find(testing_labels_class8==4);
false_postive_score_class8=proab_class8(index_class4_fp_class8,4);


false_postive_score=[false_postive_score_class1;false_postive_score_class2;false_postive_score_class3;false_postive_score_class5;false_postive_score_class6;false_postive_score_class7;false_postive_score_class8];

[Pmiss, Pfa] = Compute_DET(true_postive_score, false_postive_score);

 Plot_DET (Pmiss,Pfa,'m',2);





 %%%%%%%%%%%%%%%%%class5%%%%%%%%%%%%%%%%
  hold on
 
index_class5_tp=find(testing_labels_class5==5);
true_postive_score=proab_class5(index_class5_tp,4);

index_class5_fp_class1=find(testing_labels_class1==5);
false_postive_score_class1=proab_class1(index_class5_fp_class1,5);

index_class5_fp_class2=find(testing_labels_class2==5);
false_postive_score_class2=proab_class2(index_class5_fp_class2,5);


index_class5_fp_class3=find(testing_labels_class3==5);
false_postive_score_class3=proab_class3(index_class5_fp_class3,5);

index_class5_fp_class4=find(testing_labels_class4==5);
false_postive_score_class4=proab_class4(index_class5_fp_class4,5);


index_class5_fp_class6=find(testing_labels_class6==5);
false_postive_score_class6=proab_class6(index_class5_fp_class6,5);

index_class5_fp_class7=find(testing_labels_class7==5);
false_postive_score_class7=proab_class7(index_class5_fp_class7,5);

index_class5_fp_class8=find(testing_labels_class8==5);
false_postive_score_class8=proab_class8(index_class5_fp_class8,5);



false_postive_score=[false_postive_score_class1;false_postive_score_class2;false_postive_score_class3;false_postive_score_class4;false_postive_score_class6;false_postive_score_class7;false_postive_score_class8];

[Pmiss, Pfa] = Compute_DET(true_postive_score, false_postive_score);

 Plot_DET (Pmiss,Pfa,'y',2);






 %%%%%%%%%%%%%%%%%class6%%%%%%%%%%%%%%%%
  hold on
 
index_class6_tp=find(testing_labels_class6==6);
true_postive_score=proab_class6(index_class6_tp,6);

index_class6_fp_class1=find(testing_labels_class1==6);
false_postive_score_class1=proab_class1(index_class6_fp_class1,6);

index_class6_fp_class2=find(testing_labels_class2==6);
false_postive_score_class2=proab_class2(index_class6_fp_class2,6);


index_class6_fp_class3=find(testing_labels_class3==6);
false_postive_score_class3=proab_class3(index_class6_fp_class3,6);

index_class6_fp_class4=find(testing_labels_class4==6);
false_postive_score_class4=proab_class4(index_class6_fp_class4,6);


index_class6_fp_class5=find(testing_labels_class5==6);
false_postive_score_class5=proab_class5(index_class6_fp_class5,6);

index_class6_fp_class7=find(testing_labels_class7==6);
false_postive_score_class7=proab_class7(index_class6_fp_class7,6);


index_class6_fp_class8=find(testing_labels_class8==6);
false_postive_score_class8=proab_class8(index_class6_fp_class8,6);


false_postive_score=[false_postive_score_class1;false_postive_score_class2;false_postive_score_class3;false_postive_score_class4;false_postive_score_class5;false_postive_score_class7];

[Pmiss, Pfa] = Compute_DET(true_postive_score, false_postive_score);

 Plot_DET (Pmiss,Pfa,'k',2);


 %%%%%%%%%%%%%%%%%class7%%%%%%%%%%%%%%%%
  hold on
 
index_class7_tp=find(testing_labels_class7==7);
true_postive_score=proab_class7(index_class7_tp,6);

index_class7_fp_class1=find(testing_labels_class1==7);
false_postive_score_class1=proab_class1(index_class6_fp_class1,7);

index_class7_fp_class2=find(testing_labels_class2==7);
false_postive_score_class2=proab_class2(index_class6_fp_class2,7);


index_class7_fp_class3=find(testing_labels_class3==7);
false_postive_score_class3=proab_class3(index_class6_fp_class3,7);

index_class7_fp_class4=find(testing_labels_class4==7);
false_postive_score_class4=proab_class4(index_class6_fp_class4,7);


index_class7_fp_class5=find(testing_labels_class5==7);
false_postive_score_class5=proab_class5(index_class6_fp_class5,7);

index_class7_fp_class6=find(testing_labels_class6==7);
false_postive_score_class6=proab_class6(index_class6_fp_class7,7);


index_class7_fp_class8=find(testing_labels_class8==7);
false_postive_score_class8=proab_class8(index_class7_fp_class8,7);


false_postive_score=[false_postive_score_class1;false_postive_score_class2;false_postive_score_class3;false_postive_score_class4;false_postive_score_class5;false_postive_score_class6;false_postive_score_class8];

[Pmiss, Pfa] = Compute_DET(true_postive_score, false_postive_score);

 Plot_DET (Pmiss,Pfa,'y',2);

index_class8_tp=find(testing_labels_class8==8);
true_postive_score=proab_class8(index_class8_tp,8);

index_class8_fp_class1=find(testing_labels_class1==8);
false_postive_score_class1=proab_class1(index_class8_fp_class1,8);

index_class8_fp_class2=find(testing_labels_class2==8);
false_postive_score_class2=proab_class2(index_class8_fp_class2,8);


index_class8_fp_class3=find(testing_labels_class3==8);
false_postive_score_class3=proab_class3(index_class8_fp_class3,8);

index_class8_fp_class4=find(testing_labels_class4==8);
false_postive_score_class4=proab_class4(index_class8_fp_class4,8);


index_class8_fp_class5=find(testing_labels_class5==8);
false_postive_score_class5=proab_class5(index_class8_fp_class5,8);

index_class8_fp_class6=find(testing_labels_class6==8);
false_postive_score_class6=proab_class6(index_class8_fp_class6,8);


index_class8_fp_class7=find(testing_labels_class7==8);
false_postive_score_class7=proab_class8(index_class8_fp_class7,8);


false_postive_score=[false_postive_score_class1;false_postive_score_class2;false_postive_score_class3;false_postive_score_class4;false_postive_score_class5;false_postive_score_class6;false_postive_score_class7];

[Pmiss, Pfa] = Compute_DET(true_postive_score, false_postive_score);

 Plot_DET (Pmiss,Pfa,'c',2); 


title('DET for System')
ylabel('Miss Probability in (%)')
legend('Class1','Class2','Class3','Class4','Class5','Class6','Class7','Class8','Location','SouthEast');
screen2jpeg('HW_det.png');





























 

