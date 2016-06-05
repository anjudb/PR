clc
close all
clear all

labels=load('/home/priyanka/Documents/MATLAB/ps3_ver3/roc_cases_digit/DTW_Labels_g3.mat');
prob_scores=load('/home/priyanka/Documents/MATLAB/ps3_ver3/roc_cases_digit/DTW_Probability_g3.mat');
figure
[X,Y] = perfcurve(labels.labels,prob_scores.prob_all(:,1),1);
plot(X,Y,'r','LineWidth',2);

hold on
labels=load('/home/priyanka/Documents/MATLAB/ps3_ver3/roc_cases_digit/DTW_Labels_g4.mat');
prob_scores=load('/home/priyanka/Documents/MATLAB/ps3_ver3/roc_cases_digit/DTW_Probability_g4.mat');

[X,Y] = perfcurve(labels.labels,prob_scores.prob_all(:,1),1);
plot(X,Y,'g','LineWidth',2);
hold on

labels=load('/home/priyanka/Documents/MATLAB/ps3_ver3/roc_cases_digit/DTW_Labels_g5.mat');
prob_scores=load('/home/priyanka/Documents/MATLAB/ps3_ver3/roc_cases_digit/DTW_Probability_g5.mat');


[X,Y] = perfcurve(labels.labels,prob_scores.prob_all(:,1),1);
plot(X,Y,'b','LineWidth',2);

hold on
labels=load('/home/priyanka/Documents/MATLAB/ps3_ver3/roc_cases_digit/DTW_Labels_g6.mat');
prob_scores=load('/home/priyanka/Documents/MATLAB/ps3_ver3/roc_cases_digit/DTW_Probability_g6.mat');

[X,Y] = perfcurve(labels.labels,prob_scores.prob_all(:,1),1);
plot(X,Y,'m','LineWidth',2);

hold on
labels=load('/home/priyanka/Documents/MATLAB/ps3_ver3/roc_cases_digit/DTW_Labels_g7.mat');
prob_scores=load('/home/priyanka/Documents/MATLAB/ps3_ver3/roc_cases_digit/DTW_Probability_g7.mat');

[X,Y] = perfcurve(labels.labels,prob_scores.prob_all(:,1),1);
plot(X,Y,'k','LineWidth',2);

title('ROC for different #cluster mixtures')
ylabel('True Positive Rate')
xlabel('False Positive Rate')
legend('#cluster3','#cluster4','#cluster5','#cluster6','#cluster7','Location','SouthEast'); 
screen2jpeg('DTW_#clusters_k30.png');
figure,
TP = load('/home/priyanka/Documents/MATLAB/ps3_ver3/roc_cases_digit/DTW_TPScores_g3.mat');
FP = load('/home/priyanka/Documents/MATLAB/ps3_ver3/roc_cases_digit/DTW_FPScores_g3.mat');
[Pmiss, Pfa] = Compute_DET(TP.true_postive_score,FP.false_postive_score);
Plot_DET (Pmiss,Pfa,'r',2);
hold on

TP = load('/home/priyanka/Documents/MATLAB/ps3_ver3/roc_cases_digit/DTW_TPScores_g4.mat');
FP = load('/home/priyanka/Documents/MATLAB/ps3_ver3/roc_cases_digit/DTW_FPScores_g4.mat');
[Pmiss, Pfa] = Compute_DET(TP.true_postive_score,FP.false_postive_score);
Plot_DET (Pmiss,Pfa,'g',2);

hold on
TP = load('/home/priyanka/Documents/MATLAB/ps3_ver3/roc_cases_digit/DTW_TPScores_g5.mat');
FP = load('/home/priyanka/Documents/MATLAB/ps3_ver3/roc_cases_digit/DTW_FPScores_g5.mat');
[Pmiss, Pfa] = Compute_DET(TP.true_postive_score,FP.false_postive_score);
Plot_DET (Pmiss,Pfa,'b',2);
hold on

TP = load('/home/priyanka/Documents/MATLAB/ps3_ver3/roc_cases_digit/DTW_TPScores_g6.mat');
FP = load('/home/priyanka/Documents/MATLAB/ps3_ver3/roc_cases_digit/DTW_FPScores_g6.mat');
[Pmiss, Pfa] = Compute_DET(TP.true_postive_score,FP.false_postive_score);
Plot_DET (Pmiss,Pfa,'m',2);
hold on

TP = load('/home/priyanka/Documents/MATLAB/ps3_ver3/roc_cases_digit/DTW_TPScores_g7.mat');
FP = load('/home/priyanka/Documents/MATLAB/ps3_ver3/roc_cases_digit/DTW_FPScores_g7.mat');
[Pmiss, Pfa] = Compute_DET(TP.true_postive_score,FP.false_postive_score);
Plot_DET (Pmiss,Pfa,'k',2);

title('DET for different #cluster mixtures')
legend('#cluster3','#cluster4','#cluster5','#cluster6','#cluster7','Location','SouthEast'); 
screen2jpeg('DTW_#g_k30_det.png');