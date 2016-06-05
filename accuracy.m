clc
clear all
close all
x= [ 3 4 5 6 7 ]
y=[98 94.583 90.4167 97.0833 96.25];
plot(x,y,'LineWidth',2);
xlabel('Number of gaussians');
ylabel('Accuracy');
axis([3 7 0 100]);
screen2jpeg('DTW_accuracy.png');