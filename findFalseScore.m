function falseScore=findFalseScore(indices,Scores,predictedLabels,label)  
    falseScore=[];
    for i=1:length(indices)
        if(predictedLabels(indices(i))==label)
            falseScore=[falseScore; Scores(indices(i))];
        end
    end
end