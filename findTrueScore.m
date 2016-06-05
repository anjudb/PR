function true_score=findTrueScore(indices,Scores,predictedLabels,label)
    
    true_score=[];
    for i=1:length(indices)
        if(predictedLabels(indices(i))==label)
            true_score=[true_score; Scores(indices(i))];
        end
    end
end
