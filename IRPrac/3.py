from sklearn.metrics import precision_score, recall_score, f1_score

ground_check = [1,0,1,0,1,0,1,0,1,0]

prediction_relevance = [1,0,0,1,1,0,1,1,0,0]

PrecisionScore = precision_score(ground_check,prediction_relevance)
print(PrecisionScore)

Rs = recall_score(ground_check,prediction_relevance)
print(Rs)

Fs = f1_score(ground_check,prediction_relevance)
print(Fs)
