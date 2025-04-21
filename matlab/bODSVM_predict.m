function [pred, accu] = bODSVM_predict(Y, X, model)

P = model.P;
[pred, accu, ~] = liblinearpredict(Y, sparse(X*P), model.svm, '-q');

end
