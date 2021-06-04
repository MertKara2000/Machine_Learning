dataset = readtable('pulmoner_data.csv');
radiomics = dataset(:,3:109);
subtip = dataset(:,2);


[n_of_rows,n_of_col] = size(subtip)

subtip = table2array(subtip);


subtip = categorical(subtip);

for x = 1:n_of_rows;
    type = subtip(x);
    if type == "PF"
        subtip(x) = "1"
    elseif type == "No PF"
        subtip(x) = "2"
    elseif type == "Control"
        subtip(x) = "3"
    end
end



X = table2array(radiomics);
tmp = string(subtip);
y = double(tmp);
ozellikler = radiomics.Properties.VariableNames;
%
warning('off')
hmf = 1;
result = [];



%Tablo burada oluşturuluyor.
for k=1:25
    k
    for j=1:7
        j
        %feature selection buraya yazılacak
        if j==1
            %chi2 
            [idx,scores] = fscchi2(X,y);
            find(isinf(scores));
        elseif j==2
            %mrmr
            [idx,scores] = fscmrmr(X,y);

        elseif j==3
            %PredıctorImportance classification tree
            Mdl = fitctree(X,y);
            imp = predictorImportance(Mdl);
            imp = imp.';
            [imp,idx] = sortrows(imp,1,'descend');

        elseif j==4
            %PredıctorImportance of Classification ensemble 
            t = templateTree('MaxNumSplits',1);
            ens = fitcensemble(X,y,'Method','AdaBoostM2','Learners',t);
            imp = predictorImportance(ens);
            imp = imp.';
            [imp,idx] = sortrows(imp,1,'descend');

        elseif j==5
            %PredıctorImportance of regression ensemble
            t = templateTree('MaxNumSplits',1);
            ens = fitrensemble(X,y,'Method','LSBoost','Learners',t);
            imp = predictorImportance(ens);
            imp = imp.';
            [imp,idx] = sortrows(imp,1,'descend');

        elseif j==6
            %relieff K=3
            [idx,weights] = relieff(X,y,3);

        elseif j==7   
            %relieff K=11    
            [idx,weights] = relieff(X,y,11);

        end

        bestAccuracy =0;

        for i =1:10
        
            hmf =i;

            TrainData = [];
            TrainData = table;
            %eğitim verisi hazırlanıyor.
            for i=1:hmf
                temp = ozellikler(idx(i));
                TrainData = [TrainData,radiomics(:,temp)];
            end

            trainingData = TrainData;
            responseData = subtip;

            if k==1
                [trainedClassifier, validationAccuracy] = FineTree(trainingData, responseData, idx, ozellikler ,hmf);
            elseif k==2
                [trainedClassifier, validationAccuracy] = MediumTree(trainingData, responseData, idx, ozellikler ,hmf);
            elseif k==3
                [trainedClassifier, validationAccuracy] = CoarseTree(trainingData, responseData, idx, ozellikler ,hmf);
            elseif k==4
                %[trainedClassifier, validationAccuracy] = LineerDiscriminant(trainingData, responseData, idx, ozellikler ,hmf);
            elseif k==5
                %[trainedClassifier, validationAccuracy] = QuadricDiscriminant(trainingData, responseData, idx, ozellikler ,hmf);
            elseif k==6
            %Multiclass için desteklenmiyor
                [trainedClassifier, validationAccuracy] = LogisticRegression(trainingData, responseData, idx, ozellikler ,hmf);
            elseif k==7
                [trainedClassifier, validationAccuracy] = GaussianNaiveBayes(trainingData, responseData, idx, ozellikler ,hmf);
            elseif k==8
                [trainedClassifier, validationAccuracy] = KernelNaiveBayes(trainingData, responseData, idx, ozellikler ,hmf);
            elseif k==9
                [trainedClassifier, validationAccuracy] = LinearSVM(trainingData, responseData, idx, ozellikler ,hmf);
            elseif k==10
                [trainedClassifier, validationAccuracy] = QuadricSVM(trainingData, responseData, idx, ozellikler ,hmf);
            elseif k==11
                [trainedClassifier, validationAccuracy] = CubicSVM(trainingData, responseData, idx, ozellikler ,hmf);
            elseif k==12   
                [trainedClassifier, validationAccuracy] = FineGaussianSVM(trainingData, responseData, idx, ozellikler ,hmf);
            elseif k==13   
                [trainedClassifier, validationAccuracy] = MediumGaussianSVM(trainingData, responseData, idx, ozellikler ,hmf);
            elseif k==14   
                [trainedClassifier, validationAccuracy] = CoarseGaussianSVM(trainingData, responseData, idx, ozellikler ,hmf);
            elseif k==15   
                [trainedClassifier, validationAccuracy] = FineKNN(trainingData, responseData, idx, ozellikler ,hmf);
            elseif k==16
                [trainedClassifier, validationAccuracy] = MediumKNN(trainingData, responseData, idx, ozellikler ,hmf);
            elseif k==17   
                [trainedClassifier, validationAccuracy] = CoarseKNN(trainingData, responseData, idx, ozellikler ,hmf);
            elseif k==18   
                [trainedClassifier, validationAccuracy] = CosineKNN(trainingData, responseData, idx, ozellikler ,hmf);
            elseif k==19   
                [trainedClassifier, validationAccuracy] = CubicKNN(trainingData, responseData, idx, ozellikler ,hmf);
            elseif k==20   
                [trainedClassifier, validationAccuracy] = WeightedKNN(trainingData, responseData, idx, ozellikler ,hmf);
            elseif k==21    
                [trainedClassifier, validationAccuracy] = BoostedTrees(trainingData, responseData, idx, ozellikler ,hmf);
            elseif k==22    
                [trainedClassifier, validationAccuracy] = BaggedTrees(trainingData, responseData, idx, ozellikler ,hmf);
            elseif k==23
                [trainedClassifier, validationAccuracy] = SubspaceDiscriminant(trainingData, responseData, idx, ozellikler ,hmf);
            elseif k==24    
                [trainedClassifier, validationAccuracy] = SubspaceKNN(trainingData, responseData, idx, ozellikler ,hmf);
            elseif k==25    
                [trainedClassifier, validationAccuracy] = RUSBoosterTrees(trainingData, responseData, idx, ozellikler ,hmf);
            end
            
            if validationAccuracy>bestAccuracy
                bestAccuracy = validationAccuracy;
                besthmf = i;       
            end
        end
        result = [result,bestAccuracy];
        result = [result,besthmf];
    end
end

x=14;
table2 = table;
for k=0:24
    
    n1=((x*k)+1);
    n2=x*(k+1);
    
    table1=result(n1:n2);
    table1 =array2table(table1);
    table2=[table2;table1] ; 
    
end    

table2.Properties.VariableNames = {'chi2_acc' 'chi2_fs' 'mrmr_acc' 'mrmr_fs' 'PICT_acc' 'PICT_fs' 'PICE_acc' 'PICE_fs' 'PIRE_acc' 'PIRE_fs' 'relieff_3_acc' 'relieff_3_fs' 'relieff_11_acc' 'relieff_11_fs'};
table2.Properties.RowNames = {'FineTree' 'MediumTree' 'CoarseTree' 'LineerDiscriminant' 'QuadricDiscriminant' 'LogisticRegression' 'GaussianNaiveBayes' 'KernelNaiveBayes' 'LinearSVM' 'QuadricSVM' 'CubicSVM' 'FineGaussianSVM' 'MediumGaussianSVM' 'CoarseGaussianSVM' 'FineKNN' 'MediumKNN' 'CoarseKNN' 'CosineKNN' 'CubicKNN' 'WeightedKNN' 'BoostedTrees' 'BaggedTrees' 'SubspaceDiscriminant' 'SubspaceKNN' 'RUSBoosterTrees'};



%Makine ogrenmesi modelleri
function [trainedClassifier, validationAccuracy] = FineTree(trainingData, responseData, idx, ozellikler ,hmf)

% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
inputTable = trainingData;

predictorNames = {};

for i=1:hmf
    predictorNames = [predictorNames,char(ozellikler(idx(i)))];
end    
predictors = inputTable(:, predictorNames);
response = responseData(:);

isCategoricalPredictor = [];

for i=1:hmf
    isCategoricalPredictor = [isCategoricalPredictor,false];
end    
isCategoricalPredictor = logical(isCategoricalPredictor);

% Train a classifier
% This code specifies all the classifier options and trains the classifier.
classificationTree = fitctree(...
    predictors, ...
    response, ...
    'SplitCriterion', 'gdi', ...
    'MaxNumSplits', 100, ...
    'Surrogate', 'off', ...
    'ClassNames', categorical(unique(responseData)));

% Create the result struct with predict function
predictorExtractionFcn = @(t) t(:, predictorNames);
treePredictFcn = @(x) predict(classificationTree, x);
trainedClassifier.predictFcn = @(x) treePredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedClassifier.RequiredVariables = predictorNames;
trainedClassifier.ClassificationTree = classificationTree;
trainedClassifier.About = 'This struct is a trained model exported from Classification Learner R2020b.';
trainedClassifier.HowToPredict = sprintf('To make predictions on a new table, T, use: \n  yfit = c.predictFcn(T) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nThe table, T, must contain the variables returned by: \n  c.RequiredVariables \nVariable formats (e.g. matrix/vector, datatype) must match the original training data. \nAdditional variables are ignored. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
inputTable = trainingData;
predictorNames;
predictors = inputTable(:, predictorNames);
response = responseData(:);
isCategoricalPredictor = isCategoricalPredictor;

% Perform cross-validation
partitionedModel = crossval(trainedClassifier.ClassificationTree, 'KFold', 50);

% Compute validation predictions
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
end

function [trainedClassifier, validationAccuracy] = MediumTree(trainingData, responseData, idx, ozellikler ,hmf)

% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
inputTable = trainingData;

predictorNames = {};
for i=1:hmf
    predictorNames = [predictorNames,char(ozellikler(idx(i)))];
end 

predictors = inputTable(:, predictorNames);
response = responseData(:);

isCategoricalPredictor = [];
for i=1:hmf
    isCategoricalPredictor = [isCategoricalPredictor,false];
end    
isCategoricalPredictor = logical(isCategoricalPredictor);


% Train a classifier
% This code specifies all the classifier options and trains the classifier.
classificationTree = fitctree(...
    predictors, ...
    response, ...
    'SplitCriterion', 'gdi', ...
    'MaxNumSplits', 20, ...
    'Surrogate', 'off', ...
    'ClassNames', categorical(unique(responseData)));

% Create the result struct with predict function
predictorExtractionFcn = @(t) t(:, predictorNames);
treePredictFcn = @(x) predict(classificationTree, x);
trainedClassifier.predictFcn = @(x) treePredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedClassifier.RequiredVariables = predictorNames;
trainedClassifier.ClassificationTree = classificationTree;
trainedClassifier.About = 'This struct is a trained model exported from Classification Learner R2020b.';
trainedClassifier.HowToPredict = sprintf('To make predictions on a new table, T, use: \n  yfit = c.predictFcn(T) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nThe table, T, must contain the variables returned by: \n  c.RequiredVariables \nVariable formats (e.g. matrix/vector, datatype) must match the original training data. \nAdditional variables are ignored. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
inputTable = trainingData;
predictors = inputTable(:, predictorNames);
response = responseData(:);

% Perform cross-validation
partitionedModel = crossval(trainedClassifier.ClassificationTree, 'KFold', 50);

% Compute validation predictions
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
end

function [trainedClassifier, validationAccuracy] = CoarseTree(trainingData, responseData, idx, ozellikler ,hmf)

inputTable = trainingData;
predictorNames = {};

for i=1:hmf
    predictorNames = [predictorNames,char(ozellikler(idx(i)))];
end    
predictors = inputTable(:, predictorNames);
response = responseData(:);
isCategoricalPredictor = [];

for i=1:hmf
    isCategoricalPredictor = [isCategoricalPredictor,false];
end    
isCategoricalPredictor = logical(isCategoricalPredictor);

% Train a classifier
% This code specifies all the classifier options and trains the classifier.
classificationTree = fitctree(...
    predictors, ...
    response, ...
    'SplitCriterion', 'gdi', ...
    'MaxNumSplits', 4, ...
    'Surrogate', 'off', ...
    'ClassNames', categorical(unique(responseData)));

% Create the result struct with predict function
predictorExtractionFcn = @(t) t(:, predictorNames);
treePredictFcn = @(x) predict(classificationTree, x);
trainedClassifier.predictFcn = @(x) treePredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedClassifier.RequiredVariables = predictorNames;
trainedClassifier.ClassificationTree = classificationTree;
trainedClassifier.About = 'This struct is a trained model exported from Classification Learner R2020b.';
trainedClassifier.HowToPredict = sprintf('To make predictions on a new table, T, use: \n  yfit = c.predictFcn(T) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nThe table, T, must contain the variables returned by: \n  c.RequiredVariables \nVariable formats (e.g. matrix/vector, datatype) must match the original training data. \nAdditional variables are ignored. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
inputTable = trainingData;
predictors = inputTable(:, predictorNames);
response = responseData(:);

% Perform cross-validation
partitionedModel = crossval(trainedClassifier.ClassificationTree, 'KFold', 50);

% Compute validation predictions
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');

end
%Bu Modeldeki discrimType değişkeni ile ilgili problem.
function [trainedClassifier, validationAccuracy] = LineerDiscriminant(trainingData, responseData, idx, ozellikler ,hmf)

inputTable = trainingData;
predictorNames = {};

for i=1:hmf
    predictorNames = [predictorNames,char(ozellikler(idx(i)))];
end
predictors = inputTable(:, predictorNames);
response = responseData(:);
isCategoricalPredictor = [];

for i=1:hmf
    isCategoricalPredictor = [isCategoricalPredictor,false];
end    
isCategoricalPredictor = logical(isCategoricalPredictor);

% Train a classifier
% This code specifies all the classifier options and trains the classifier.
classificationDiscriminant = fitcdiscr(...
    predictors, ...
    response, ...
    'discrimType', 'linear', ...
    'Gamma', 0, ...
    'FillCoeffs', 'off', ...
    'ClassNames', categorical(unique(responseData)));   

% Create the result struct with predict function
predictorExtractionFcn = @(t) t(:, predictorNames);
discriminantPredictFcn = @(x) predict(classificationDiscriminant, x);
trainedClassifier.predictFcn = @(x) discriminantPredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedClassifier.RequiredVariables = predictorNames;
trainedClassifier.ClassificationDiscriminant = classificationDiscriminant;
trainedClassifier.About = 'This struct is a trained model exported from Classification Learner R2020b.';
trainedClassifier.HowToPredict = sprintf('To make predictions on a new table, T, use: \n  yfit = c.predictFcn(T) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nThe table, T, must contain the variables returned by: \n  c.RequiredVariables \nVariable formats (e.g. matrix/vector, datatype) must match the original training data. \nAdditional variables are ignored. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
inputTable = trainingData;
predictors = inputTable(:, predictorNames);
response = responseData(:);
isCategoricalPredictor;
% Perform cross-validation
partitionedModel = crossval(trainedClassifier.ClassificationDiscriminant, 'KFold', 50);

% Compute validation predictions
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
end

function [trainedClassifier, validationAccuracy] = QuadricDiscriminant(trainingData, responseData, idx, ozellikler ,hmf)

inputTable = trainingData;
predictorNames = {};

for i=1:hmf
    predictorNames = [predictorNames,char(ozellikler(idx(i)))];
end
predictors = inputTable(:, predictorNames);
response = responseData(:);
isCategoricalPredictor = [];

for i=1:hmf
    isCategoricalPredictor = [isCategoricalPredictor,false];
end    
isCategoricalPredictor = logical(isCategoricalPredictor);

% Train a classifier
% This code specifies all the classifier options and trains the classifier.
classificationDiscriminant = fitcdiscr(...
    predictors, ...
    response, ...
    'DiscrimType', 'quadratic', ...
    'FillCoeffs', 'off', ...
    'ClassNames', categorical(unique(responseData)));

% Create the result struct with predict function
predictorExtractionFcn = @(t) t(:, predictorNames);
discriminantPredictFcn = @(x) predict(classificationDiscriminant, x);
trainedClassifier.predictFcn = @(x) discriminantPredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedClassifier.RequiredVariables = predictorNames;
trainedClassifier.ClassificationDiscriminant = classificationDiscriminant;
trainedClassifier.About = 'This struct is a trained model exported from Classification Learner R2020b.';
trainedClassifier.HowToPredict = sprintf('To make predictions on a new table, T, use: \n  yfit = c.predictFcn(T) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nThe table, T, must contain the variables returned by: \n  c.RequiredVariables \nVariable formats (e.g. matrix/vector, datatype) must match the original training data. \nAdditional variables are ignored. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
inputTable = trainingData;
predictors = inputTable(:, predictorNames);
response = responseData(:);

% Perform cross-validation
partitionedModel = crossval(trainedClassifier.ClassificationDiscriminant, 'KFold', 50);

% Compute validation predictions
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
end
%Multiclass için uygun değil
function [trainedClassifier, validationAccuracy] = LogisticRegression(trainingData, responseData, idx, ozellikler ,hmf)

inputTable = trainingData;
predictorNames = {};

for i=1:hmf
    predictorNames = [predictorNames,char(ozellikler(idx(i)))];
end    
predictors = inputTable(:, predictorNames);
response = responseData(:);
isCategoricalPredictor = [];

for i=1:hmf
    isCategoricalPredictor = [isCategoricalPredictor,false];
end    
isCategoricalPredictor = logical(isCategoricalPredictor);

% Train a classifier
% This code specifies all the classifier options and trains the classifier.
% For logistic regression, the response values must be converted to zeros
% and ones because the responses are assumed to follow a binomial
% distribution.
% 1 or true = 'successful' class
% 0 or false = 'failure' class
% NaN - missing response.
successClass = '0';
failureClass = '1';
% Compute the majority response class. If there is a NaN-prediction from
% fitglm, convert NaN to this majority class label.
numSuccess = sum(response == successClass);
numFailure = sum(response == failureClass);
if numSuccess > numFailure
    missingClass = successClass;
else
    missingClass = failureClass;
end
responseCategories = {successClass, failureClass};
successFailureAndMissingClasses = categorical({successClass; failureClass; missingClass}, responseCategories);
isMissing = isundefined(response);
zeroOneResponse = double(ismember(response, successClass));
zeroOneResponse(isMissing) = NaN;
% Prepare input arguments to fitglm.
concatenatedPredictorsAndResponse = [predictors, table(zeroOneResponse)];
% Train using fitglm.
GeneralizedLinearModel = fitglm(...
    concatenatedPredictorsAndResponse, ...
    'Distribution', 'binomial', ...
    'link', 'logit');

% Convert predicted probabilities to predicted class labels and scores.
convertSuccessProbsToPredictions = @(p) successFailureAndMissingClasses( ~isnan(p).*( (p<0.5) + 1 ) + isnan(p)*3 );
returnMultipleValuesFcn = @(varargin) varargin{1:max(1,nargout)};
scoresFcn = @(p) [p, 1-p];
predictionsAndScoresFcn = @(p) returnMultipleValuesFcn( convertSuccessProbsToPredictions(p), scoresFcn(p) );

% Create the result struct with predict function
predictorExtractionFcn = @(t) t(:, predictorNames);
logisticRegressionPredictFcn = @(x) predictionsAndScoresFcn( predict(GeneralizedLinearModel, x) );
trainedClassifier.predictFcn = @(x) logisticRegressionPredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedClassifier.RequiredVariables = predictorNames;
trainedClassifier.GeneralizedLinearModel = GeneralizedLinearModel;
trainedClassifier.SuccessClass = successClass;
trainedClassifier.FailureClass = failureClass;
trainedClassifier.MissingClass = missingClass;
trainedClassifier.ClassNames = {successClass; failureClass};
trainedClassifier.About = 'This struct is a trained model exported from Classification Learner R2020b.';
trainedClassifier.HowToPredict = sprintf('To make predictions on a new table, T, use: \n  yfit = c.predictFcn(T) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nThe table, T, must contain the variables returned by: \n  c.RequiredVariables \nVariable formats (e.g. matrix/vector, datatype) must match the original training data. \nAdditional variables are ignored. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
inputTable = trainingData;
predictors = inputTable(:, predictorNames);
response = responseData(:);

% Perform cross-validation
KFolds = 50;
cvp = cvpartition(response, 'KFold', KFolds);
% Initialize the predictions to the proper sizes
validationPredictions = response;
numObservations = size(predictors, 1);
numClasses = 2;
validationScores = NaN(numObservations, numClasses);
for fold = 1:KFolds
    trainingPredictors = predictors(cvp.training(fold), :);
    trainingResponse = response(cvp.training(fold), :);
    foldIsCategoricalPredictor = isCategoricalPredictor;
    
    % Train a classifier
    % This code specifies all the classifier options and trains the classifier.
    % For logistic regression, the response values must be converted to zeros
    % and ones because the responses are assumed to follow a binomial
    % distribution.
    % 1 or true = 'successful' class
    % 0 or false = 'failure' class
    % NaN - missing response.
    successClass = '0';
    failureClass = '1';
    % Compute the majority response class. If there is a NaN-prediction from
    % fitglm, convert NaN to this majority class label.
    numSuccess = sum(trainingResponse == successClass);
    numFailure = sum(trainingResponse == failureClass);
    if numSuccess > numFailure
        missingClass = successClass;
    else
        missingClass = failureClass;
    end
    responseCategories = {successClass, failureClass};
    successFailureAndMissingClasses = categorical({successClass; failureClass; missingClass}, responseCategories);
    isMissing = isundefined(trainingResponse);
    zeroOneResponse = double(ismember(trainingResponse, successClass));
    zeroOneResponse(isMissing) = NaN;
    % Prepare input arguments to fitglm.
    concatenatedPredictorsAndResponse = [trainingPredictors, table(zeroOneResponse)];
    % Train using fitglm.
    GeneralizedLinearModel = fitglm(...
        concatenatedPredictorsAndResponse, ...
        'Distribution', 'binomial', ...
        'link', 'logit');
    
    % Convert predicted probabilities to predicted class labels and scores.
    convertSuccessProbsToPredictions = @(p) successFailureAndMissingClasses( ~isnan(p).*( (p<0.5) + 1 ) + isnan(p)*3 );
    returnMultipleValuesFcn = @(varargin) varargin{1:max(1,nargout)};
    scoresFcn = @(p) [p, 1-p];
    predictionsAndScoresFcn = @(p) returnMultipleValuesFcn( convertSuccessProbsToPredictions(p), scoresFcn(p) );
    
    % Create the result struct with predict function
    logisticRegressionPredictFcn = @(x) predictionsAndScoresFcn( predict(GeneralizedLinearModel, x) );
    validationPredictFcn = @(x) logisticRegressionPredictFcn(x);
    
    % Add additional fields to the result struct
    
    % Compute validation predictions
    validationPredictors = predictors(cvp.test(fold), :);
    [foldPredictions, foldScores] = validationPredictFcn(validationPredictors);
    
    % Store predictions in the original order
    validationPredictions(cvp.test(fold), :) = foldPredictions;
    validationScores(cvp.test(fold), :) = foldScores;
end

% Compute validation accuracy
correctPredictions = (validationPredictions == response);
isMissing = ismissing(response);
correctPredictions = correctPredictions(~isMissing);
validationAccuracy = sum(correctPredictions)/length(correctPredictions);
end

function [trainedClassifier, validationAccuracy] = GaussianNaiveBayes(trainingData, responseData, idx, ozellikler ,hmf)

inputTable = trainingData;
predictorNames = {};

for i=1:hmf
    predictorNames = [predictorNames,char(ozellikler(idx(i)))];
end
predictors = inputTable(:, predictorNames);
response = responseData(:);
isCategoricalPredictor = [];

for i=1:hmf
    isCategoricalPredictor = [isCategoricalPredictor,false];
end    
isCategoricalPredictor = logical(isCategoricalPredictor);

% Train a classifier
% This code specifies all the classifier options and trains the classifier.

% Expand the Distribution Names per predictor
% Numerical predictors are assigned either Gaussian or Kernel distribution and categorical predictors are assigned mvmn distribution
% Gaussian is replaced with Normal when passing to the fitcnb function
distributionNames =  repmat({'Normal'}, 1, length(isCategoricalPredictor));
distributionNames(isCategoricalPredictor) = {'mvmn'};

if any(strcmp(distributionNames,'Kernel'))
    classificationNaiveBayes = fitcnb(...
        predictors, ...
        response, ...
        'Kernel', 'Normal', ...
        'Support', 'Unbounded', ...
        'DistributionNames', distributionNames, ...
        'ClassNames', categorical(unique(responseData)));
else
    classificationNaiveBayes = fitcnb(...
        predictors, ...
        response, ...
        'DistributionNames', distributionNames, ...
        'ClassNames', categorical(unique(responseData)));
end

% Create the result struct with predict function
predictorExtractionFcn = @(t) t(:, predictorNames);
naiveBayesPredictFcn = @(x) predict(classificationNaiveBayes, x);
trainedClassifier.predictFcn = @(x) naiveBayesPredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedClassifier.RequiredVariables = predictorNames;
trainedClassifier.ClassificationNaiveBayes = classificationNaiveBayes;
trainedClassifier.About = 'This struct is a trained model exported from Classification Learner R2020b.';
trainedClassifier.HowToPredict = sprintf('To make predictions on a new table, T, use: \n  yfit = c.predictFcn(T) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nThe table, T, must contain the variables returned by: \n  c.RequiredVariables \nVariable formats (e.g. matrix/vector, datatype) must match the original training data. \nAdditional variables are ignored. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
inputTable = trainingData;
predictorNames;
predictors = inputTable(:, predictorNames);
response = responseData(:);
isCategoricalPredictor;
% Perform cross-validation
partitionedModel = crossval(trainedClassifier.ClassificationNaiveBayes, 'KFold', 50);

% Compute validation predictions
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
end

function [trainedClassifier, validationAccuracy] = KernelNaiveBayes(trainingData, responseData, idx, ozellikler ,hmf)

inputTable = trainingData;
predictorNames = {};

for i=1:hmf
    predictorNames = [predictorNames,char(ozellikler(idx(i)))];
end
predictors = inputTable(:, predictorNames);
response = responseData(:);
isCategoricalPredictor = [];

for i=1:hmf
    isCategoricalPredictor = [isCategoricalPredictor,false];
end    
isCategoricalPredictor = logical(isCategoricalPredictor);

% Train a classifier
% This code specifies all the classifier options and trains the classifier.

% Expand the Distribution Names per predictor
% Numerical predictors are assigned either Gaussian or Kernel distribution and categorical predictors are assigned mvmn distribution
% Gaussian is replaced with Normal when passing to the fitcnb function
distributionNames =  repmat({'Kernel'}, 1, length(isCategoricalPredictor));
distributionNames(isCategoricalPredictor) = {'mvmn'};

if any(strcmp(distributionNames,'Kernel'))
    classificationNaiveBayes = fitcnb(...
        predictors, ...
        response, ...
        'Kernel', 'Normal', ...
        'Support', 'Unbounded', ...
        'DistributionNames', distributionNames, ...
        'ClassNames', categorical(unique(responseData)));
else
    classificationNaiveBayes = fitcnb(...
        predictors, ...
        response, ...
        'DistributionNames', distributionNames, ...
        'ClassNames', categorical(unique(responseData)));
end

% Create the result struct with predict function
predictorExtractionFcn = @(t) t(:, predictorNames);
naiveBayesPredictFcn = @(x) predict(classificationNaiveBayes, x);
trainedClassifier.predictFcn = @(x) naiveBayesPredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedClassifier.RequiredVariables = predictorNames;
trainedClassifier.ClassificationNaiveBayes = classificationNaiveBayes;
trainedClassifier.About = 'This struct is a trained model exported from Classification Learner R2020b.';
trainedClassifier.HowToPredict = sprintf('To make predictions on a new table, T, use: \n  yfit = c.predictFcn(T) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nThe table, T, must contain the variables returned by: \n  c.RequiredVariables \nVariable formats (e.g. matrix/vector, datatype) must match the original training data. \nAdditional variables are ignored. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
inputTable = trainingData;
predictorNames;
predictors = inputTable(:, predictorNames);
response = responseData(:);
isCategoricalPredictor;

% Perform cross-validation
partitionedModel = crossval(trainedClassifier.ClassificationNaiveBayes, 'KFold', 50);

% Compute validation predictions
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
end

function [trainedClassifier, validationAccuracy] = LinearSVM(trainingData, responseData, idx, ozellikler ,hmf)

inputTable = trainingData;
predictorNames = {};

for i=1:hmf
    predictorNames = [predictorNames,char(ozellikler(idx(i)))];
end
predictors = inputTable(:, predictorNames);
response = responseData(:);
isCategoricalPredictor = [];

for i=1:hmf
    isCategoricalPredictor = [isCategoricalPredictor,false];
end    
isCategoricalPredictor = logical(isCategoricalPredictor);

% Train a classifier
% This code specifies all the classifier options and trains the classifier.
template = templateSVM(...
    'KernelFunction', 'linear', ...
    'PolynomialOrder', [], ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true);
classificationSVM = fitcecoc(...
    predictors, ...
    response, ...
    'Learners', template, ...
    'Coding', 'onevsone', ...
    'ClassNames', categorical(unique(responseData)));

% Create the result struct with predict function
predictorExtractionFcn = @(t) t(:, predictorNames);
svmPredictFcn = @(x) predict(classificationSVM, x);
trainedClassifier.predictFcn = @(x) svmPredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedClassifier.RequiredVariables = predictorNames;
trainedClassifier.ClassificationSVM = classificationSVM;
trainedClassifier.About = 'This struct is a trained model exported from Classification Learner R2020b.';
trainedClassifier.HowToPredict = sprintf('To make predictions on a new table, T, use: \n  yfit = c.predictFcn(T) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nThe table, T, must contain the variables returned by: \n  c.RequiredVariables \nVariable formats (e.g. matrix/vector, datatype) must match the original training data. \nAdditional variables are ignored. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
inputTable = trainingData;
predictorNames;
predictors = inputTable(:, predictorNames);
response = responseData(:);
isCategoricalPredictor;

% Perform cross-validation
partitionedModel = crossval(trainedClassifier.ClassificationSVM, 'KFold', 50);

% Compute validation predictions
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
end

function [trainedClassifier, validationAccuracy] = QuadricSVM(trainingData, responseData, idx, ozellikler ,hmf)

inputTable = trainingData;
predictorNames = {};

for i=1:hmf
    predictorNames = [predictorNames,char(ozellikler(idx(i)))];
end
predictors = inputTable(:, predictorNames);
response = responseData(:);
isCategoricalPredictor = [];

for i=1:hmf
    isCategoricalPredictor = [isCategoricalPredictor,false];
end    
isCategoricalPredictor = logical(isCategoricalPredictor);

% Train a classifier
% This code specifies all the classifier options and trains the classifier.
template = templateSVM(...
    'KernelFunction', 'polynomial', ...
    'PolynomialOrder', 2, ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true);
classificationSVM = fitcecoc(...
    predictors, ...
    response, ...
    'Learners', template, ...
    'Coding', 'onevsone', ...
    'ClassNames', categorical(unique(responseData)));

% Create the result struct with predict function
predictorExtractionFcn = @(t) t(:, predictorNames);
svmPredictFcn = @(x) predict(classificationSVM, x);
trainedClassifier.predictFcn = @(x) svmPredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedClassifier.RequiredVariables = predictorNames;
trainedClassifier.ClassificationSVM = classificationSVM;
trainedClassifier.About = 'This struct is a trained model exported from Classification Learner R2020b.';
trainedClassifier.HowToPredict = sprintf('To make predictions on a new table, T, use: \n  yfit = c.predictFcn(T) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nThe table, T, must contain the variables returned by: \n  c.RequiredVariables \nVariable formats (e.g. matrix/vector, datatype) must match the original training data. \nAdditional variables are ignored. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
inputTable = trainingData;
predictors = inputTable(:, predictorNames);
response = responseData(:);

% Perform cross-validation
partitionedModel = crossval(trainedClassifier.ClassificationSVM, 'KFold', 50);

% Compute validation predictions
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
end

function [trainedClassifier, validationAccuracy] = CubicSVM(trainingData, responseData, idx, ozellikler ,hmf)

inputTable = trainingData;
predictorNames = {};

for i=1:hmf
    predictorNames = [predictorNames,char(ozellikler(idx(i)))];
end
predictors = inputTable(:, predictorNames);
response = responseData(:);
isCategoricalPredictor = [];

for i=1:hmf
    isCategoricalPredictor = [isCategoricalPredictor,false];
end    
isCategoricalPredictor = logical(isCategoricalPredictor);

% Train a classifier
% This code specifies all the classifier options and trains the classifier.
template = templateSVM(...
    'KernelFunction', 'polynomial', ...
    'PolynomialOrder', 3, ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true);
classificationSVM = fitcecoc(...
    predictors, ...
    response, ...
    'Learners', template, ...
    'Coding', 'onevsone', ...
    'ClassNames', categorical(unique(responseData)));

% Create the result struct with predict function
predictorExtractionFcn = @(t) t(:, predictorNames);
svmPredictFcn = @(x) predict(classificationSVM, x);
trainedClassifier.predictFcn = @(x) svmPredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedClassifier.RequiredVariables = predictorNames;
trainedClassifier.ClassificationSVM = classificationSVM;
trainedClassifier.About = 'This struct is a trained model exported from Classification Learner R2020b.';
trainedClassifier.HowToPredict = sprintf('To make predictions on a new table, T, use: \n  yfit = c.predictFcn(T) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nThe table, T, must contain the variables returned by: \n  c.RequiredVariables \nVariable formats (e.g. matrix/vector, datatype) must match the original training data. \nAdditional variables are ignored. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
inputTable = trainingData;
predictors = inputTable(:, predictorNames);
response = responseData(:);

% Perform cross-validation
partitionedModel = crossval(trainedClassifier.ClassificationSVM, 'KFold', 50);

% Compute validation predictions
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
end

function [trainedClassifier, validationAccuracy] = FineGaussianSVM(trainingData, responseData, idx, ozellikler ,hmf)

inputTable = trainingData;
predictorNames = {};

for i=1:hmf
    predictorNames = [predictorNames,char(ozellikler(idx(i)))];
end
predictors = inputTable(:, predictorNames);
response = responseData(:);
isCategoricalPredictor = [];

for i=1:hmf
    isCategoricalPredictor = [isCategoricalPredictor,false];
end    
isCategoricalPredictor = logical(isCategoricalPredictor);

% Train a classifier
% This code specifies all the classifier options and trains the classifier.
template = templateSVM(...
    'KernelFunction', 'gaussian', ...
    'PolynomialOrder', [], ...
    'KernelScale', 0.5600000000000001, ...
    'BoxConstraint', 1, ...
    'Standardize', true);
classificationSVM = fitcecoc(...
    predictors, ...
    response, ...
    'Learners', template, ...
    'Coding', 'onevsone', ...
    'ClassNames', categorical(unique(responseData)));

% Create the result struct with predict function
predictorExtractionFcn = @(t) t(:, predictorNames);
svmPredictFcn = @(x) predict(classificationSVM, x);
trainedClassifier.predictFcn = @(x) svmPredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedClassifier.RequiredVariables = predictorNames;
trainedClassifier.ClassificationSVM = classificationSVM;
trainedClassifier.About = 'This struct is a trained model exported from Classification Learner R2020b.';
trainedClassifier.HowToPredict = sprintf('To make predictions on a new table, T, use: \n  yfit = c.predictFcn(T) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nThe table, T, must contain the variables returned by: \n  c.RequiredVariables \nVariable formats (e.g. matrix/vector, datatype) must match the original training data. \nAdditional variables are ignored. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
inputTable = trainingData;
predictors = inputTable(:, predictorNames);
response = responseData(:);

% Perform cross-validation
partitionedModel = crossval(trainedClassifier.ClassificationSVM, 'KFold', 50);

% Compute validation predictions
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
end

function [trainedClassifier, validationAccuracy] = MediumGaussianSVM(trainingData, responseData, idx, ozellikler ,hmf)

inputTable = trainingData;
predictorNames = {};

for i=1:hmf
    predictorNames = [predictorNames,char(ozellikler(idx(i)))];
end
predictors = inputTable(:, predictorNames);
response = responseData(:);
isCategoricalPredictor = [];

for i=1:hmf
    isCategoricalPredictor = [isCategoricalPredictor,false];
end    
isCategoricalPredictor = logical(isCategoricalPredictor);

% Train a classifier
% This code specifies all the classifier options and trains the classifier.
template = templateSVM(...
    'KernelFunction', 'gaussian', ...
    'PolynomialOrder', [], ...
    'KernelScale', 2.2, ...
    'BoxConstraint', 1, ...
    'Standardize', true);
classificationSVM = fitcecoc(...
    predictors, ...
    response, ...
    'Learners', template, ...
    'Coding', 'onevsone', ...
    'ClassNames', categorical(unique(responseData)));

% Create the result struct with predict function
predictorExtractionFcn = @(t) t(:, predictorNames);
svmPredictFcn = @(x) predict(classificationSVM, x);
trainedClassifier.predictFcn = @(x) svmPredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedClassifier.RequiredVariables = predictorNames;
trainedClassifier.ClassificationSVM = classificationSVM;
trainedClassifier.About = 'This struct is a trained model exported from Classification Learner R2020b.';
trainedClassifier.HowToPredict = sprintf('To make predictions on a new table, T, use: \n  yfit = c.predictFcn(T) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nThe table, T, must contain the variables returned by: \n  c.RequiredVariables \nVariable formats (e.g. matrix/vector, datatype) must match the original training data. \nAdditional variables are ignored. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
inputTable = trainingData;
predictors = inputTable(:, predictorNames);
response = responseData(:);

% Perform cross-validation
partitionedModel = crossval(trainedClassifier.ClassificationSVM, 'KFold', 50);

% Compute validation predictions
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
end

function [trainedClassifier, validationAccuracy] = CoarseGaussianSVM(trainingData, responseData, idx, ozellikler ,hmf)

inputTable = trainingData;
predictorNames = {};

for i=1:hmf
    predictorNames = [predictorNames,char(ozellikler(idx(i)))];
end
predictors = inputTable(:, predictorNames);
response = responseData(:);
isCategoricalPredictor = [];

for i=1:hmf
    isCategoricalPredictor = [isCategoricalPredictor,false];
end    
isCategoricalPredictor = logical(isCategoricalPredictor);

% Train a classifier
% This code specifies all the classifier options and trains the classifier.
template = templateSVM(...
    'KernelFunction', 'gaussian', ...
    'PolynomialOrder', [], ...
    'KernelScale', 8.9, ...
    'BoxConstraint', 1, ...
    'Standardize', true);
classificationSVM = fitcecoc(...
    predictors, ...
    response, ...
    'Learners', template, ...
    'Coding', 'onevsone', ...
    'ClassNames', categorical(unique(responseData)));
% Create the result struct with predict function
predictorExtractionFcn = @(t) t(:, predictorNames);
svmPredictFcn = @(x) predict(classificationSVM, x);
trainedClassifier.predictFcn = @(x) svmPredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedClassifier.RequiredVariables = predictorNames;
trainedClassifier.ClassificationSVM = classificationSVM;
trainedClassifier.About = 'This struct is a trained model exported from Classification Learner R2020b.';
trainedClassifier.HowToPredict = sprintf('To make predictions on a new table, T, use: \n  yfit = c.predictFcn(T) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nThe table, T, must contain the variables returned by: \n  c.RequiredVariables \nVariable formats (e.g. matrix/vector, datatype) must match the original training data. \nAdditional variables are ignored. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
inputTable = trainingData;
predictors = inputTable(:, predictorNames);
response = responseData(:);

% Perform cross-validation
partitionedModel = crossval(trainedClassifier.ClassificationSVM, 'KFold', 50);

% Compute validation predictions
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
end

function [trainedClassifier, validationAccuracy] = FineKNN(trainingData, responseData, idx, ozellikler ,hmf)

inputTable = trainingData;
predictorNames = {};

for i=1:hmf
    predictorNames = [predictorNames,char(ozellikler(idx(i)))];
end
predictors = inputTable(:, predictorNames);
response = responseData(:);
isCategoricalPredictor = [];

for i=1:hmf
    isCategoricalPredictor = [isCategoricalPredictor,false];
end    
isCategoricalPredictor = logical(isCategoricalPredictor);

% Train a classifier
% This code specifies all the classifier options and trains the classifier.
classificationKNN = fitcknn(...
    predictors, ...
    response, ...
    'Distance', 'Euclidean', ...
    'Exponent', [], ...
    'NumNeighbors', 1, ...
    'DistanceWeight', 'Equal', ...
    'Standardize', true, ...
    'ClassNames', categorical(unique(responseData)));

% Create the result struct with predict function
predictorExtractionFcn = @(t) t(:, predictorNames);
knnPredictFcn = @(x) predict(classificationKNN, x);
trainedClassifier.predictFcn = @(x) knnPredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedClassifier.RequiredVariables = predictorNames;
trainedClassifier.ClassificationKNN = classificationKNN;
trainedClassifier.About = 'This struct is a trained model exported from Classification Learner R2020b.';
trainedClassifier.HowToPredict = sprintf('To make predictions on a new table, T, use: \n  yfit = c.predictFcn(T) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nThe table, T, must contain the variables returned by: \n  c.RequiredVariables \nVariable formats (e.g. matrix/vector, datatype) must match the original training data. \nAdditional variables are ignored. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
inputTable = trainingData;
predictors = inputTable(:, predictorNames);
response = responseData(:);

% Perform cross-validation
partitionedModel = crossval(trainedClassifier.ClassificationKNN, 'KFold', 50);

% Compute validation predictions
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
end

function [trainedClassifier, validationAccuracy] = MediumKNN(trainingData, responseData, idx, ozellikler ,hmf)

inputTable = trainingData;
predictorNames = {};

for i=1:hmf
    predictorNames = [predictorNames,char(ozellikler(idx(i)))];
end
predictors = inputTable(:, predictorNames);
response = responseData(:);
isCategoricalPredictor = [];

for i=1:hmf
    isCategoricalPredictor = [isCategoricalPredictor,false];
end    
isCategoricalPredictor = logical(isCategoricalPredictor);

% Train a classifier
% This code specifies all the classifier options and trains the classifier.
classificationKNN = fitcknn(...
    predictors, ...
    response, ...
    'Distance', 'Euclidean', ...
    'Exponent', [], ...
    'NumNeighbors', 10, ...
    'DistanceWeight', 'Equal', ...
    'Standardize', true, ...
    'ClassNames', categorical(unique(responseData)));

% Create the result struct with predict function
predictorExtractionFcn = @(t) t(:, predictorNames);
knnPredictFcn = @(x) predict(classificationKNN, x);
trainedClassifier.predictFcn = @(x) knnPredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedClassifier.RequiredVariables = predictorNames;
trainedClassifier.ClassificationKNN = classificationKNN;
trainedClassifier.About = 'This struct is a trained model exported from Classification Learner R2020b.';
trainedClassifier.HowToPredict = sprintf('To make predictions on a new table, T, use: \n  yfit = c.predictFcn(T) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nThe table, T, must contain the variables returned by: \n  c.RequiredVariables \nVariable formats (e.g. matrix/vector, datatype) must match the original training data. \nAdditional variables are ignored. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
inputTable = trainingData;
predictors = inputTable(:, predictorNames);
response = responseData(:);

% Perform cross-validation
partitionedModel = crossval(trainedClassifier.ClassificationKNN, 'KFold', 50);

% Compute validation predictions
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
end

function [trainedClassifier, validationAccuracy] = CoarseKNN(trainingData, responseData, idx, ozellikler ,hmf)

inputTable = trainingData;
predictorNames = {};

for i=1:hmf
    predictorNames = [predictorNames,char(ozellikler(idx(i)))];
end
predictors = inputTable(:, predictorNames);
response = responseData(:);
isCategoricalPredictor = [];

for i=1:hmf
    isCategoricalPredictor = [isCategoricalPredictor,false];
end    
isCategoricalPredictor = logical(isCategoricalPredictor);

% Train a classifier
% This code specifies all the classifier options and trains the classifier.
classificationKNN = fitcknn(...
    predictors, ...
    response, ...
    'Distance', 'Euclidean', ...
    'Exponent', [], ...
    'NumNeighbors', 100, ...
    'DistanceWeight', 'Equal', ...
    'Standardize', true, ...
    'ClassNames', categorical(unique(responseData)));

% Create the result struct with predict function
predictorExtractionFcn = @(t) t(:, predictorNames);
knnPredictFcn = @(x) predict(classificationKNN, x);
trainedClassifier.predictFcn = @(x) knnPredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedClassifier.RequiredVariables = predictorNames;
trainedClassifier.ClassificationKNN = classificationKNN;
trainedClassifier.About = 'This struct is a trained model exported from Classification Learner R2020b.';
trainedClassifier.HowToPredict = sprintf('To make predictions on a new table, T, use: \n  yfit = c.predictFcn(T) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nThe table, T, must contain the variables returned by: \n  c.RequiredVariables \nVariable formats (e.g. matrix/vector, datatype) must match the original training data. \nAdditional variables are ignored. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
inputTable = trainingData;
predictors = inputTable(:, predictorNames);
response = responseData(:);

% Perform cross-validation
partitionedModel = crossval(trainedClassifier.ClassificationKNN, 'KFold', 50);

% Compute validation predictions
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
end

function [trainedClassifier, validationAccuracy] = CosineKNN(trainingData, responseData, idx, ozellikler ,hmf)

inputTable = trainingData;
predictorNames = {};

for i=1:hmf
    predictorNames = [predictorNames,char(ozellikler(idx(i)))];
end
predictors = inputTable(:, predictorNames);
response = responseData(:);
isCategoricalPredictor = [];

for i=1:hmf
    isCategoricalPredictor = [isCategoricalPredictor,false];
end    
isCategoricalPredictor = logical(isCategoricalPredictor);

% Train a classifier
% This code specifies all the classifier options and trains the classifier.
classificationKNN = fitcknn(...
    predictors, ...
    response, ...
    'Distance', 'Cosine', ...
    'Exponent', [], ...
    'NumNeighbors', 10, ...
    'DistanceWeight', 'Equal', ...
    'Standardize', true, ...
    'ClassNames', categorical(unique(responseData)));

% Create the result struct with predict function
predictorExtractionFcn = @(t) t(:, predictorNames);
knnPredictFcn = @(x) predict(classificationKNN, x);
trainedClassifier.predictFcn = @(x) knnPredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedClassifier.RequiredVariables = predictorNames;
trainedClassifier.ClassificationKNN = classificationKNN;
trainedClassifier.About = 'This struct is a trained model exported from Classification Learner R2020b.';
trainedClassifier.HowToPredict = sprintf('To make predictions on a new table, T, use: \n  yfit = c.predictFcn(T) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nThe table, T, must contain the variables returned by: \n  c.RequiredVariables \nVariable formats (e.g. matrix/vector, datatype) must match the original training data. \nAdditional variables are ignored. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
inputTable = trainingData;
predictors = inputTable(:, predictorNames);
response = responseData(:);

% Perform cross-validation
partitionedModel = crossval(trainedClassifier.ClassificationKNN, 'KFold', 50);

% Compute validation predictions
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
end

function [trainedClassifier, validationAccuracy] = CubicKNN(trainingData, responseData, idx, ozellikler ,hmf)

inputTable = trainingData;
predictorNames = {};

for i=1:hmf
    predictorNames = [predictorNames,char(ozellikler(idx(i)))];
end
predictors = inputTable(:, predictorNames);
response = responseData(:);
isCategoricalPredictor = [];

for i=1:hmf
    isCategoricalPredictor = [isCategoricalPredictor,false];
end    
isCategoricalPredictor = logical(isCategoricalPredictor);

% Train a classifier
% This code specifies all the classifier options and trains the classifier.
classificationKNN = fitcknn(...
    predictors, ...
    response, ...
    'Distance', 'Minkowski', ...
    'Exponent', 3, ...
    'NumNeighbors', 10, ...
    'DistanceWeight', 'Equal', ...
    'Standardize', true, ...
    'ClassNames', categorical(unique(responseData)));

% Create the result struct with predict function
predictorExtractionFcn = @(t) t(:, predictorNames);
knnPredictFcn = @(x) predict(classificationKNN, x);
trainedClassifier.predictFcn = @(x) knnPredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedClassifier.RequiredVariables = predictorNames;
trainedClassifier.ClassificationKNN = classificationKNN;
trainedClassifier.About = 'This struct is a trained model exported from Classification Learner R2020b.';
trainedClassifier.HowToPredict = sprintf('To make predictions on a new table, T, use: \n  yfit = c.predictFcn(T) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nThe table, T, must contain the variables returned by: \n  c.RequiredVariables \nVariable formats (e.g. matrix/vector, datatype) must match the original training data. \nAdditional variables are ignored. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
inputTable = trainingData;
predictors = inputTable(:, predictorNames);
response = responseData(:);

% Perform cross-validation
partitionedModel = crossval(trainedClassifier.ClassificationKNN, 'KFold', 50);

% Compute validation predictions
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
end

function [trainedClassifier, validationAccuracy] = WeightedKNN(trainingData, responseData, idx, ozellikler ,hmf)

inputTable = trainingData;
predictorNames = {};

for i=1:hmf
    predictorNames = [predictorNames,char(ozellikler(idx(i)))];
end    
predictors = inputTable(:, predictorNames);
response = responseData(:);
isCategoricalPredictor = [];

for i=1:hmf
    isCategoricalPredictor = [isCategoricalPredictor,false];
end    
isCategoricalPredictor = logical(isCategoricalPredictor);

% Train a classifier
% This code specifies all the classifier options and trains the classifier.
classificationKNN = fitcknn(...
    predictors, ...
    response, ...
    'Distance', 'Euclidean', ...
    'Exponent', [], ...
    'NumNeighbors', 10, ...
    'DistanceWeight', 'SquaredInverse', ...
    'Standardize', true, ...
    'ClassNames', categorical(unique(responseData)));

% Create the result struct with predict function
predictorExtractionFcn = @(t) t(:, predictorNames);
knnPredictFcn = @(x) predict(classificationKNN, x);
trainedClassifier.predictFcn = @(x) knnPredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedClassifier.RequiredVariables = predictorNames;
trainedClassifier.ClassificationKNN = classificationKNN;
trainedClassifier.About = 'This struct is a trained model exported from Classification Learner R2020b.';
trainedClassifier.HowToPredict = sprintf('To make predictions on a new table, T, use: \n  yfit = c.predictFcn(T) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nThe table, T, must contain the variables returned by: \n  c.RequiredVariables \nVariable formats (e.g. matrix/vector, datatype) must match the original training data. \nAdditional variables are ignored. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
inputTable = trainingData;
predictors = inputTable(:, predictorNames);
response = responseData(:);

% Perform cross-validation
partitionedModel = crossval(trainedClassifier.ClassificationKNN, 'KFold', 50);

% Compute validation predictions
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
end

function [trainedClassifier, validationAccuracy] = BoostedTrees(trainingData, responseData, idx, ozellikler ,hmf)

inputTable = trainingData;
predictorNames = {};

for i=1:hmf
    predictorNames = [predictorNames,char(ozellikler(idx(i)))];
end
predictors = inputTable(:, predictorNames);
response = responseData(:);
isCategoricalPredictor = [];

for i=1:hmf
    isCategoricalPredictor = [isCategoricalPredictor,false];
end    
isCategoricalPredictor = logical(isCategoricalPredictor);

% Train a classifier
% This code specifies all the classifier options and trains the classifier.
template = templateTree(...
    'MaxNumSplits', 20);
classificationEnsemble = fitcensemble(...
    predictors, ...
    response, ...
    'Method', 'AdaBoostM2', ...
    'NumLearningCycles', 30, ...
    'Learners', template, ...
    'LearnRate', 0.1, ...
    'ClassNames', categorical(unique(responseData)));

% Create the result struct with predict function
predictorExtractionFcn = @(t) t(:, predictorNames);
ensemblePredictFcn = @(x) predict(classificationEnsemble, x);
trainedClassifier.predictFcn = @(x) ensemblePredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedClassifier.RequiredVariables = predictorNames;
trainedClassifier.ClassificationEnsemble = classificationEnsemble;
trainedClassifier.About = 'This struct is a trained model exported from Classification Learner R2020b.';
trainedClassifier.HowToPredict = sprintf('To make predictions on a new table, T, use: \n  yfit = c.predictFcn(T) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nThe table, T, must contain the variables returned by: \n  c.RequiredVariables \nVariable formats (e.g. matrix/vector, datatype) must match the original training data. \nAdditional variables are ignored. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
inputTable = trainingData;
predictors = inputTable(:, predictorNames);
response = responseData(:);


% Perform cross-validation
partitionedModel = crossval(trainedClassifier.ClassificationEnsemble, 'KFold', 50);

% Compute validation predictions
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
end

function [trainedClassifier, validationAccuracy] = BaggedTrees(trainingData, responseData, idx, ozellikler ,hmf)

inputTable = trainingData;
predictorNames = {};

for i=1:hmf
    predictorNames = [predictorNames,char(ozellikler(idx(i)))];
end
predictors = inputTable(:, predictorNames);
response = responseData(:);
isCategoricalPredictor = [];

for i=1:hmf
    isCategoricalPredictor = [isCategoricalPredictor,false];
end    
isCategoricalPredictor = logical(isCategoricalPredictor);

% Train a classifier
% This code specifies all the classifier options and trains the classifier.
template = templateTree(...
    'MaxNumSplits', 60);
classificationEnsemble = fitcensemble(...
    predictors, ...
    response, ...
    'Method', 'Bag', ...
    'NumLearningCycles', 30, ...
    'Learners', template, ...
    'ClassNames', categorical(unique(responseData)));

% Create the result struct with predict function
predictorExtractionFcn = @(t) t(:, predictorNames);
ensemblePredictFcn = @(x) predict(classificationEnsemble, x);
trainedClassifier.predictFcn = @(x) ensemblePredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedClassifier.RequiredVariables = predictorNames;
trainedClassifier.ClassificationEnsemble = classificationEnsemble;
trainedClassifier.About = 'This struct is a trained model exported from Classification Learner R2020b.';
trainedClassifier.HowToPredict = sprintf('To make predictions on a new table, T, use: \n  yfit = c.predictFcn(T) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nThe table, T, must contain the variables returned by: \n  c.RequiredVariables \nVariable formats (e.g. matrix/vector, datatype) must match the original training data. \nAdditional variables are ignored. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
inputTable = trainingData;
predictors = inputTable(:, predictorNames);
response = responseData(:);


% Perform cross-validation
partitionedModel = crossval(trainedClassifier.ClassificationEnsemble, 'KFold', 50);

% Compute validation predictions
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
end

function [trainedClassifier, validationAccuracy] = SubspaceDiscriminant(trainingData, responseData, idx, ozellikler ,hmf)

inputTable = trainingData;
predictorNames = {};

for i=1:hmf
    predictorNames = [predictorNames,char(ozellikler(idx(i)))];
end
predictors = inputTable(:, predictorNames);
response = responseData(:);
isCategoricalPredictor = [];

for i=1:hmf
    isCategoricalPredictor = [isCategoricalPredictor,false];
end    
isCategoricalPredictor = logical(isCategoricalPredictor);

% Train a classifier
% This code specifies all the classifier options and trains the classifier.
subspaceDimension = max(1, min(11, width(predictors) - 1));
classificationEnsemble = fitcensemble(...
    predictors, ...
    response, ...
    'Method', 'Subspace', ...
    'NumLearningCycles', 30, ...
    'Learners', 'discriminant', ...
    'NPredToSample', subspaceDimension, ...
    'ClassNames', categorical(unique(responseData)));

% Create the result struct with predict function
predictorExtractionFcn = @(t) t(:, predictorNames);
ensemblePredictFcn = @(x) predict(classificationEnsemble, x);
trainedClassifier.predictFcn = @(x) ensemblePredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedClassifier.RequiredVariables = predictorNames;
trainedClassifier.ClassificationEnsemble = classificationEnsemble;
trainedClassifier.About = 'This struct is a trained model exported from Classification Learner R2020b.';
trainedClassifier.HowToPredict = sprintf('To make predictions on a new table, T, use: \n  yfit = c.predictFcn(T) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nThe table, T, must contain the variables returned by: \n  c.RequiredVariables \nVariable formats (e.g. matrix/vector, datatype) must match the original training data. \nAdditional variables are ignored. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
inputTable = trainingData;
predictors = inputTable(:, predictorNames);
response = responseData(:);

% Perform cross-validation
partitionedModel = crossval(trainedClassifier.ClassificationEnsemble, 'KFold', 50);

% Compute validation predictions
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
end

function [trainedClassifier, validationAccuracy] = SubspaceKNN(trainingData, responseData, idx, ozellikler ,hmf)

inputTable = trainingData;
predictorNames = {};

for i=1:hmf
    predictorNames = [predictorNames,char(ozellikler(idx(i)))];
end
predictors = inputTable(:, predictorNames);
response = responseData(:);
isCategoricalPredictor = [];

for i=1:hmf
    isCategoricalPredictor = [isCategoricalPredictor,false];
end    
isCategoricalPredictor = logical(isCategoricalPredictor);

% Train a classifier
% This code specifies all the classifier options and trains the classifier.
subspaceDimension = max(1, min(11, width(predictors) - 1));
classificationEnsemble = fitcensemble(...
    predictors, ...
    response, ...
    'Method', 'Subspace', ...
    'NumLearningCycles', 30, ...
    'Learners', 'knn', ...
    'NPredToSample', subspaceDimension, ...
    'ClassNames', categorical(unique(responseData)));

% Create the result struct with predict function
predictorExtractionFcn = @(t) t(:, predictorNames);
ensemblePredictFcn = @(x) predict(classificationEnsemble, x);
trainedClassifier.predictFcn = @(x) ensemblePredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedClassifier.RequiredVariables = predictorNames;
trainedClassifier.ClassificationEnsemble = classificationEnsemble;
trainedClassifier.About = 'This struct is a trained model exported from Classification Learner R2020b.';
trainedClassifier.HowToPredict = sprintf('To make predictions on a new table, T, use: \n  yfit = c.predictFcn(T) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nThe table, T, must contain the variables returned by: \n  c.RequiredVariables \nVariable formats (e.g. matrix/vector, datatype) must match the original training data. \nAdditional variables are ignored. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
inputTable = trainingData;
predictors = inputTable(:, predictorNames);
response = responseData(:);


% Perform cross-validation
partitionedModel = crossval(trainedClassifier.ClassificationEnsemble, 'KFold', 50);

% Compute validation predictions
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
end

function [trainedClassifier, validationAccuracy] = RUSBoosterTrees(trainingData, responseData, idx, ozellikler ,hmf)

inputTable = trainingData;
predictorNames = {};

for i=1:hmf
    predictorNames = [predictorNames,char(ozellikler(idx(i)))];
end
predictors = inputTable(:, predictorNames);
response = responseData(:);
isCategoricalPredictor = [];

for i=1:hmf
    isCategoricalPredictor = [isCategoricalPredictor,false];
end    
isCategoricalPredictor = logical(isCategoricalPredictor);

% Train a classifier
% This code specifies all the classifier options and trains the classifier.
template = templateTree(...
    'MaxNumSplits', 20);
classificationEnsemble = fitcensemble(...
    predictors, ...
    response, ...
    'Method', 'RUSBoost', ...
    'NumLearningCycles', 30, ...
    'Learners', template, ...
    'LearnRate', 0.1, ...
    'ClassNames', categorical(unique(responseData)));

% Create the result struct with predict function
predictorExtractionFcn = @(t) t(:, predictorNames);
ensemblePredictFcn = @(x) predict(classificationEnsemble, x);
trainedClassifier.predictFcn = @(x) ensemblePredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedClassifier.RequiredVariables = predictorNames;
trainedClassifier.ClassificationEnsemble = classificationEnsemble;
trainedClassifier.About = 'This struct is a trained model exported from Classification Learner R2020b.';
trainedClassifier.HowToPredict = sprintf('To make predictions on a new table, T, use: \n  yfit = c.predictFcn(T) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nThe table, T, must contain the variables returned by: \n  c.RequiredVariables \nVariable formats (e.g. matrix/vector, datatype) must match the original training data. \nAdditional variables are ignored. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
inputTable = trainingData;
predictors = inputTable(:, predictorNames);
response = responseData(:);

% Perform cross-validation
partitionedModel = crossval(trainedClassifier.ClassificationEnsemble, 'KFold', 50);

% Compute validation predictions
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
end