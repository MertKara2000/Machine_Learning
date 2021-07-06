dataset = readtable('pulmoner_data.csv');
dataset = sortrows(dataset,'Hastalik','descend');
radiomics = dataset(:,3:109);
subtip = dataset(:,2);
subtip = table2array(subtip);
subtip = categorical(subtip);

for i =289:-1:129
    radiomics(i,:) = [];
    subtip(i,:) = [];
end
[n_of_rows,n_of_col] = size(subtip)
for x = 1:n_of_rows;
    type = subtip(x);
    if type == "PF"
        subtip(x) = "1"
    elseif type == "No PF"
        subtip(x) = "2"
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
            ens = fitcensemble(X,y,'Method','AdaBoostM1','Learners',t);
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

Table3 = [];
for j=1:7
    %feature selection buraya yazılacak
    if j==1
        %chi2 
        [idx,scores] = fscchi2(X,y);
        find(isinf(scores));
        bar(scores(idx))
        xlabel('Predictor rank')
        ylabel('Predictor importance score')
        idx(1:5)
        
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
        ens = fitcensemble(X,y,'Method','AdaBoostM1','Learners',t);
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
        
    for i =1:10
        hmf =i;
        TrainData = [];
        TrainData = table;
        %eğitim verisi hazırlanıyor.
        for i=1:hmf
            temp = ozellikler(idx(i));
            TrainData = [TrainData,temp];
            
        end
    end
    Table3 = [Table3;TrainData];
end

%Makine ogrenmesi modelleri
function [trainedClassifier, validationAccuracy] = FineTree(trainingData, responseData, idx, ozellikler ,hmf)

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

classificationTree = fitctree(...
    predictors, ...
    response, ...
    'SplitCriterion', 'gdi', ...
    'MaxNumSplits', 100, ...
    'Surrogate', 'off', ...
    'ClassNames', categorical(unique(responseData)));

predictorExtractionFcn = @(t) t(:, predictorNames);
treePredictFcn = @(x) predict(classificationTree, x);
trainedClassifier.predictFcn = @(x) treePredictFcn(predictorExtractionFcn(x));

trainedClassifier.ClassificationTree = classificationTree;

inputTable = trainingData;
predictors = inputTable(:, predictorNames);
response = responseData(:);

partitionedModel = crossval(trainedClassifier.ClassificationTree, 'KFold', 50);

[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
end

function [trainedClassifier, validationAccuracy] = MediumTree(trainingData, responseData, idx, ozellikler ,hmf)

predictorNames = {};
 
inputTable = trainingData;
 
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

classificationTree = fitctree(...
    predictors, ...
    response, ...
    'SplitCriterion', 'gdi', ...
    'MaxNumSplits', 20, ...
    'Surrogate', 'off', ...
    'ClassNames', categorical(unique(responseData)));

predictorExtractionFcn = @(t) t(:, predictorNames);
treePredictFcn = @(x) predict(classificationTree, x);
trainedClassifier.predictFcn = @(x) treePredictFcn(predictorExtractionFcn(x));

trainedClassifier.ClassificationTree = classificationTree;

inputTable = trainingData;
predictors = inputTable(:, predictorNames);
response = responseData(:);

partitionedModel = crossval(trainedClassifier.ClassificationTree, 'KFold', 50);

[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

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

classificationTree = fitctree(...
    predictors, ...
    response, ...
    'SplitCriterion', 'gdi', ...
    'MaxNumSplits', 4, ...
    'Surrogate', 'off', ...
    'ClassNames', categorical(unique(responseData)));

predictorExtractionFcn = @(t) t(:, predictorNames);
treePredictFcn = @(x) predict(classificationTree, x);
trainedClassifier.predictFcn = @(x) treePredictFcn(predictorExtractionFcn(x));

trainedClassifier.ClassificationTree = classificationTree;

inputTable = trainingData;
predictors = inputTable(:, predictorNames);
response = responseData(:);

partitionedModel = crossval(trainedClassifier.ClassificationTree, 'KFold', 50);

[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');

end

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

classificationDiscriminant = fitcdiscr(...
    predictors, ...
    response, ...
    'DiscrimType', 'linear', ...
    'Gamma', 0, ...
    'FillCoeffs', 'off', ...
    'ClassNames', categorical(unique(responseData)));   
 
predictorExtractionFcn = @(t) t(:, predictorNames);
discriminantPredictFcn = @(x) predict(classificationDiscriminant, x);
trainedClassifier.predictFcn = @(x) discriminantPredictFcn(predictorExtractionFcn(x));

trainedClassifier.ClassificationDiscriminant = classificationDiscriminant;

inputTable = trainingData;
predictors = inputTable(:, predictorNames);
response = responseData(:);
 
partitionedModel = crossval(trainedClassifier.ClassificationDiscriminant, 'KFold', 50);

 
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

 
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

classificationDiscriminant = fitcdiscr(...
    predictors, ...
    response, ...
    'DiscrimType', 'diagquadratic', ...
    'FillCoeffs', 'off', ...
    'ClassNames', categorical(unique(responseData)));

predictorExtractionFcn = @(t) t(:, predictorNames);
discriminantPredictFcn = @(x) predict(classificationDiscriminant, x);
trainedClassifier.predictFcn = @(x) discriminantPredictFcn(predictorExtractionFcn(x));

 
trainedClassifier.ClassificationDiscriminant = classificationDiscriminant;
 
inputTable = trainingData;
predictors = inputTable(:, predictorNames);
response = responseData(:);

partitionedModel = crossval(trainedClassifier.ClassificationDiscriminant, 'KFold', 50);

[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

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

successClass = '0';
failureClass = '1';

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

concatenatedPredictorsAndResponse = [predictors, table(zeroOneResponse)];

GeneralizedLinearModel = fitglm(...
    concatenatedPredictorsAndResponse, ...
    'Distribution', 'binomial', ...
    'link', 'logit');

convertSuccessProbsToPredictions = @(p) successFailureAndMissingClasses( ~isnan(p).*( (p<0.5) + 1 ) + isnan(p)*3 );
returnMultipleValuesFcn = @(varargin) varargin{1:max(1,nargout)};
scoresFcn = @(p) [p, 1-p];
predictionsAndScoresFcn = @(p) returnMultipleValuesFcn( convertSuccessProbsToPredictions(p), scoresFcn(p) );

predictorExtractionFcn = @(t) t(:, predictorNames);
logisticRegressionPredictFcn = @(x) predictionsAndScoresFcn( predict(GeneralizedLinearModel, x) );
trainedClassifier.predictFcn = @(x) logisticRegressionPredictFcn(predictorExtractionFcn(x));

trainedClassifier.GeneralizedLinearModel = GeneralizedLinearModel;
trainedClassifier.SuccessClass = successClass;
trainedClassifier.FailureClass = failureClass;
trainedClassifier.MissingClass = missingClass;
trainedClassifier.ClassNames = {successClass; failureClass};

inputTable = trainingData;
predictors = inputTable(:, predictorNames);
response = responseData(:);

KFolds = 50;
cvp = cvpartition(response, 'KFold', KFolds);

validationPredictions = response;
numObservations = size(predictors, 1);
numClasses = 2;
validationScores = NaN(numObservations, numClasses);
for fold = 1:KFolds
    trainingPredictors = predictors(cvp.training(fold), :);
    trainingResponse = response(cvp.training(fold), :);
    foldIsCategoricalPredictor = isCategoricalPredictor;
    
    successClass = '0';
    failureClass = '1';
    
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
    
    concatenatedPredictorsAndResponse = [trainingPredictors, table(zeroOneResponse)];
    
    GeneralizedLinearModel = fitglm(...
        concatenatedPredictorsAndResponse, ...
        'Distribution', 'binomial', ...
        'link', 'logit');

    convertSuccessProbsToPredictions = @(p) successFailureAndMissingClasses( ~isnan(p).*( (p<0.5) + 1 ) + isnan(p)*3 );
    returnMultipleValuesFcn = @(varargin) varargin{1:max(1,nargout)};
    scoresFcn = @(p) [p, 1-p];
    predictionsAndScoresFcn = @(p) returnMultipleValuesFcn( convertSuccessProbsToPredictions(p), scoresFcn(p) );

    logisticRegressionPredictFcn = @(x) predictionsAndScoresFcn( predict(GeneralizedLinearModel, x) );
    validationPredictFcn = @(x) logisticRegressionPredictFcn(x);
 
    validationPredictors = predictors(cvp.test(fold), :);
    [foldPredictions, foldScores] = validationPredictFcn(validationPredictors);
    
    validationPredictions(cvp.test(fold), :) = foldPredictions;
    validationScores(cvp.test(fold), :) = foldScores;
end

 
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

predictorExtractionFcn = @(t) t(:, predictorNames);
naiveBayesPredictFcn = @(x) predict(classificationNaiveBayes, x);
trainedClassifier.predictFcn = @(x) naiveBayesPredictFcn(predictorExtractionFcn(x));

trainedClassifier.ClassificationNaiveBayes = classificationNaiveBayes;
 
inputTable = trainingData;
predictors = inputTable(:, predictorNames);
response = responseData(:);
 
partitionedModel = crossval(trainedClassifier.ClassificationNaiveBayes, 'KFold', 50);
 
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

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

 
predictorExtractionFcn = @(t) t(:, predictorNames);
naiveBayesPredictFcn = @(x) predict(classificationNaiveBayes, x);
trainedClassifier.predictFcn = @(x) naiveBayesPredictFcn(predictorExtractionFcn(x));

 
trainedClassifier.ClassificationNaiveBayes = classificationNaiveBayes;


inputTable = trainingData;
predictors = inputTable(:, predictorNames);
response = responseData(:);

 
partitionedModel = crossval(trainedClassifier.ClassificationNaiveBayes, 'KFold', 50);

[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

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

 
 
classificationSVM = fitcsvm(...
    predictors, ...
    response, ...
    'KernelFunction', 'linear', ...
    'PolynomialOrder', [], ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true, ...
    'ClassNames', categorical(unique(responseData)));

 
predictorExtractionFcn = @(t) t(:, predictorNames);
svmPredictFcn = @(x) predict(classificationSVM, x);
trainedClassifier.predictFcn = @(x) svmPredictFcn(predictorExtractionFcn(x));

trainedClassifier.ClassificationSVM = classificationSVM;

inputTable = trainingData;
predictors = inputTable(:, predictorNames);
response = responseData(:);

partitionedModel = crossval(trainedClassifier.ClassificationSVM, 'KFold', 50);

[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

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

classificationSVM = fitcsvm(...
    predictors, ...
    response, ...
    'KernelFunction', 'polynomial', ...
    'PolynomialOrder', 2, ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true, ...
    'ClassNames', categorical(unique(responseData)));

predictorExtractionFcn = @(t) t(:, predictorNames);
svmPredictFcn = @(x) predict(classificationSVM, x);
trainedClassifier.predictFcn = @(x) svmPredictFcn(predictorExtractionFcn(x));

trainedClassifier.ClassificationSVM = classificationSVM;

inputTable = trainingData;
predictors = inputTable(:, predictorNames);
response = responseData(:);

partitionedModel = crossval(trainedClassifier.ClassificationSVM, 'KFold', 50);

[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

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

classificationSVM = fitcsvm(...
    predictors, ...
    response, ...
    'KernelFunction', 'polynomial', ...
    'PolynomialOrder', 3, ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true, ...
    'ClassNames', categorical(unique(responseData)));

predictorExtractionFcn = @(t) t(:, predictorNames);
svmPredictFcn = @(x) predict(classificationSVM, x);
trainedClassifier.predictFcn = @(x) svmPredictFcn(predictorExtractionFcn(x));

trainedClassifier.ClassificationSVM = classificationSVM;

inputTable = trainingData;
predictors = inputTable(:, predictorNames);
response = responseData(:);

partitionedModel = crossval(trainedClassifier.ClassificationSVM, 'KFold', 50);

[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

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

classificationSVM = fitcsvm(...
    predictors, ...
    response, ...
    'KernelFunction', 'gaussian', ...
    'PolynomialOrder', [], ...
    'KernelScale', 1.1, ...
    'BoxConstraint', 1, ...
    'Standardize', true, ...
    'ClassNames', categorical(unique(responseData)));

predictorExtractionFcn = @(t) t(:, predictorNames);
svmPredictFcn = @(x) predict(classificationSVM, x);
trainedClassifier.predictFcn = @(x) svmPredictFcn(predictorExtractionFcn(x));

trainedClassifier.ClassificationSVM = classificationSVM;

inputTable = trainingData;
predictors = inputTable(:, predictorNames);
response = responseData(:);

partitionedModel = crossval(trainedClassifier.ClassificationSVM, 'KFold', 50);

[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

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

classificationSVM = fitcsvm(...
    predictors, ...
    response, ...
    'KernelFunction', 'gaussian', ...
    'PolynomialOrder', [], ...
    'KernelScale', 4.6, ...
    'BoxConstraint', 1, ...
    'Standardize', true, ...
    'ClassNames', categorical(unique(responseData)));

predictorExtractionFcn = @(t) t(:, predictorNames);
svmPredictFcn = @(x) predict(classificationSVM, x);
trainedClassifier.predictFcn = @(x) svmPredictFcn(predictorExtractionFcn(x));

trainedClassifier.ClassificationSVM = classificationSVM;

inputTable = trainingData;
predictors = inputTable(:, predictorNames);
response = responseData(:);

partitionedModel = crossval(trainedClassifier.ClassificationSVM, 'KFold', 50);

[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

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

classificationSVM = fitcsvm(...
    predictors, ...
    response, ...
    'KernelFunction', 'gaussian', ...
    'PolynomialOrder', [], ...
    'KernelScale', 18, ...
    'BoxConstraint', 1, ...
    'Standardize', true, ...
    'ClassNames', categorical(unique(responseData)));

predictorExtractionFcn = @(t) t(:, predictorNames);
svmPredictFcn = @(x) predict(classificationSVM, x);
trainedClassifier.predictFcn = @(x) svmPredictFcn(predictorExtractionFcn(x));

trainedClassifier.ClassificationSVM = classificationSVM;

inputTable = trainingData;
predictors = inputTable(:, predictorNames);
response = responseData(:);

partitionedModel = crossval(trainedClassifier.ClassificationSVM, 'KFold', 50);

[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

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

classificationKNN = fitcknn(...
    predictors, ...
    response, ...
    'Distance', 'Euclidean', ...
    'Exponent', [], ...
    'NumNeighbors', 1, ...
    'DistanceWeight', 'Equal', ...
    'Standardize', true, ...
    'ClassNames', categorical(unique(responseData)));

predictorExtractionFcn = @(t) t(:, predictorNames);
knnPredictFcn = @(x) predict(classificationKNN, x);
trainedClassifier.predictFcn = @(x) knnPredictFcn(predictorExtractionFcn(x));

trainedClassifier.ClassificationKNN = classificationKNN;

inputTable = trainingData;
predictors = inputTable(:, predictorNames);
response = responseData(:);

partitionedModel = crossval(trainedClassifier.ClassificationKNN, 'KFold', 50);

[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

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

classificationKNN = fitcknn(...
    predictors, ...
    response, ...
    'Distance', 'Euclidean', ...
    'Exponent', [], ...
    'NumNeighbors', 10, ...
    'DistanceWeight', 'Equal', ...
    'Standardize', true, ...
    'ClassNames', categorical(unique(responseData)));

predictorExtractionFcn = @(t) t(:, predictorNames);
knnPredictFcn = @(x) predict(classificationKNN, x);
trainedClassifier.predictFcn = @(x) knnPredictFcn(predictorExtractionFcn(x));

trainedClassifier.ClassificationKNN = classificationKNN;

inputTable = trainingData;
predictors = inputTable(:, predictorNames);
response = responseData(:);

partitionedModel = crossval(trainedClassifier.ClassificationKNN, 'KFold', 50);

[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

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

classificationKNN = fitcknn(...
    predictors, ...
    response, ...
    'Distance', 'Euclidean', ...
    'Exponent', [], ...
    'NumNeighbors', 100, ...
    'DistanceWeight', 'Equal', ...
    'Standardize', true, ...
    'ClassNames', categorical(unique(responseData)));

predictorExtractionFcn = @(t) t(:, predictorNames);
knnPredictFcn = @(x) predict(classificationKNN, x);
trainedClassifier.predictFcn = @(x) knnPredictFcn(predictorExtractionFcn(x));

trainedClassifier.ClassificationKNN = classificationKNN;

inputTable = trainingData;
predictors = inputTable(:, predictorNames);
response = responseData(:);

partitionedModel = crossval(trainedClassifier.ClassificationKNN, 'KFold', 50);

[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

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

classificationKNN = fitcknn(...
    predictors, ...
    response, ...
    'Distance', 'Cosine', ...
    'Exponent', [], ...
    'NumNeighbors', 10, ...
    'DistanceWeight', 'Equal', ...
    'Standardize', true, ...
    'ClassNames', categorical(unique(responseData)));

predictorExtractionFcn = @(t) t(:, predictorNames);
knnPredictFcn = @(x) predict(classificationKNN, x);
trainedClassifier.predictFcn = @(x) knnPredictFcn(predictorExtractionFcn(x));

trainedClassifier.ClassificationKNN = classificationKNN;

inputTable = trainingData;
predictors = inputTable(:, predictorNames);
response = responseData(:);

partitionedModel = crossval(trainedClassifier.ClassificationKNN, 'KFold', 50);

[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

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

classificationKNN = fitcknn(...
    predictors, ...
    response, ...
    'Distance', 'Minkowski', ...
    'Exponent', 3, ...
    'NumNeighbors', 10, ...
    'DistanceWeight', 'Equal', ...
    'Standardize', true, ...
    'ClassNames', categorical(unique(responseData)));

predictorExtractionFcn = @(t) t(:, predictorNames);
knnPredictFcn = @(x) predict(classificationKNN, x);
trainedClassifier.predictFcn = @(x) knnPredictFcn(predictorExtractionFcn(x));

trainedClassifier.ClassificationKNN = classificationKNN;

inputTable = trainingData;
predictors = inputTable(:, predictorNames);
response = responseData(:);

partitionedModel = crossval(trainedClassifier.ClassificationKNN, 'KFold', 50);

[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

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

classificationKNN = fitcknn(...
    predictors, ...
    response, ...
    'Distance', 'Euclidean', ...
    'Exponent', [], ...
    'NumNeighbors', 10, ...
    'DistanceWeight', 'SquaredInverse', ...
    'Standardize', true, ...
    'ClassNames', categorical(unique(responseData)));

predictorExtractionFcn = @(t) t(:, predictorNames);
knnPredictFcn = @(x) predict(classificationKNN, x);
trainedClassifier.predictFcn = @(x) knnPredictFcn(predictorExtractionFcn(x));

trainedClassifier.ClassificationKNN = classificationKNN;

inputTable = trainingData;
predictors = inputTable(:, predictorNames);
response = responseData(:);

partitionedModel = crossval(trainedClassifier.ClassificationKNN, 'KFold', 50);

[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

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

template = templateTree(...
    'MaxNumSplits', 20);
classificationEnsemble = fitcensemble(...
    predictors, ...
    response, ...
    'Method', 'AdaBoostM1', ...
    'NumLearningCycles', 30, ...
    'Learners', template, ...
    'LearnRate', 0.1, ...
    'ClassNames', categorical(unique(responseData)));

predictorExtractionFcn = @(t) t(:, predictorNames);
ensemblePredictFcn = @(x) predict(classificationEnsemble, x);
trainedClassifier.predictFcn = @(x) ensemblePredictFcn(predictorExtractionFcn(x));

trainedClassifier.ClassificationEnsemble = classificationEnsemble;

inputTable = trainingData;
predictors = inputTable(:, predictorNames);
response = responseData(:);

partitionedModel = crossval(trainedClassifier.ClassificationEnsemble, 'KFold', 50);

[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

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

template = templateTree(...
    'MaxNumSplits', 60);
classificationEnsemble = fitcensemble(...
    predictors, ...
    response, ...
    'Method', 'Bag', ...
    'NumLearningCycles', 30, ...
    'Learners', template, ...
    'ClassNames', categorical(unique(responseData)));

predictorExtractionFcn = @(t) t(:, predictorNames);
ensemblePredictFcn = @(x) predict(classificationEnsemble, x);
trainedClassifier.predictFcn = @(x) ensemblePredictFcn(predictorExtractionFcn(x));

trainedClassifier.ClassificationEnsemble = classificationEnsemble;

inputTable = trainingData;
predictors = inputTable(:, predictorNames);
response = responseData(:);

partitionedModel = crossval(trainedClassifier.ClassificationEnsemble, 'KFold', 50);

[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

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

subspaceDimension = max(1, min(11, width(predictors) - 1));
classificationEnsemble = fitcensemble(...
    predictors, ...
    response, ...
    'Method', 'Subspace', ...
    'NumLearningCycles', 30, ...
    'Learners', 'discriminant', ...
    'NPredToSample', subspaceDimension, ...
    'ClassNames', categorical(unique(responseData)));

predictorExtractionFcn = @(t) t(:, predictorNames);
ensemblePredictFcn = @(x) predict(classificationEnsemble, x);
trainedClassifier.predictFcn = @(x) ensemblePredictFcn(predictorExtractionFcn(x));

trainedClassifier.ClassificationEnsemble = classificationEnsemble;

inputTable = trainingData;
predictors = inputTable(:, predictorNames);
response = responseData(:);

partitionedModel = crossval(trainedClassifier.ClassificationEnsemble, 'KFold', 50);

[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

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

subspaceDimension = max(1, min(11, width(predictors) - 1));
classificationEnsemble = fitcensemble(...
    predictors, ...
    response, ...
    'Method', 'Subspace', ...
    'NumLearningCycles', 30, ...
    'Learners', 'knn', ...
    'NPredToSample', subspaceDimension, ...
    'ClassNames', categorical(unique(responseData)));

predictorExtractionFcn = @(t) t(:, predictorNames);
ensemblePredictFcn = @(x) predict(classificationEnsemble, x);
trainedClassifier.predictFcn = @(x) ensemblePredictFcn(predictorExtractionFcn(x));

trainedClassifier.ClassificationEnsemble = classificationEnsemble;
 
inputTable = trainingData;
predictors = inputTable(:, predictorNames);
response = responseData(:);
 
partitionedModel = crossval(trainedClassifier.ClassificationEnsemble, 'KFold', 50);

[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

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

predictorExtractionFcn = @(t) t(:, predictorNames);
ensemblePredictFcn = @(x) predict(classificationEnsemble, x);
trainedClassifier.predictFcn = @(x) ensemblePredictFcn(predictorExtractionFcn(x));

trainedClassifier.ClassificationEnsemble = classificationEnsemble;

inputTable = trainingData;
predictors = inputTable(:, predictorNames);
response = responseData(:);

partitionedModel = crossval(trainedClassifier.ClassificationEnsemble, 'KFold', 50);

[validationPredictions, validationScores] = kfoldPredict(partitionedModel);
 
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
end