---
title: "Exercise Prediction"
author: "James Liu"
date: "Oct 22, 2015"
output: html_document
---

This repository contain reproduction of analysis from Groupware http://groupware.les.inf.puc-rio.br/har

The purpose of exercise is to create machine learning algorithm to predict activity quality from activity monitors.

### Load data
```{r}
data <- read.csv("pml-training.csv", na.strings=c("NA",""))

# move classes to column 0
data <- data[, c(ncol(data), 1:ncol(data)-1)]
```

### Data extraction
```{r}
# removing non sensor features from data
uselessData <- grep("timestamp|X|user_name|new_window|num_window", names(data))
data <- data[,-uselessData]

# removing columns with any NA data
nonNAFeatures <- colSums(is.na(data))==0;
data <- data[, nonNAFeatures];
```

### Data preparation

```{r}
library(caret)
inTrain <- createDataPartition(y = data[,1], p = 0.6, list=FALSE)
trainingData <- data[inTrain,]

others <- data[-inTrain,]
inCrossValidation <- createDataPartition(y = others[,1],  p =  0.5, list=FALSE)
  
crossValidationData <- others[inCrossValidation,]
testData<- others[-inCrossValidation,]
  
```

```{r, eval=FALSE }
preproc <- preProcess(trainingData[,-1], method='pca', thresh=0.9)
trainingData.pca <- predict(preproc, trainingData[,-1])     
crossValidationData.pca <- predict(preproc, crossValidationData[,-1])    
testData.pca <- predict(preproc, testData[,-1])  
```

### Training
Use 4 different optimization models: SVM, GBM, random forest and LDA.
Use trainingData optimized with pca.

```{r, eval=FALSE }
library(kernlab)	
# Train SVM
modelFitSVM <- train(trainingData$classe ~., data=trainingData.pca, method='svmRadial')
	
library(gbm)
# Train GBM
modelFitGBM <- train(trainingData$classe ~., data=trainingData.pca, method="gbm")

# Train Random Forest
trainControl <- trainControl(method = "cv", number = 10)
modelFitRF <- train(trainingData$classe ~., data=trainingData.pca, method='rf', trControl = trainControl)
    	
# Train LDA
modelFitLDA <- train(trainingData$classe ~., data=trainingData.pca, method="lda")
```

```{r, eval=FALSE }
predictionSVM <- predict(modelFitSVM,  crossValidationData.pca)
predictionGBM <- predict(modelFitGBM,  crossValidationData.pca)
predictionRF <- predict(modelFitRF,  crossValidationData.pca)
predictionLDA <- predict(modelFitLDA,  crossValidationData.pca)
```

Algorithm accuracy for different models with cross validation:

```{r, eval=FALSE }
confusionMatrix(predictionRF, crossValidationData$classe)$overall["Accuracy"]

confusionMatrix(predictionSVM, crossValidationData$classe)$overall["Accuracy"]

confusionMatrix(predictionGBM, crossValidationData$classe)$overall["Accuracy"]

confusionMatrix(predictionLDA, crossValidationData$classe)$overall["Accuracy"]

```

### Assemble models

dataCombined <- data.frame(predictionSVM,predictionGBM,predictionLDA, predictionRF, classe=crossValidationData$classe)
modelFitCombined <- train(classe ~.,method="rf", data=dataCombined)

predictionCombined <- predict(modelFitCombined,  dataCombined)
confusionMatrix(predictionCombined, dataCombined$classe)$overall["Accuracy"]

### Check final model on test set
testDataCombined <- data.frame(predict(modelFitRF,  testData.pca),predict(modelFitSVM,  testData.pca),predict(modelFitGBM,  testData.pca),predict(modelFitLDA,  testData.pca))

testPrediction <- predict(modelFitCombined, testDataCombined)
confusionMatrix(testPrediction, testData$classe)$overall["Accuracy"]

### Results
Accuracy results: Random forest gives us the best result with 0.9655876 accuracy out of sample error with cross-validation

> confusionMatrix(predictionRF, crossValidationData$classe)$overall["Accuracy"]
 Accuracy 
0.9655876 
> confusionMatrix(predictionSVM, crossValidationData$classe)$overall["Accuracy"]
 Accuracy 
0.8761152 
> confusionMatrix(predictionGBM, crossValidationData$classe)$overall["Accuracy"]
 Accuracy 
0.8039765 
> confusionMatrix(predictionLDA, crossValidationData$classe)$overall["Accuracy"]
 Accuracy 
0.5108335 
> confusionMatrix(predictionCombined, dataCombined$classe)$overall["Accuracy"]
 Accuracy 
0.9650777
 
### Test cases 

```{r, eval=FALSE }
submissionData <- read.csv("pml-testing.csv", na.strings=c("NA",""));
submissionData <- submissionData[, c(ncol(submissionData), 1:ncol(submissionData)-1)]
submissionData <- submissionData[, -uselessData]
submissionData <- submissionData[, nonNAFeatures]
submissionData.pca <- predict(preproc, submissionData[,-1]) 
predictionForSubmission <- predict(modelFitRF, submissionData.pca)
answers = predictionForSubmission
pml_write_files(answers)
```

