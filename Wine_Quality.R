#Hiren Jagani (A20432602)
#Parth Kaushik (A20417447)
#Yasavi Gude (A20432513)


#importing data 
white_wine= read.table("white_wine.csv",header = TRUE, sep=',')
head(white_wine)

as.null(white_wine)
summary(white_wine)


white_wine_cor <- as.matrix(subset(white_wine))
response <- white_wine$quality
corrplot(cor(white_wine_cor), method = c("number"))

z.test(white_wine$volatile.acidity, NULL, alternative = "two.sided", mu = 0, sigma.x = 0.5, sigma.y = 0.5, conf.level = 0.95)
z.test(white_wine$alcohol, NULL, alternative = "two.sided", mu = 0, sigma.x = 0.5, sigma.y = 0.5, conf.level = 0.95)


white_wine$quality<-factor(white_wine$quality, ordered = T)

white_wine$rating <- ifelse(white_wine$quality<=5,'bad','good')

white_wine$rating<-ordered(white_wine$rating, levels=c('bad','good'))


#train and test the data
train.data <- white_wine[1:3918,]
test.data <- white_wine[3918:4898,]

# Build X_train, y_train, X_test, y_test

basemodel=glm(rating~citric.acid, data = train.data,family = binomial())
summary(basemodel)


fullmodel=glm(rating~fixed.acidity+volatile.acidity+citric.acid+residual.sugar+chlorides+free.sulfur.dioxide+total.sulfur.dioxide+density+pH+sulphates+alcohol,data=train.data,family = binomial())
summary(fullmodel)


forwardmodel = step(basemodel,scope=list(upper = fullmodel, lower=~1),direction = "forward", trace = FALSE)

summary(forwardmodel)


backwardmodel = step (fullmodel,scope=list(upper = fullmodel, lower=~1),direction = "backward", trace = FALSE)
summary(backwardmodel)

bestsubsetmodel=step(basemodel,scope=list(upper = fullmodel, lower=~1),direction = "both", trace = FALSE)
summary(bestsubsetmodel)
'''
#nullmod
nullmod<-glm(white_wine$rating~1,family="binomial")
1-logLik(forwardmodel)/logLik(nullmod)
1-logLik(backwardmodel)/logLik(nullmod)
1-logLik(bestsubsetmodel)/logLik(nullmod)
'''
exp(coef(bestsubsetmodel))
exp(coef(forwardmodel))
exp(coef(backwardmodel))

best.subset.train.model=predict(bestsubsetmodel,type="response",newdata = train.data)
best.subset.train.model


write.csv(data.frame(predict(bestsubsetmodel,type = "response",newdata = train.data)),"bestsubset_train.csv")
best.subset.train.output<-cbind(train.data,best.subset.train.model)
write.csv(best.subset.train.output,"trainbestsubset.csv")

forward.train.model=predict(forwardmodel,type="response",newdata = train.data)
forward.train.model

write.csv(data.frame(predict(forwardmodel,type = "response",newdata = train.data)),"forward_train.csv")
forward.train.output<-cbind(train.data,forward.train.model)
write.csv(forward.train.output,"trainforward.csv")

backward.train.model=predict(backwardmodel,type="response",newdata = train.data)
backward.train.model

write.csv(data.frame(predict(backwardmodel,type = "response",newdata = train.data)),"backward_train.csv")
backward.train.output<-cbind(train.data,backward.train.model)
write.csv(backward.train.output,"trainbackward.csv")






best.subset.test.model.predict = predict (bestsubsetmodel,type="response",newdata = test.data)
best.subset.test.model.predict


accuracy_1<-table(test.data$rating,best.subset.test.model.predict>0.6)
accuracy_1
sum(diag(accuracy_1))/sum(accuracy_1)

accuracy_2<-table(test.data$rating,best.subset.test.model.predict>0.7)
accuracy_2
sum(diag(accuracy_2))/sum(accuracy_2)

accuracy_3<-table(test.data$rating,best.subset.test.model.predict>0.8)
accuracy_3
sum(diag(accuracy_3))/sum(accuracy_3)

accuracy_4<-table(test.data$rating,best.subset.test.model.predict>0.9)
accuracy_4
sum(diag(accuracy_4))/sum(accuracy_4)







forward.test.model.predict = predict (forwardmodel,type="response",newdata = test.data)
forward.test.model.predict

accuracyf_1<-table(test.data$rating,forward.test.model.predict>0.6)
accuracyf_1
sum(diag(accuracyf_1))/sum(accuracyf_1)

accuracyf_2<-table(test.data$rating,forward.test.model.predict>0.7)
accuracyf_2
sum(diag(accuracyf_2))/sum(accuracyf_2)

accuracyf_3<-table(test.data$rating,forward.test.model.predict>0.8)
accuracyf_3
sum(diag(accuracyf_3))/sum(accuracyf_3)

accuracyf_4<-table(test.data$rating,forward.test.model.predict>0.9)
accuracyf_4
sum(diag(accuracyf_4))/sum(accuracyf_4)









backward.test.model.predict = predict (backwardmodel,type="response",newdata = test.data)
backward.test.model.predict


accuracyb_1<-table(test.data$rating,backward.test.model.predict>0.6)
accuracyb_1
sum(diag(accuracyb_1))/sum(accuracyb_1)

accuracyb_2<-table(test.data$rating,backward.test.model.predict>0.7)
accuracyb_2
sum(diag(accuracyb_2))/sum(accuracyb_2)

accuracyb_3<-table(test.data$rating,backward.test.model.predict>0.8)
accuracyb_3
sum(diag(accuracyb_3))/sum(accuracyb_3)

accuracyb_4<-table(test.data$rating,backward.test.model.predict>0.9)
accuracyb_4
sum(diag(accuracyb_4))/sum(accuracyb_4)


forward_cook <- round(cooks.distance(forwardmodel),5)
plot(forward_cook)

backward_cook <- round(cooks.distance(backwardmodel),5)
plot(backward_cook)

bestsubsetmodel_cook <- round(cooks.distance(bestsubsetmodel),5)
plot(bestsubsetmodel_cook)

#Naive bayes classification
x<-white_wine[,1:11]
y<-white_wine[,13]
library(caret)
naive_bayes_model=train(x,y,'nb',trControl=trainControl(method = 'cv',number = 10),na.action=na.pass)
naive_bayes_model

#n fold for logistic regression
library(glmnet)
library(boot)

error_fm_train = cv.glm(train.data, forwardmodel, K = 10)$delta
accuracy_fm_train = 1 - error_fm_train
accuracy_fm_train

error_bm_train = cv.glm(train.data, backwardmodel, K = 10)$delta
accuracy_bm_trian = 1 - error_bm_train
accuracy_bm_trian

error_bsm_train = cv.glm(train.data, bestsubsetmodel, K = 10)$delta
accuracy_bsm_train = 1 - error_bsm_train
accuracy_bsm_train



#test model accuries
error_fm_test = cv.glm(test.data, forwardmodel, K = 10)$delta
accuracy_fm_test = 1 - error_fm_test
accuracy_fm_test

error_bm_test = cv.glm(test.data, backwardmodel, K = 10)$delta
accuracy_bm_test = 1 - error_bm_test
accuracy_bm_test

error_bsm_test = cv.glm(test.data, bestsubsetmodel, K = 10)$delta
accuracy_bsm_test = 1 - error_bsm_test
accuracy_bsm_test


#knn
numeric.variables.data<-sapply(white_wine,is.numeric)
numeric.variables.data
knn_white_wine_data=white_wine
knn_white_wine_data[numeric.variables.data]=lapply(knn_white_wine_data[numeric.variables.data],scale)
knn_white_wine_data
knn_x=white_wine[,1:11]
knn_y=white_wine[,13]
knn_model=train(x,y,'knn',trControl = trainControl(method = 'cv',number = 10),tuneGrid = expand.grid(k=1:20))
print(knn_model)
