#------------------Step 1: Importing library-------------------------------------------------
  library(ggplot2)
  library(dplyr)
  library(plyr)
  library(mlr)
  library(caret)
  library(ROCR)
  library(parallel)
  library(parallelMap) 
  library(dummies)
  library(corrplot)
  library(randomForest)
  library(FSelector)
  library(xgboost)


#-----------------Step 2: Read CSV file and prepare data quality report---------------------
  #Read CSV file
  setwd("W:\\Kaggle compettion\\Bigmart sales")
  Train_df <- read.csv("Train.csv",header=T,as.is =T,na.strings ="")
  Test_df  <- read.csv("Test.csv",header=T,as.is =T,na.strings ="")
  Test_df$Item_Outlet_Sales <-0
  
  SalesRecords<-rbind(Train_df,Test_df)
  View(SalesRecords)


  #Data report generation
  summary(SalesRecords)
  Varible <-1
  Maximum  <-1
  Minimum <-1
  Data_type <-1
  StandardDeviation<-1
  Uniques <-1
  
  list <-names(SalesRecords)
  for(i in 1:length(list))
  {
    Column <- SalesRecords[,list[i]]
    Varible[i]<-list[i]
    Data_type[i] <- class(Column)
    Maximum[i] <-ifelse( Data_type[i]=="integer"| Data_type[i]=="numeric",max(Column,na.rm = T),0)
    Minimum[i] <-ifelse( Data_type[i]=="integer"| Data_type[i]=="numeric",min(Column,na.rm = T),0)
    StandardDeviation[i] <- ifelse( Data_type[i]=="integer"| Data_type[i]=="numeric",sd(Column,na.rm = T),0)
    Uniques[i]<- ifelse( Data_type[i]=="character",nlevels(as.factor(Column)),0)
  
    
  }
 
  missing_count <-colSums(is.na(SalesRecords))
 
  Summary_report <-data.frame(
                               Varible =Varible,
                               Data_type=Data_type,
                               Maximum=Maximum,
                               Minimum=Minimum,
                               StandardDeviation=StandardDeviation,
                               Unique=Uniques,
                               missing_count=missing_count
                               
                              )
  
  write.csv(Summary_report,file="DataReport.csv",row.names =F)
  
  
  
#-----------------------------Step 3:Missing Value Treatment--------------------------------------#
  
   # Treatment for Item_Weight
  SalesRecords[which(is.na(SalesRecords$Item_Weight)),"Item_Weight"] <-mean(SalesRecords$Item_Weight,na.rm = T)
  
   # Treatment for Outlet_Size
  SalesRecords[which(is.na(SalesRecords$Outlet_Size)),"Outlet_Size"] <- "Missing"

  
#--------------------------Step 4: Outlier Treatment----------------------------------------------#
   # Outlier treatment for only integer and nummeric variable
   # here 1.Item_Weight  2.Item_MRP  3.Item_Outlet_Sales
  dim(Train_df)
  #Item_Weight 
   boxplot(SalesRecords[,"Item_Weight"],main="Item_Weight")
    # No outlier , No operation needed
   
   #Item_MRP 
   boxplot(SalesRecords[,"Item_MRP"],main="Item_MRP")
    # No outlier , No operation needed
   
   #Item_Outlet_Sales 
    boxplot(SalesRecords[1:8523,"Item_Outlet_Sales"],main="Item_Outlet_Sales")
    
    x<- boxplot(SalesRecords[1:8523,"Item_Outlet_Sales"],main="Item_Outlet_Sales")
    
    SalesRecords[which(SalesRecords$Item_Outlet_Sales %in% x$out),"Item_Outlet_Sales"] <- 
      mean(SalesRecords$Item_Outlet_Sales,na.rm = T)
   
  
   
   
   
#---------------------------Step 5:Data Exploration & Feature engineering--------------------------------##
   
    SalesRecords_Analysis <- SalesRecords[1:8523,]
    
    # Analying relation between Item_Weight & Item_Outlet_Sales
    p <- ggplot(SalesRecords,aes(x=Item_Weight,y=Item_Outlet_Sales))
    p+geom_point()+labs(x="Item_Weight",y="Item_Outlet_Sales",
                        title="Item_Weight vs Item_Outlet_Sales")
    
    # Analying relation between Item_MRP & Item_Outlet_Sales
    p <- ggplot(SalesRecords,aes(x=Item_MRP,y=Item_Outlet_Sales))
    p+geom_point()+labs(x="Item_MRP",y="Item_Outlet_Sales",
                        title="Item_MRP vs Item_Outlet_Sales")
    
    #Creating new variable Item_Type_combined from Item_Identifier
    SalesRecords$Item_Type_combined <- substr(SalesRecords$Item_Identifier,1,2)
    table(SalesRecords$Item_Type_combined)
    
    # Modifying  Item_Fat_Content according exploration analysis
    table(SalesRecords$Item_Fat_Content)
    SalesRecords$Item_Fat_Content <- mapvalues(SalesRecords$Item_Fat_Content,
                                               from=c('low fat','Low Fat','Regular'),
                                               to=c('LF','LF','reg')
                                               )
    
    
    SalesRecords[which(SalesRecords$Item_Type_combined=='NC'),"Item_Fat_Content"]<-'NE'
    
    #Creating new variable No. of year - outlet running
    SalesRecords$No_of_year=2017-SalesRecords$Outlet_Establishment_Year
    
    #Creating dummy variable for Item_Type_combined
    Item_Type_combined_dummies <- dummy(SalesRecords$Item_Type_combined, sep = "_")
    Item_Type_combined_dummies <- as.data.frame(Item_Type_combined_dummies)
    Item_Type_combined_dummies <- Item_Type_combined_dummies[-1]
    SalesRecords               <- cbind(SalesRecords,Item_Type_combined_dummies)
    
    #Creating dummy variable for Item_Fat_Content
    Item_Fat_Content_dummies <- dummy(SalesRecords$Item_Fat_Content, sep = "_")
    Item_Fat_Content_dummies <- as.data.frame(Item_Fat_Content_dummies)
    Item_Fat_Content_dummies <- Item_Fat_Content_dummies[-1]
    SalesRecords             <- cbind(SalesRecords,Item_Fat_Content_dummies)
    
    
    #Creating dummy variable for Outlet_Size
    Outlet_Size_dummies <- dummy(SalesRecords$Outlet_Size, sep = "_")
    Outlet_Size_dummies <- as.data.frame(Outlet_Size_dummies)
    Outlet_Size_dummies <- Outlet_Size_dummies[-1]
    SalesRecords             <- cbind(SalesRecords,Outlet_Size_dummies)
    
    
    #Creating dummy variable for Outlet_Location_Type
    Outlet_Location_Type_dummies <- dummy(SalesRecords$Outlet_Location_Type, sep = "_")
    Outlet_Location_Type_dummies <- as.data.frame(Outlet_Location_Type_dummies)
    Outlet_Location_Type_dummies <- Outlet_Location_Type_dummies[-1]
    SalesRecords             <- cbind(SalesRecords,Outlet_Location_Type_dummies)
    
    #Creating dummy variable for Outlet_Type
    Outlet_Type_dummies <- dummy(SalesRecords$Outlet_Type, sep = "_")
    Outlet_Type_dummies <- as.data.frame(Outlet_Type_dummies)
    Outlet_Type_dummies <- Outlet_Type_dummies[-1]
    SalesRecords             <- cbind(SalesRecords,Outlet_Type_dummies)
    
   
    
    
#-------------------Step 6: droping dependent variable and Calculating Multicollinearity----
    
    #---------------------------------- Droping unnecessary columns---------------
    
    SalesRecords <- SalesRecords %>% select(-c(Item_Identifier,Item_Type,Outlet_Establishment_Year,
                                               Outlet_Identifier,Item_Type_combined,Item_Fat_Content,
                                               Outlet_Size,Outlet_Location_Type,Outlet_Type)) 
    names(SalesRecords)
    #---------------------------------- Calculating Multicollinearity---------------
    
    SalesRecords_1 <- SalesRecords%>%select(-Item_Outlet_Sales)
    
    #Identifying numeric variables
    numericData <- SalesRecords_1[sapply(SalesRecords_1, is.numeric)]
    
    #Calculating Correlation
    descrCor <- cor(numericData)
    
    # Checking Variables that are highly correlated
   
    highlyCorrelated = findCorrelation(descrCor, cutoff=0.7)
    
    #Identifying Variable Names of Highly Correlated Variables
    #highlyCorCol = names(numericData[,highlyCorrelated])
    highlyCorCol = names(numericData)[highlyCorrelated]
    
    #Remove highly correlated variables if present and create a new dataset
    SalesRecords <-SalesRecords[,-which(names(SalesRecords) %in% highlyCorCol)]
    # Visualize Correlation Matrix
    corrplot(descrCor, order = "FPC", method = "color", type = "lower",
             tl.cex = 0.7, tl.col = rgb(0, 0, 0))
    

  
#----------------------Step 7: Separating train/test dataset and Normalize data-------------   
    Train <- SalesRecords[1:8523,]
    Test  <- SalesRecords[8524:14204,]
    
    X_train <- Train %>% select(-Item_Outlet_Sales)
    Y_train <- Train %>% select(Item_Outlet_Sales)
    
    X_test  <- Test %>% select(-Item_Outlet_Sales)
    Y_test  <- Test %>% select(Item_Outlet_Sales)
  
    #-----Scaling
    X_train <- scale(X_train,center=TRUE,scale=TRUE)
    X_test  <- scale(X_test,center=TRUE,scale=TRUE)
  
    
    
    #--------PCA Analysis
    prin_comp <- prcomp(X_train, scale. = F)
    pr_var <- (prin_comp$sdev)^2
    prop_varex <- pr_var/sum(pr_var)
    
    #Cumulative Variance explains
    Cum_prop_varex<-cumsum(prop_varex)
    plot(Cum_prop_varex, xlab = "Principal Component",
         ylab = "Cumulative Proportion of Variance Explained",
         type = "b")
    
    # As per analysis, we can skip 2 principal componet, use only 13 components
    X_train <-prin_comp$x[,1:11] 
 
    X_test_predict  <-predict(prin_comp, newdata = X_test)
    X_test_predict <- as.data.frame(X_test_predict)
    X_test <-X_test_predict[,1:11]
   
    

#---------------------------Step 8: Run Algorithm-------------------------------------------------
    
    Train_data <- data.frame(X_train,"Item_Outlet_Sales" =Y_train)
    Test_data  <- data.frame(X_test ,"Item_Outlet_Sales" =Y_test)
  
    TrainTask <- makeRegrTask(data=Train_data,target ="Item_Outlet_Sales")
    TestTask  <- makeRegrTask(data=Test_data,target ="Item_Outlet_Sales")
    
 #---------------To Check importance of variable---------------------------------------------#
    
   
    im_feat <- generateFilterValuesData(TrainTask, method = c("information.gain","chi.squared"
                                                              ))
    
    plotFilterValues(im_feat,n.show = 20)
    l
 #1.Linear regression:  
     Linear.learner <- makeLearner("regr.lm",predict.type ="response")
    
    #cross validation (cv) accuracy
     cv.linear <- crossval(learner = Linear.learner,task = TrainTask,iters = 3,
                             measures = rmse,show.info = T)
     
    #cross validation accuracy
     cv.linear$aggr
     
     #train model
   
     LinearModel <-mlr::train(learner=Linear.learner,task=TrainTask)
     getLearnerModel(LinearModel)
     
     
   #2.Decision Tree 
      DecisionTree.learner <- makeLearner("regr.rpart", predict.type = "response")   
     #set 3 fold cross validation
      set_cv <- makeResampleDesc("CV",iters = 3L)
      #Search for hyperparameters
      DecisionTree.Parameter <- makeParamSet(
        makeIntegerParam("minsplit",lower = 2, upper = 10),
        makeIntegerParam("minbucket", lower = 2, upper = 10),
        makeIntegerParam("maxdepth", lower = 3, upper = 10),
        makeNumericParam("cp", lower = 0.001, upper = 0.2)
      )
    
      #search strategy
      ctrl <- makeTuneControlRandom(maxit = 200L)
      
      parallelStartSocket(cpus = detectCores())
      
      DecisionTree.tune <- tuneParams(learner = DecisionTree.learner, task = TrainTask, 
                                      resampling = set_cv,measures = rmse, par.set = DecisionTree.Parameter, 
                                      control = ctrl, show.info = T)
    
    
      parallelStop()
    
      #set hyperparameters
      DecisionTree.learner.tune <- setHyperPars(DecisionTree.learner,par.vals = DecisionTree.tune$x)
      
      #cross validation (cv) accuracy
      cv.DecisionTree <- crossval(learner = DecisionTree.learner.tune,task = TrainTask,iters = 3,
                                 measures = rmse,show.info = T)
      
      #cross validation accuracy
      cv.DecisionTree$aggr
      
      #train model
      Model_DecisionTree <- mlr::train(learner = DecisionTree.learner.tune,task = TrainTask)
      
      summary(getLearnerModel(Model_DecisionTree))
     
    #3.Random Forest
      RandomForest.learner <- makeLearner("regr.randomForest", predict.type = "response")  
      getParamSet("regr.randomForest")
      RandomForest.learner$par.vals <- list(
        importance = TRUE
      )
      #set 3 fold cross validation
      set_cv <- makeResampleDesc("CV",iters = 3L)
      
      #Search for hyperparameters
      RandomForest.Parameter <- makeParamSet(
        makeIntegerParam("ntree",lower = 50, upper = 100),
        makeIntegerParam("mtry", lower = 2, upper = 10),
        makeIntegerParam("nodesize", lower = 1, upper = 10)
      )
      
      #search strategy
      ctrl <- makeTuneControlRandom(maxit = 5L)
      
      parallelStartSocket(cpus = detectCores())
      
      RandomForest.tune <- tuneParams(learner = RandomForest.learner, task = TrainTask, 
                                      resampling = set_cv,measures = rmse, par.set = RandomForest.Parameter, 
                                      control = ctrl, show.info = T)
      
      
      parallelStop()
      
      #set hyperparameters
      RandomForest.learner.tune <- setHyperPars(RandomForest.learner,par.vals = RandomForest.tune$x)
      
      #cross validation (cv) accuracy
      cv.RandomForest <- crossval(learner = RandomForest.learner.tune,task = TrainTask,iters = 3,
                              measures = rmse,show.info = T)
      
      #cross validation accuracy
      cv.RandomForest$aggr
      #train model
      Model_RandomForest <- mlr::train(learner = RandomForest.learner,task = TrainTask)
      summary(getLearnerModel(Model_RandomForest))
     
    
    #4.Xgboost:
      Xgboost.learner <- makeLearner("regr.xgboost", predict.type = "response")  
      
      Xgboost.learner$par.vals <- list( objective="reg:linear", 
                                         nrounds=100L, eta=0.1)
      #set 3 fold cross validation
      set_cv <- makeResampleDesc("CV",iters = 3L)
      
      #Search for hyperparameters
      Xgboost.Parameter <- makeParamSet(
        makeNumericParam("lambda",lower=0.55,upper=0.60),
        makeIntegerParam("max_depth",lower = 3L,upper = 10L), 
        makeNumericParam("min_child_weight",lower = 1L,upper = 10L),
        makeNumericParam("subsample",lower = 0.5,upper = 1),
        makeNumericParam("colsample_bytree",lower = 0.5,upper = 1)
      )
      
      #search strategy
      ctrl <- makeTuneControlRandom(maxit = 2L)
      
      
      parallelStartSocket(cpus = detectCores())
      
      Xgboost.tune <- tuneParams(learner = Xgboost.learner, task = TrainTask, 
                                      resampling = set_cv,measures = rmse, par.set = Xgboost.Parameter, 
                                      control = ctrl, show.info = T)
      
      
      parallelStop()
      
      #set hyperparameters
      Xgboost.learner.tune <- setHyperPars(Xgboost.learner,par.vals = Xgboost.tune$x)
      
      #cross validation (cv) accuracy
      cv.Xgboost <- crossval(learner = Xgboost.learner.tune,task = TrainTask,iters = 3,
                              measures = rmse ,show.info = T)
      
      #cross validation accuracy
      cv.Xgboost$aggr
      
      #train model
      Model_Xgboost <- mlr::train(learner = Xgboost.learner.tune,task = TrainTask)
      summary(getLearnerModel(Model_Xgboost))
      
    
#-----------------------------------Ensembling Different Model-----------------------------#
      
      
 # Prdiction of Test task:
    Predict_Xgboost      <- predict(Model_Xgboost, TestTask)
    Predict_RandomForest <- predict(Model_RandomForest, TestTask)
    Predict_DecisionTree <- predict(Model_DecisionTree, TestTask)
    Predict_LinearModel  <- predict(LinearModel, TestTask)
    
    
    
    Model_prediction_avg <-  (Predict_LinearModel$data$response+
                              Predict_DecisionTree$data$response+
                              Predict_RandomForest$data$response + 
                              Predict_Xgboost$data$response)/4
                              
    
    
   
    
    
    Df<-data.frame(Item_Identifier=Test_df$Item_Identifier,
                   Outlet_Identifier=Test_df$Outlet_Identifier,
                   Item_Outlet_Sales=Model_prediction_avg
                    )
    write.csv(Df,file="Report.csv",row.names = F)