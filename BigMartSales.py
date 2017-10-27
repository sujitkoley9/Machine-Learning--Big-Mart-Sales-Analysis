 #----------------- Step 1:  Importing required packages for this problem ----------------------------------- 
   # data analysis and wrangling
    import pandas as pd
    import numpy as np
    import random as rn
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    
    # visualization
    import seaborn as sns
    import matplotlib.pyplot as plt
   
    # machine learning
    from sklearn.linear_model import LinearRegression
    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    from sklearn.model_selection import cross_val_score
    
    import xgboost as xgb
    from xgboost.sklearn import XGBClassifier
    from xgboost.sklearn import XGBRegressor
    from xgboost  import plot_importance
    
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import RandomizedSearchCV
    
    import keras
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.wrappers.scikit_learn import KerasClassifier
    
    
  #--------- Step 2:  Reading and loading train and test datasets and generate data quality report----------- 
   
    # loading train and test sets with pandas 
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    #append two train  and test dataframe
    full  = train_df.append(test_df,ignore_index=True)
    
    # Print the columns of dataframe
    print(full.columns.values)
    
    # Returns first n rows
    full.head(10)
    
    
    # Retrive data type of object and no. of non-null object
    full.info()
    
    # Retrive details of integer and float data type 
    full.describe()
    
    # To get  details of the categorical types
    full.describe(include=['O'])

   

  #Prepare data quality report-
  # To get count of no. of NULL for each data type columns = full.columns.values
    columns = full.columns.values
    data_types = pd.DataFrame(full.dtypes, columns=['data types'])
    
    missing_data_counts = pd.DataFrame(full.isnull().sum(),
                            columns=['Missing Values'])
    
    present_data_counts = pd.DataFrame(full.count(), columns=['Present Values'])
    
    UniqueValues = pd.DataFrame(full.nunique(), columns=['Unique Values'])
    
    MinimumValues = pd.DataFrame(columns=['Minimum Values'])
    for c in list(columns):
       if (full[c].dtypes == 'float64' ) | (full[c].dtypes == 'int64'):
            MinimumValues.loc[c]=full[c].min()
       else:
            MinimumValues.loc[c]=0
 
    MaximumValues = pd.DataFrame(columns=['Maximum Values'])
    for c in list(columns):
       if (full[c].dtypes == 'float64' ) |(full[c].dtypes == 'int64'):
            MaximumValues.loc[c]=full[c].max()
       else:
            MaximumValues.loc[c]=0
    
    data_quality_report=data_types.join(missing_data_counts).join(present_data_counts).join(UniqueValues).join(MinimumValues).join(MaximumValues)
    data_quality_report.to_csv('Big_Mart_sales.csv', index=True)



   
   
   
#---------------Step 3: Missing value treatment----------------------------------------------------------------  
  
    
     # Treatment for Item_Weight
   full['Item_Weight'].fillna(full['Item_Weight'].mean(), inplace=True)
 
  
     # Treatment for Outlet_Size
  
   full['Outlet_Size'].fillna('Missing', inplace=True)
   
 
#--------------Step 4:Outlier Treatment ----------------------------------------------------------------------  

#  outlier treatment using BoxPlot 
   
    # Item_MRP

     BoxPlot=boxplot(full['Item_MRP'])
     outlier= BoxPlot['fliers'][0].get_data()[1]
     # No outlier , No need any operation for Item_MRP
     #full.loc[full['Item_MRP'].isin(outlier),'Item_MRP']=full['Item_MRP'].mean()
     
     #Item_Weight
     BoxPlot=boxplot(full['Item_Weight'])
     outlier= BoxPlot['fliers'][0].get_data()[1]
     # No outlier , No need any operation for Item_Weight
     #full.loc[full['Item_Weight'].isin(outlier),'Item_Weight']=full['Item_Weight'].mean()

     #Item_Outlet_Sales
     BoxPlot=boxplot(full[0:8522]['Item_Outlet_Sales'])
     outlier= BoxPlot['fliers'][0].get_data()[1]
     full.loc[full['Item_Outlet_Sales'].isin(outlier),'Item_Outlet_Sales']=full[0:8522]['Item_Outlet_Sales'].mean()
    

#-----------------Step 5:Exploration analysis of data---------------------------------------------------------

        # Create photocopy of  trian portion of full and assign it full1
        full1=full[0:8522].copy()
        
       # Analying relation between Item_Weight & Item_Outlet_Sales
        sns.lmplot(x='Item_Weight', y='Item_Outlet_Sales', data=full1)
        
       # Analying relation between Item_MRP & Item_Outlet_Sales
        sns.lmplot(x='Item_MRP', y='Item_Outlet_Sales', data=full1)
        
     # Analying relation between Item_Visibility & Item_Outlet_Sales
       full2= full1[(full1['Item_MRP']>=240) & (full1['Item_MRP']<=241)]
       
       sns.lmplot(x='Item_Visibility', y='Item_Outlet_Sales', data=full2)
       
   
      # Analying relation between Item_Id & Item_Outlet_Sales
      # Retrieve numeric part of Item_Identifier and create new column
       full1['Item_Id'] = full1['Item_Identifier'].str[3:].astype(int)
       full2= full1[(full1['Item_MRP']>=240) & (full1['Item_MRP']<=241)]
       
       sns.lmplot(x='Item_Id', y='Item_Outlet_Sales', data=full2)
       
     
       
       

#------------------------------------Step 6:Feature Engineering--------------------------------------
  
   #Creating new variable Item_Type_combined from Item_Identifier
   full['Item_Type_combined'] = full['Item_Identifier'].str[0:2]
   full['Item_Type_combined'].value_counts()
   
  
  
   
   
   # Modifying  Item_Fat_Content according exploration analysis
   full['Item_Fat_Content'].value_counts()
   full['Item_Fat_Content']=full['Item_Fat_Content'].replace({'Low Fat':'LF',
                                                             'low fat':'LF',
                                                             'Regular':'reg'}
                                                            )
   
   
   full.loc[full['Item_Type_combined']=='NC','Item_Fat_Content']='NE'
   
   #Creating new variable No. of year - outlet running
   full['No_of_year']=2017-full['Outlet_Establishment_Year']
   
   
   
   # Create dummy variable for Item_Type_combined
   Item_Type_combined_dummies = pd.get_dummies(full['Item_Type_combined'],prefix='Item_Type_combined')
   Item_Type_combined_dummies=Item_Type_combined_dummies.iloc[:,1:]
   full=full.join(Item_Type_combined_dummies)
   
   
   #Creating dummy variable for Item_Fat_Content
   Item_Fat_Content_dummies = pd.get_dummies(full['Item_Fat_Content'],prefix='Item_Fat_Content')
   Item_Fat_Content_dummies=Item_Fat_Content_dummies.iloc[:,1:]
   full=full.join(Item_Fat_Content_dummies)
   
    #Creating dummy variable for Outlet_Size
   Outlet_Size_dummies = pd.get_dummies(full['Outlet_Size'],prefix='Outlet_Size')
   Outlet_Size_dummies=Outlet_Size_dummies.iloc[:,1:]
   full=full.join(Outlet_Size_dummies)
   
    #Creating dummy variable for Outlet_Location_Type
   Outlet_Location_Type_dummies = pd.get_dummies(full['Outlet_Location_Type'],prefix='Outlet_Location_Type')
   Outlet_Location_Type_dummies=Outlet_Location_Type_dummies.iloc[:,1:]
   full=full.join(Outlet_Location_Type_dummies)
   
   
    #Creating dummy variable for Outlet_Type
   Outlet_Type_dummies = pd.get_dummies(full['Outlet_Type'],prefix='Outlet_Type')
   Outlet_Type_dummies=Outlet_Type_dummies.iloc[:,1:]
   full=full.join(Outlet_Type_dummies)
   
  
   
 
   
   
   
     #---------------------------------- Droping unnecessary columns-------------------------------
    full.drop(['Item_Identifier','Item_Type','Outlet_Establishment_Year','Outlet_Identifier','Item_Type_combined',
               'Item_Fat_Content','Outlet_Size','Outlet_Location_Type','Outlet_Type'], axis=1, inplace=True)
    
    full.columns.values

   
   
   
#----------------------Step 7: Separating train/test dataset and Normalize data--------------------------------   
    train_new=full[0:8523]
    test_new=full[8523:]
    
    X_train = train_new.drop(['Item_Outlet_Sales'], axis=1)
    Y_train = train_new["Item_Outlet_Sales"]
    
    X_test  = test_new.drop(['Item_Outlet_Sales'], axis=1)
   
    #-----Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    
     #--------------------PCA to reduce dimension and remove correlation----------------------------    
     pca = PCA(n_components =16)
     pca.fit_transform(X_train)
     #The amount of variance that each PC explains
     var= pca.explained_variance_ratio_
     #Cumulative Variance explains
     var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
     plt.plot(var1)
     
     # As per analysis, we can skip 4 principal componet, use only 11 components
     
     pca = PCA(n_components =13)
     X_train=pca.fit_transform(X_train)
     X_test=pca.fit_transform(X_test)
     
     
#----------------------Step 8:Run Algorithm----------------------------------------------------------------------
  #1.Logistic Regression
    
    Linreg = LinearRegression()
    Linreg.fit(X_train, Y_train)
    Linreg_score = cross_val_score(estimator = Linreg, X = X_train, y = Y_train, cv =    10,
                                 scoring='neg_mean_squared_error')
    
    Linreg_score = (np.sqrt(np.abs(Linreg_score)))
    
    Linreg_score_mean = Linreg_score.mean()
    Linreg_score_std = Linreg_score.std()
    
  #2.Decision Tree
    decision_tree = DecisionTreeRegressor()
    decision_tree.fit(X_train, Y_train)
    decision_tree_score = cross_val_score(estimator = decision_tree, X = X_train, y = Y_train, cv = 10,
                                        scoring='neg_mean_squared_error')
    
    decision_tree_score = (np.sqrt(np.abs(decision_tree_score)))
    
    decision_tree_score_mean = decision_tree_score.mean()
    decision_tree_score_std = decision_tree_score.std()
    
    # Choose some parameter combinations to try
    parameters = {
                  'max_features': ['log2', 'sqrt','auto'],
                  'criterion': ['mse', 'friedman_mse'],
                  'max_depth': range(2,10), 
                  'min_samples_split': range(2,10),
                  'min_samples_leaf': range(1,10)
                 }

    # Search for best parameters
    grid_obj = GridSearchCV(estimator=decision_tree, 
                                    param_grid= parameters,
                                    scoring = 'neg_mean_squared_error',
                                    cv = 10,n_jobs=-1)
    

    
    grid_obj = grid_obj.fit(X_train, Y_train)

    # Set the clf to the best combination of parameters
    decision_tree_best = grid_obj.best_estimator_
    
    # Fit the best algorithm to the data. 
    decision_tree_best.fit(X_train, Y_train)
    
    # Calculate accuracy of decisison tree again
    decision_tree_score = cross_val_score(estimator = decision_tree_best, X = X_train, y = Y_train, cv = 10,
                                        scoring='neg_mean_squared_error')
    
    decision_tree_score = (np.sqrt(np.abs(decision_tree_score)))
    
    decision_tree_score_mean = decision_tree_score.mean()
    
    decision_tree_score_std = decision_tree_score.std()
    #---To Know importanve of variable
    feature_importance = pd.Series(decision_tree_best.feature_importances_, X_train.columns.values).sort_values(ascending=False)
    feature_importance.plot(kind='bar', title='Feature Importances')
    
   
    
    
    
    #3.Random Forest
    random_forest = RandomForestRegressor(n_estimators=400)
    random_forest.fit(X_train, Y_train)
    random_forest_score = cross_val_score(estimator = random_forest, X = X_train, y = Y_train, cv = 10,
                                        scoring='neg_mean_squared_error')
    random_forest_score = (np.sqrt(np.abs(random_forest_score)))
    
    random_forest_score_mean = random_forest_score.mean()
    random_forest_score_std  = random_forest_score.std()


    # Choose some parameter combinations to try
    parameters = { 
                 'max_features': ['log2', 'sqrt','auto'], 
                 'criterion': ['mse', 'mae'],
                 'max_depth': range(2,10), 
                 'min_samples_split': range(2,10),
                 'min_samples_leaf': range(1,10)
                 }
    
     # Choose some parameter combinations to try
    parameters = { 
                  'max_features': ['auto'], 
                  'criterion': ['mse'],
                  'max_depth': [8], 
                 'min_samples_split': [3],
                 'min_samples_leaf': [3]
                 }
   
    grid_obj = GridSearchCV(estimator=random_forest, 
                                    param_grid= parameters,
                                    scoring = 'neg_mean_squared_error',
                                    cv = 3,n_jobs=-1)
    

    
    grid_obj = grid_obj.fit(X_train, Y_train)
    

    
   

    # Set the clf to the best combination of parameters
    random_forest_best = grid_obj.best_estimator_
    
    # Fit the best algorithm to the data. 
    random_forest_best.fit(X_train, Y_train)
    random_forest_score = cross_val_score(estimator = random_forest_best, X = X_train, y = Y_train, cv = 10,
                                        scoring='neg_mean_squared_error')
    
    random_forest_score = (np.sqrt(np.abs(random_forest_score)))
    
    random_forest_score_mean = random_forest_score.mean()
    random_forest_score_std  = random_forest_score.std()
    
    #---To Know importanve of variable
    feature_importance = pd.Series(random_forest_best.feature_importances_, X_train.columns.values).sort_values(ascending=False)
    feature_importance.plot(kind='bar', title='Feature Importances')
    
    #4.XGBOOST
    Xgboost = XGBRegressor()
    Xgboost.fit(X_train, Y_train)
    Xgboost_score = cross_val_score(estimator = Xgboost, X = X_train, y = Y_train, cv = 10,
                                        scoring='neg_mean_squared_error')
    Xgboost_score = (np.sqrt(np.abs(Xgboost_score)))
    
    Xgboost_score_mean = Xgboost_score.mean()
    Xgboost_score_std  = Xgboost_score.std()



    # Choose some parameter combinations to try
   parameters = {'learning_rate':np.arange(0.1, .5, 0.1),
                  'n_estimators':[200],
                  'max_depth': range(4,10),
                  'min_child_weight':range(1,5),
                  'reg_lambda':np.arange(0.55, .9, 0.05),
                  'subsample':np.arange(0.1, 1, 0.1),
                  'colsample_bytree':np.arange(0.1, 1, 0.1)
               }
   
     # Choose some parameter combinations to try
   parameters = {'learning_rate':[.1,.3,.02],
                  'n_estimators':[73,74,75],
                  'max_depth': [3,4],
                  'min_child_weight':[1,2],
                  'reg_lambda':np.arange(0.55, .9, 0.05),
                  'subsample':[.8,0.9,1],
                  'colsample_bytree':[.8,0.9,1]
               }
    
    # Search for best parameters
   Random_obj = RandomizedSearchCV(estimator=Xgboost, 
                                  param_distributions = parameters,
                                  scoring = 'neg_mean_squared_error',
                                  cv = 3,n_iter=600,n_jobs=-1)
    

    
    Random_obj = Random_obj.fit(X_train, Y_train) 

    # Set the clf to the best combination of parameters
    Xgboost_best = Random_obj.best_estimator_
    
    # Fit the best algorithm to the data. 
    Xgboost_best.fit(X_train, Y_train)
    
    Xgboost_score = cross_val_score(estimator = Xgboost_best, X = X_train, y = Y_train, cv = 10,
                                        scoring='neg_mean_squared_error')
    Xgboost_score = (np.sqrt(np.abs(Xgboost_score)))
    
    Xgboost_score_mean = Xgboost_score.mean()
    Xgboost_score_std  = Xgboost_score.std()


    #---To Know importanve of variable
    plot_importance(Xgboost_best)
    pyplot.show()
    
 #5.SVM
    SVM_model=SVR()
    SVM_model.fit(X_train, Y_train)
    SVM_model_score = cross_val_score(estimator = SVM_model, X = X_train, y = Y_train, cv = 10,
                                        scoring='neg_mean_squared_error')
    SVM_model_score = (np.sqrt(np.abs(SVM_model_score)))
    
    SVM_model_score_mean = SVM_model_score.mean()
    SVM_model_score_std  = SVM_model_score.std()




    # Choose some parameter combinations to try
   parameters = { 'kernel':('linear', 'rbf'),
                  'gamma': [0.01,0.02,0.03,0.04,0.05,0.10,0.2,0.3,0.4,0.5],
                  'C': np.arange(1, 10,1)
                 }
    
    # Search for best parameters
    Random_obj = RandomizedSearchCV(estimator=SVM_model, 
                                  param_distributions = parameters,
                                  scoring = 'neg_mean_squared_error',
                                  cv = 3,n_iter=100,n_jobs=-1)
    

    
    Random_obj = Random_obj.fit(X_train, Y_train)

    # Set the clf to the best combination of parameters
    SVM_model_best = Random_obj.best_estimator_
    
    # Fit the best algorithm to the data. 
    SVM_model_best.fit(X_train, Y_train)
    SVM_model_score = cross_val_score(estimator = SVM_model_best, X = X_train, y = Y_train, cv = 10,
                                        scoring='neg_mean_squared_error')
    SVM_model_score = (np.sqrt(np.abs(SVM_model_score)))
    
    SVM_model_score_mean = SVM_model_score.mean()
    SVM_model_score_std  = SVM_model_score.std()
  

  #.6.KNN
    KNN_model=KNeighborsRegressor() 
    KNN_model.fit(X_train, Y_train)
    KNN_model_score = cross_val_score(estimator = KNN_model, X = X_train, y = Y_train, cv = 10,
                                        scoring='neg_mean_squared_error')
    KNN_model_score = (np.sqrt(np.abs(KNN_model_score)))
    
    KNN_model_score_mean = KNN_model_score.mean()
    KNN_model_score_std  = KNN_model_score.std()



    # Choose some parameter combinations to try
    parameters = { 'n_neighbors': np.arange(1, 31, 1),
	              'metric': ["minkowski"]
                 }
    
   #Search for best parameters
    Random_obj = RandomizedSearchCV(estimator=KNN_model, 
                                  param_distributions = parameters,
                                  scoring = 'neg_mean_squared_error',
                                  cv = 10,n_iter=30,n_jobs=-1)
    

    
    Random_obj = Random_obj.fit(X_train, Y_train)

    # Set the clf to the best combination of parameters
    KNN_model_best = Random_obj.best_estimator_
    
    # Fit the best algorithm to the data. 
    KNN_model_best.fit(X_train, Y_train)
    KNN_model_score = cross_val_score(estimator = KNN_model_best, X = X_train, y = Y_train, cv = 10,
                                        scoring='neg_mean_squared_error')
    KNN_model_score = (np.sqrt(np.abs(KNN_model_score)))
    
    KNN_model_score_mean = KNN_model_score.mean()
    KNN_model_score_std  = KNN_model_score.std()
    
    
    
 
    

 #---------------Step 9:Prediction on test data -------------------------------------------------------
     Y_pred1 = logreg.predict(X_test)
     Y_pred2 = decision_tree_best.predict(X_test)
     Y_pred3 = random_forest_best.predict(X_test)
     Y_pred4 = Xgboost_best.predict(X_test)
     Y_pred5 = SVM_Classifier_best.predict(X_test)
     Y_pred6 = KNN_Classifier_best.predict(X_test)
    
    
    
     
    submission = pd.DataFrame({
            "Item_Identifier": test_df["Item_Identifier"],
            "Outlet_Identifier": test_df["Outlet_Identifier"],
            "Item_Outlet_Sales": Y_pred4
        })
    submission=submission[["Item_Identifier","Outlet_Identifier","Item_Outlet_Sales"]]
    
    submission.to_csv('submission.csv', index=False)





 
 