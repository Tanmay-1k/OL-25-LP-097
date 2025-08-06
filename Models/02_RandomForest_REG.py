import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import RobustScaler,OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import r2_score,mean_absolute_error,root_mean_squared_error




df=pd.read_csv('cleaned_survey.csv')

feautures = ['Gender', 'self_employed', 'family_history', 'treatment',
       'work_interfere', 'remote_work', 'benefits', 'care_options',
       'wellness_program', 'seek_help', 'leave', 'mental_health_consequence',
       'coworkers', 'supervisor', 'mental_health_interview'

]

# creating feauture dataset and log transforming the target variable to reduce skewness
X= df[feautures]
y=np.log1p(df['Age'])


#getting categorical and numerical columns for preprocessing
cat_cols = X.select_dtypes(include = 'object').columns.tolist()
num_cols = X.select_dtypes(include = ['int64','float64']).columns.tolist()

#setting up pipeline for better workflow
preprocessing = ColumnTransformer([('cat',OrdinalEncoder(),cat_cols),('num',RobustScaler(),num_cols)],remainder='passthrough')
pipe = Pipeline([('pre',preprocessing),('model',RandomForestRegressor(random_state=42))])



param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [20, 50, None],
    'model__min_samples_split': [2, 5],
    'model__min_samples_leaf': [1, 2],
    'model__max_features': ['sqrt', 'log2']
}


#Tuning the hyperparameters to get the best features
search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=2
)



#Splitting the dataset 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


#fitting and predicting on best parameters
search.fit(X_train,y_train)
y_pred=search.predict(X_test)


#Evaluation
print('MAE :',mean_absolute_error(y_test,y_pred))
print('RMSE :',root_mean_squared_error(y_test,y_pred))
print('R2',r2_score(y_test,y_pred))







