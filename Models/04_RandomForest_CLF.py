import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report,roc_auc_score,RocCurveDisplay


df=pd.read_csv('cleaned_survey.csv')



features = ['Gender', 'self_employed', 'family_history',
       'work_interfere', 'remote_work', 'benefits', 'care_options',
       'wellness_program', 'seek_help', 'leave', 'mental_health_consequence',
       'coworkers', 'supervisor', 'mental_health_interview'

]


X= df[features]

# encoding target variable
y=df['treatment']
y=y.map({'Yes':1,'No':0})

#splitting dataset
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# Preprocessing
cat_cols = X.select_dtypes(include='object').columns.tolist()
preprocessing = ColumnTransformer([('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols)], remainder='passthrough')

# Setting up pipeline
pipe = Pipeline([
    ('pre', preprocessing),
    ('model', RandomForestClassifier(max_depth=1000,random_state=42))
])


# Setting up our parmeter grid
param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [20, 50, None],
    'model__min_samples_split': [2, 5],
    'model__min_samples_leaf': [1, 2],
    'model__max_features': ['sqrt', 'log2']
}

# Searching best parameters to train our model
search = RandomizedSearchCV(
    pipe, param_distributions=param_grid,
    n_iter=10, cv=5, scoring='roc_auc', random_state=42
)

# Fitting our dataset
search.fit(X_train, y_train)

# Predicting
y_pred = search.predict(X_test)


#Evaluation
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_pred))


import joblib
joblib.dump(search,'clf_model.pkl')





import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, RocCurveDisplay
import joblib
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('cleaned_survey.csv')

# Feature columns and target
features = ['Gender', 'self_employed', 'family_history', 'work_interfere', 'remote_work',
            'benefits', 'care_options', 'wellness_program', 'seek_help', 'leave',
            'mental_health_consequence', 'coworkers', 'supervisor', 'mental_health_interview']

X = df[features]
y = df['treatment'].map({'Yes': 1, 'No': 0})  # Encode target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing: OneHotEncode categorical features
cat_cols = X.select_dtypes(include='object').columns.tolist()
preprocessing = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols)],
    remainder='passthrough'
)

# Create pipeline with logistic regression
pipe = Pipeline([
    ('pre', preprocessing),
    ('model', LogisticRegression( max_iter=10000, random_state=42))
])



# Parameter grid for logistic regression
param_grid = {
    'model__penalty': ['l1', 'l2', 'elasticnet', None],
    'model__C': [0.01, 0.1, 1, 10],
    'model__l1_ratio': [0, 0.5, 1]  
}



# Randomized search
search1 = RandomizedSearchCV(
    pipe,
    param_distributions=param_grid,
    n_iter=10,
    cv=5,
    scoring='roc_auc',
    random_state=42
)

# Fit the model
search1.fit(X_train, y_train)

# Predict labels and probabilities
y_pred = search.predict(X_test)
y_proba = search1.predict_proba(X_test)[:, 1]

# Evaluation
print("Best Parameters:", search1.best_params_)
print(classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_proba))




