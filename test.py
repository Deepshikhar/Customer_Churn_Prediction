# %%
# Import the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
sns.set_style('darkgrid')

# %% [markdown]
# ## Loading the Data

# %%
# Loading the Data
df = pd.read_csv('train.csv', index_col=False,delimiter=';')

test = pd.read_csv("test.csv")

print("Shape :",df.shape)
print("Describe :",df.describe())
print("Null_values :",df.isnull().sum())
df.head(5)

# %% [markdown]
# ## Feature Selection

# %%
# Converting Categorical to numerical
df["marital"]=df["marital"].map({'single':0, 'married':1, 'divorced':2})
df["default"]=df["default"].map({'no':0, 'yes':1})
df["housing"]=df["housing"].map({'no':0, 'yes':1})
df["loan"]=df["loan"].map({'no':0, 'yes':1})


df.head()

# %%
# Calculating 'unknown' values in poutcome
df['poutcome'].replace({'unknown': None},inplace =True, regex= True)
print(df['poutcome'].unique())

df.head()


# %%
df['poutcome'].unique()


# %%
df["education"]=df["education"].map({'primary':0, 'secondary':1, 'tertiary':2,'unknown':3})
df["poutcome"]=df["poutcome"].map({'None':0, 'failure':1, 'other':2,'success':3})

# %%
# using one hot encoder for nominal data in job feature
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
ohe = OneHotEncoder()
# df[list(df["job"].unique())] = ohe.fit_transform(df[["job"]]).A
# df.drop("job",axis = 1, inplace = True)

# %%
label = LabelEncoder()
df["y"] =  label.fit_transform(df["y"])

# %%
df.drop("contact", inplace = True, axis = 1)
df.drop("month", inplace = True, axis = 1)

# %%
df = pd.get_dummies(df,drop_first=True)
df.head()

# %%
df.head()


# %%
df.describe()

# %%
from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = df[[ 'age', 'marital','education','default', 'balance', 'housing', 'loan',
       'duration', 'campaign','pdays','previous']]
vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(variables.values,i) for i in range(variables.shape[1])]
vif['features'] = variables.columns
vif

# %% [markdown]
# From the vif we can see that all the features are significant except age

# %%
df

# %%
df.columns

# %% [markdown]
# ### Split the data into train and test set to avoid overfitting

# %%
x = df[['age', 'education', 'default', 'balance', 'housing', 'loan',
       'duration', 'campaign', 'pdays', 'previous', 'poutcome',
       'job_blue-collar', 'job_entrepreneur', 'job_housemaid',
       'job_management', 'job_retired', 'job_self-employed', 'job_services',
       'job_student', 'job_technician', 'job_unemployed', 'job_unknown'
       ]]
y = df['y']

# %%

from sklearn.model_selection import train_test_split
X_train , X_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=3)

# %%
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)

# %%


# %% [markdown]
# ## Logistic Regression Model

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

LR_model = LogisticRegression(C=0.05, solver="liblinear").fit(X_train_scaled,y_train)
y_pred= LR_model.predict(X_test)

# Calculate metrics
metrics = {'Accuracy': accuracy_score(y_test, y_pred),
           'Precision': precision_score(y_test, y_pred),
           'Recall': recall_score(y_test, y_pred),
           'F1-score': f1_score(y_test, y_pred),
           'AUC-ROC': roc_auc_score(y_test, y_pred)}

eval_metrics = pd.DataFrame([metrics], columns=metrics.keys())
eval_metrics

# %% [markdown]
# ## Random Forest

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

RF_model = RandomForestClassifier().fit(X_train, y_train)
y_pred = RF_model.predict(X_test)

# Calculate metrics
metrics = {'Accuracy': accuracy_score(y_test, y_pred),
           'Precision': precision_score(y_test, y_pred),
           'Recall': recall_score(y_test, y_pred),
           'F1-score': f1_score(y_test, y_pred),
           'AUC-ROC': roc_auc_score(y_test, y_pred)}

eval_metrics = pd.DataFrame([metrics], columns=metrics.keys())
eval_metrics


# %% [markdown]
# ## Gradient boost

# %%
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

GB_model = GradientBoostingClassifier().fit(X_train, y_train)
y_pred = GB_model.predict(X_test)

# Calculate metrics
metrics = {'Accuracy': accuracy_score(y_test, y_pred),
           'Precision': precision_score(y_test, y_pred),
           'Recall': recall_score(y_test, y_pred),
           'F1-score': f1_score(y_test, y_pred),
           'AUC-ROC': roc_auc_score(y_test, y_pred)}

eval_metrics = pd.DataFrame([metrics], columns=metrics.keys())
eval_metrics


# %% [markdown]
# ## Hyperparameter Tunning

# %% [markdown]
# 

# %%
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

# Define the gradient boosting classifier
gbc = GradientBoostingClassifier()

param_grid = {'n_estimators': [100,200, 300],
           'learning_rate': [0.001, 0.01, 0.1, 1],
           'max_depth': [3, 4, 6, 7],
         }


# Create the grid search object
grid_search = GridSearchCV(gbc, param_grid, cv=5,
                           scoring='accuracy', return_train_score=True)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Print the best parameters and best score
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# %%
# Logistic Regression Grid Search

LR_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
           'penalty': ['l1', 'l2'],
           'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
           'max_iter': [100, 200, 300, 400, 500],
           'tol': [1e-4, 1e-3, 1e-2, 1e-1]}


# Initialize the GridSearchCV object
LR_grid_search = GridSearchCV(LogisticRegression(), LR_grid, cv=5, scoring='accuracy')

# Fit the GridSearchCV object to the training data
LR_grid_search.fit(X_train, y_train)

# Print the best parameters and best score
print("Best parameters: {}".format(LR_grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(LR_grid_search.best_score_))

# %%
# Random Forest Grid Search

RF_grid = {'n_estimators': [100, 200, 300, 400, 500],
           'max_depth': [3, 4, 5, 6, 7],
           'min_samples_split': [2, 3, 4, 5],
           'min_samples_leaf': [1, 2, 3, 4, 5]}

# Initialize the GridSearchCV object
RF_grid_search = GridSearchCV(RandomForestClassifier(), RF_grid, cv=5, scoring='accuracy')

# Fit the GridSearchCV object to the training data
RF_grid_search.fit(X_train, y_train)

# Get the best hyperparameters
RF_best_params = RF_grid_search.best_params_

# Print the best parameters and best score
print("Best parameters: {}".format(RF_grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(RF_grid_search.best_score_))

# %% [markdown]
# ## Plotting Learning Curves

# %%
from sklearn.model_selection import learning_curve
# Plot learning curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """Generate a simple plot of the test and training learning curve"""
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=1, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

g = plot_learning_curve(RF_grid_search.best_estimator_,"RF learning curves",X_train,y_train,cv=10)
g = plot_learning_curve(LR_grid_search.best_estimator_,"LR learning curves",X_train,y_train,cv=10)
g = plot_learning_curve(grid_search.best_estimator_,"GradientBoosting learning curves",X_train,y_train,cv=10)


# %%
# Define the gradient boosting classifier
gbc = GradientBoostingClassifier(learning_rate= 0.01, max_depth= 4, n_estimators= 300)

# Fit the grid search to the data
gbc.fit(X_train, y_train)


# %%
from sklearn.metrics import accuracy_score, precision_recall_curve,classification_report,roc_curve, confusion_matrix
y_pred = gbc.predict(X_test)
acc_train = accuracy_score(y_test,y_pred)
class_re = classification_report(y_test,y_pred)
con_mat = confusion_matrix(y_test,y_pred)
print("Confusion Matrix:\n",con_mat)
print("\n")
print("The accuracy of the model:",(acc_train)*100)
print("\n")
print("The classification report:\n",class_re)


# %%
roc_auc_score(y_test,y_pred)

# %%
lr = LogisticRegression(C= 1, max_iter= 100, penalty= 'l2', solver= 'sag', tol= 0.1)

# Fit the grid search to the data
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
acc_train = accuracy_score(y_test,y_pred)
class_re = classification_report(y_test,y_pred)
con_mat = confusion_matrix(y_test,y_pred)
print("Confusion Matrix:\n",con_mat)
print("\n")
print("The accuracy of the model:",(acc_train)*100)
print("\n")
print("The classification report:\n",class_re)

# %%
roc_auc_score(y_test,y_pred)

# %%
rf = RandomForestClassifier(max_depth= 3, n_estimators= 100,min_samples_leaf = 1, min_samples_split = 2)

# Fit the grid search to the data
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
acc_train = accuracy_score(y_test,y_pred)
class_re = classification_report(y_test,y_pred)
con_mat = confusion_matrix(y_test,y_pred)
print("Confusion Matrix:\n",con_mat)
print("\n")
print("The accuracy of the model:",(acc_train)*100)
print("\n")
print("The classification report:\n",class_re)

# %%
roc_auc_score(y_test,y_pred)

# %% [markdown]
# ### From the above result we can observe that Gradient Boosting Algorithm works best while rest to try to overfit the data

# %% [markdown]
# ## Test the Model on Test Set
# 

# %%
test_dummies = test.copy()
test_dummies.head()

# %%
test_dummies["marital"]=test_dummies["marital"].map({'single':0, 'married':1, 'divorced':2})
test_dummies["default"]=test_dummies["default"].map({'no':0, 'yes':1})
test_dummies["housing"]=test_dummies["housing"].map({'no':0, 'yes':1})
test_dummies["loan"]=test_dummies["loan"].map({'no':0, 'yes':1})

test_dummies = pd.get_dummies(test_dummies,drop_first=True)


# %%
# Scale the data

# select the columns that are dummy variables
dummy_vars = test_dummies.loc[:, test_dummies.columns.str.startswith(('Id','job_', 'education_'))]

# # select the columns that are not dummy variables
features = test_dummies.loc[:, ~test_dummies.columns.isin(dummy_vars.columns)]

# # scale the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# # concatenate the scaled features with the original dummy variables
test_scaled = pd.concat([pd.DataFrame(features_scaled, columns=features.columns), dummy_vars], axis=1)

# %%
x_test = test_dummies[['age', 'marital', 'balance', 'housing', 'loan', 'day',
       'duration', 'campaign', 'job_blue-collar',
       'job_entrepreneur', 'job_housemaid', 'job_management', 'job_retired',
       'job_self-employed', 'job_services', 'job_student', 'job_technician',
       'job_unemployed', 'job_unknown', 'education_secondary',
       'education_tertiary', 'education_unknown']]


# %% [markdown]
# ### Testing using Gradient Boost Classifier

# %%
predictions = gbc.predict(x_test)

# Creating dataframe of predictions
pred_report = pd.DataFrame(predictions.tolist(),index = [i for i in range(len(predictions))],columns=["client will subscribe a term deposit"])
pred_report.index.name = 'ID'

# %%
# saving the prediction
pred_report.to_csv("final_submission.csv")


# >>>>>>>>>>>>>>>>>>>