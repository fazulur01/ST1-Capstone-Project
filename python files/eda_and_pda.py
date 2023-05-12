# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
#Import Required Packages for EDA 
import os
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno 
import plotly.graph_objects as go 
import plotly.express as px 
# %matplotlib inline
import warnings
warnings.filterwarnings('ignore')

#Read the dataset/s
df = pd.read_csv('datasetForTask_f.csv')

#Checking description(first 5 and last 5 rows)
df.head() #first 5 row

df.tail() # last 5 row

#rows and columns-data shape(attributes & samples)
df.shape

# name of the attributes
df.columns

#unique values for each attribute
df.nunique()

#Complete info about data frame
df.info()

le = LabelEncoder()
df['Country/Region/World'] = le.fit_transform(df['Country/Region/World'])
df['ISO'] = le.fit_transform(df['ISO'])
df['Sex'] = le.fit_transform(df['Sex'])

#Visualising data distribution in detail
fig = plt.figure(figsize =(18,18))
ax=fig.gca()
df.hist(ax=ax,bins =30)
plt.show()

#detecting outliers
df.plot(kind='box', subplots=True, layout=(2,7), sharex=False, sharey=False, figsize=(20, 10), color='deeppink');

#identify the outliers
# define continuous variable & plot
continous_features = ['Country/Region/World','ISO','Sex','Year','Age-standardised diabetes prevalence','Lower 95% uncertainty interval','Upper 95% uncertainty interval']  
def outliers(df_out, drop = False):
    for each_feature in df_out.columns:
        feature_data = df_out[each_feature]
        Q1 = np.percentile(feature_data, 25.) # 25th percentile of the data of the given feature
        Q3 = np.percentile(feature_data, 75.) # 75th percentile of the data of the given feature
        IQR = Q3-Q1 #Interquartile Range
        outlier_step = IQR * 1.5 #That's we were talking about above
        outliers = feature_data[~((feature_data >= Q1 - outlier_step) & (feature_data <= Q3 + outlier_step))].index.tolist()  
        if not drop:
            print('For the feature {}, No of Outliers is {}'.format(each_feature, len(outliers)))
        if drop:
            df.drop(outliers, inplace = True, errors = 'ignore')
            print('Outliers from {} feature removed'.format(each_feature))
outliers(df[continous_features])

#drop the outliers
outliers(df[continous_features], drop = True)

#check if outliers got removed
df.plot(kind='box', subplots=True,layout=(2,7),sharex=False,sharey=False,figsize=(20, 10),color='deeppink');

#Checking data shape after outlier removal
df.shape

df

df.rename(columns = {'Age-standardised diabetes prevalence':'ASDP'}, inplace = True)

df.rename(columns = {'Lower 95% uncertainty interval':'Lower_95_uncertainty'}, inplace = True)

df.rename(columns = {'Upper 95% uncertainty interval':'Upper_95_uncertainty'}, inplace = True)

df.head()

print(df.Upper_95_uncertainty.value_counts())
fig, ax = plt.subplots(figsize=(5,4))
name = ["Disease", "No_Disease"]
ax = df.Upper_95_uncertainty.value_counts().plot(kind='bar')
ax.set_title("Upper_95_uncertainty", fontsize = 13, weight = 'bold')
ax.set_xticks(range(len(name)))
ax.set_xticklabels (name)


    
plt.tight_layout()

plt.figure(figsize=(20,5))
sns.barplot(x = "Upper_95_uncertainty",y = "ISO", data = df, palette = "rainbow")
plt.xticks(rotation=45)
plt.show()

#check correlation between variables
sns.set(style="darkgrid") 
plt.rcParams['figure.figsize'] = (15, 10) 
sns.heatmap(df.corr(), annot = True, linewidths=.5, cmap="Blues")
plt.title('Corelation Between Variables', fontsize = 30)
plt.show()

from ydata_profiling import ProfileReport

profile = ProfileReport(df,title="Diabetes EDA",html={'style':{'full_width':True}})
profile.to_notebook_iframe()

#pre-processing
from sklearn.exceptions import DataDimensionalityWarning
#encode object columns to integers
from sklearn import preprocessing
from sklearn.preprocessing import OrdinalEncoder

for col in df:
    if df[col].dtype =='object':
        df[col]=OrdinalEncoder().fit_transform(df[col].values.reshape(-1,1))
df

class_label =df['Upper_95_uncertainty']
df = df.drop(['Upper_95_uncertainty'], axis =1)
df = (df-df.min())/(df.max()-df.min())
df['Upper_95_uncertainty']=class_label
df

#pre-processing
diabetes_data = df.copy()
le = preprocessing.LabelEncoder()
Country = le.fit_transform(list(diabetes_data["Country/Region/World"])) # country
ISO = le.fit_transform(list(diabetes_data["ISO"])) # country code as from 0 to 199
Sex = le.fit_transform(list(diabetes_data["Sex"])) # gender (1 = male; 0 = female)
Year = le.fit_transform(list(diabetes_data["Year"])) # distinct years
ASDP = le.fit_transform(list(diabetes_data["ASDP"])) # Age-standardised diabetes prevalence
Lower_95_uncertainty = le.fit_transform(list(diabetes_data["Lower_95_uncertainty"])) # lower 95% uncertainty level
Upper_95_uncertainty = le.fit_transform(list(diabetes_data["Upper_95_uncertainty"])) # upper 95% uncertainty level

x = list(zip(Country, ISO, Sex, Year, ASDP, Lower_95_uncertainty))
y = list(Upper_95_uncertainty)

# Predictive analytics model development by comparing different Scikit-learn classification algorithms
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate
from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error

accu = 0
for i in range(0,14000):
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = .80, random_state = i)
    mod = LinearRegression()
    mod.fit(x_train,y_train)
    y_pred = mod.predict(x_test)
    tempacc = r2_score(y_test,y_pred)
    if tempacc> accu:
        accu= tempacc
        best_rstate=i

print(f"Best Accuracy {accu*100} found on randomstate {best_rstate}")

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor

models = [SVC(), RandomForestClassifier()]

model_names = ["SVM", "RF"]

score= []
mean_abs_e=[]
mean_sqr_e=[]
root_mean_e=[]
r2=[]

for m in models:
    m.fit(x_train,y_train)
    print("Score of", m, "is:", m.score(x_train,y_train))
    score.append(m.score(x_train,y_train))
    predm=m.predict(x_test)
    print("\nERROR:")
    print("MEAN ABSOLUTE ERROR: ",mean_absolute_error(y_test,predm))
    mean_abs_e.append(mean_absolute_error(y_test,predm))
    print("MEAN SQUARED ERROR: ", mean_squared_error(y_test,predm))
    mean_sqr_e.append(mean_squared_error(y_test,predm))
    print("ROOT MEAN SQUARED ERROR :",np.sqrt(mean_squared_error(y_test,predm)))
    root_mean_e.append(np.sqrt(mean_squared_error(y_test,predm)))
    print("R2 SCORE: ", r2_score(y_test,predm))
    r2.append(r2_score(y_test,predm))
    print("**********************************************************************************************************")
    print('\n\n')

import numpy as np

print(np.isnan(x_train).sum())
print(np.isnan(y_train).sum())

from sklearn.model_selection import KFold

mean_score=[]
STD=[]
cv = KFold(n_splits=5, shuffle=True, random_state=42)
for m in models:
    CV=cross_val_score(m, x_train, y_train, cv=cv, scoring="r2")
    print("SCORE OF",m,"Is as follows...")
    print("SCORE IS:", CV)
    print("MEAN OF SCORE is :", CV.mean())
    mean_score.append(CV.mean())
    print("Standard Deviation :", CV.std())
    STD.append(CV.std())
    print("**************************************************************************************************")
    print("\n\n")

Regression_result = pd.DataFrame({"MODEL": model_names,
                                  "SCORE": score,
                                  "CV_mean_score": mean_score,
                                  "CV_STD": STD,
                                  "MBE": mean_abs_e,
                                  "MSE": mean_sqr_e,
                                  "RMSE": root_mean_e,
                                  "R2":r2 
                                 })
Regression_result.sort_values(by="CV_mean_score", ascending=False)

metrics_list = ["SCORE", "CV_mean_score", "CV_STD", "MBE", "MSE", "RMSE", "R2"]

for metric in metrics_list:
    Regression_result.sort_values(by=metric).plot.bar("MODEL", metric, color = "orange")
    plt.title(f"MODEL by {metric}")
    plt.show()

rf = RandomForestRegressor(random_state=42)

rf.fit(x_train, y_train)

predm=rf.predict(x_test)

predm

#saving the file
import pickle
filename = 'savedmodel.sav'
pickle.dump(rf, open(filename,'wb'))

load_model = pickle.load(open(filename, 'rb'))

os.getcwd()

"""Does the average age-standardized death rate (ASDP) vary by sex?"""

# Create a barplot to compare ASDP by sex
sns.barplot(x="Sex", y="ASDP", data=df)

"""How has ASDP changed over the years?"""

# Create a lineplot to visualize ASDP over time
sns.lineplot(x="Year", y="ASDP", data=df)

"""Which countries have the highest and lowest ASDP?"""

# Create a boxplot to visualize ASDP distribution by country
sns.boxplot(x="Country/Region/World", y="ASDP", data=df)

"""Can we predict the ASDP for a given country based on its sex and year?"""

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[["Sex", "Year"]], df["ASDP"], test_size=0.2, random_state=42)

# Train a support vector regression model
svr = SVR(kernel="linear")
svr.fit(X_train, y_train)

# Predict the ASDP for the testing set and calculate the mean squared error
y_pred = svr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean squared error:", mse)

# Create a scatterplot to visualize the relationship between ASDP and Lower_95_uncertainty per capita
sns.scatterplot(x="Lower_95_uncertainty", y="ASDP", data=df)

