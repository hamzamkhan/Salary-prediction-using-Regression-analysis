import pandas as pd
from scipy import stats
import numpy as np

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from wordcloud import WordCloud, STOPWORDS


############################### HAMZA MUSTAFA KHAN 15K - 2832 ##############################
############################### SARIM BALKHI 15K - 2828 ####################################

df=pd.read_csv('results.csv')

###########################################Exploratory Data Analysis##############################################33
print(df.head())
print(df.shape)
print(df.describe())


print("Skewness: %f" % (df["Salary"].skew()))
print("Kurtosis: %f" % (df["Salary"].kurt()))




print(df.skew())
sns.distplot((df["Salary"]))
df.Salary.describe() #Salary Distribution


sns.boxplot(df['Salary'])
plt.show()


###################### DATA CLEANING #################
df=(df[df.Salary != 0])
df = df.drop_duplicates()



table=pd.crosstab(df.Cities, df.Company, normalize=True)
print(table)


sns.distplot(df['Salary'], fit=norm);
fig = plt.figure()
res = stats.probplot(df['Salary'], plot=plt)


f, ax = plt.subplots(figsize=(10, 10))
cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(df.corr(),linewidths=0.25,vmax=1.0,cmap=cmap,square=True, linecolor='black')
plt.show()


#So yes, Salary is not normalized.

array=df.values
X = df[['Company','Jobtitle','Location']]
print(X)
Y=df['Salary']
#print(Y)

from sklearn import tree
from sklearn.metrics import accuracy_score

X_train, X_test,Y_train, Y_test=train_test_split(X,Y,train_size=0.75, test_size=0.25)

df['Jobtitle'].value_counts().head(12)
df[df['Jobtitle'].str.contains("Senior")].head(2)



def to_spec(row):
    if "Senior" in row['Jobtitle']:
        return 'Higher title'
    if "Head" in row['Jobtitle']:
        return 'Higher title'
    elif "Lead" in row['Jobtitle']:
        return 'Higher title'
    else:
        return 'Not higher title'


df['Higher position'] = df.apply(to_spec, axis=1)
print(df.head())

#######################################################################MODEL EVALUATION

median = np.median(df['Salary'])
df['salary_binary'] = df['Salary'].apply(lambda x: 1 if x > median else 0)


df.loc[:,'Location'] = df.loc[:,'Location'].str.replace('\d+', '')

y = df['salary_binary']

X_dumm = pd.get_dummies(X)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=10**10,solver='lbfgs')
logreg.fit(X_dumm, y)
print(cross_val_score(logreg, X_dumm, y, cv=10))
print(np.mean(cross_val_score(logreg, X_dumm, y, cv=10)))
folds=[1,2,3,4,5,6,7,8,9,10]
sns.lineplot(x=folds, y=cross_val_score(logreg, X_dumm, y, cv=10))
plt.show()



y = df['Salary'].values
X = df['Location'].values
X=le.fit_transform(X) #one hot encoding
print(type(X),type(y))

# Print the dimensions of X and y before reshaping
print("Dimensions of y before reshaping: {}".format(y.shape))
print("Dimensions of X before reshaping: {}".format(X.shape))

# Reshape X and y
y = y.reshape(-1,1)
X = X.reshape(-1,1)

print(type(X),type(y))

# Print the dimensions of X and y after reshaping
print("Dimensions of y after reshaping: {}".format(y.shape))
print("Dimensions of X after reshaping: {}".format(X.shape))

X = X.astype(float)
y=Y.astype(float)

plt.scatter(X,y)
plt.show()


Xtrain,Xtest,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=3)



reg = LinearRegression()

# Create the prediction space
prediction_space = np.linspace(min(X), max(X)).reshape(-1,1)

# Fit the model to the data
reg.fit(Xtrain, y_train)
predictions = reg.predict(Xtest)
print(predictions[0:5]) #Top 5 predicted salaries
# Compute predictions over the prediction space: y_pred
y_pred = reg.predict(prediction_space)

# Print R^2
print(reg.score(X, y))

# Plot regression line
plt.plot(prediction_space, y_pred, color='black', linewidth=3)
plt.show()





#######################################LINEAR REGRESSION FOR JOBTITLE#########################################3

y = df['Salary'].values
X = df['Jobtitle'].values
X=le.fit_transform(X)
print(type(X),type(y))

# Print the dimensions of X and y before reshaping
print("Dimensions of y before reshaping: {}".format(y.shape))
print("Dimensions of X before reshaping: {}".format(X.shape))

# Reshape X and y
y = y.reshape(-1,1)
X = X.reshape(-1,1)

print(type(X),type(y))

# Print the dimensions of X and y after reshaping
print("Dimensions of y after reshaping: {}".format(y.shape))
print("Dimensions of X after reshaping: {}".format(X.shape))

X = X.astype(float)
y=Y.astype(float)

plt.scatter(X,y)
plt.show()


Xtrain,Xtest,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=3)



reg = LinearRegression()

# Create the prediction space
prediction_space = np.linspace(min(X), max(X)).reshape(-1,1)

# Fit the model to the data
reg.fit(Xtrain, y_train)
predictions = reg.predict(Xtest)
print(predictions[0:5]) #Top 5 predicted salaries
# Compute predictions over the prediction space: y_pred
y_pred = reg.predict(prediction_space)

# Print R^2
print(reg.score(X, y))

# Plot regression line
plt.plot(prediction_space, y_pred, color='black', linewidth=3)
plt.show()

##########################################LINEAR REGRESSION USING HIGHER POSITION###################
print(df['Higher position'])
df['Higher position']=le.fit_transform(df['Higher position'])
print(df['Higher position'])
X=df['Higher position'].values
y=df['Salary'].values

y = y.reshape(-1,1)
X = X.reshape(-1,1)
print("Dimensions of y after reshaping: {}".format(y.shape))
print("Dimensions of X after reshaping: {}".format(X.shape))
X = X.astype(int)
y=Y.astype(int)

X_Train,X_Test,yTrain,ytest=train_test_split(X,y,test_size=0.25,random_state=1)
from sklearn.ensemble import RandomForestRegressor




regressor_ = RandomForestRegressor(n_estimators=10,random_state=1)
regressor_.fit(X_Train, yTrain.values.ravel())
predictions_=regressor_.predict(X_Test)
print(predictions_)
print(ytest.values)


#####################################################WORD CLOUD###################################
comment_words = ' '
stopwords = set(STOPWORDS)

# iterate through the csv file
for val in df.Jobtitle:

    # typecaste each val to string
    val = str(val)

    # split the value
    tokens = val.split()

    # Converts each token into lowercase
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()

    for words in tokens:
        comment_words = comment_words + words + ' '

wordcloud = WordCloud(width=500, height=800,
                      background_color='black',
                      stopwords=stopwords,
                      min_font_size=10).generate(comment_words)

# plot the WordCloud image
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()


