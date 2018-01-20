import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.linear_model import LogisticRegression 
from sklearn import svm 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict
kfold = KFold(n_splits=10, random_state=22)



df=pd.read_csv("/home/bora3i/Downloads/Datasets/titanic/train.csv",skipinitialspace=True)
#print dfhead()
#print df.isnull().sum()
#print df.columns
frame=pd.DataFrame(df)
age=frame['Age']
survive=frame['Survived']
#print age
#print survive
group=frame.groupby('Sex')['Survived'].count()
print group
##group=frame['Sex'].groupby(frame['Survived'].count())
##print group
frame['named']=0
for i in frame:
    frame['named']=frame.Name.str.extract('([A-Za-z]+)\.')
frame['named'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)
group2=frame.groupby('named')['Age'].count()

print group2

frame.loc[(frame.Age.isnull())&(frame.named=='Mr'),'Age']=40
frame.loc[(frame.Age.isnull())&(frame.named=='Mrs'),'Age']=35
frame.loc[(frame.Age.isnull())&(frame.named=='Master'),'Age']=7
frame.loc[(frame.Age.isnull())&(frame.named=='Miss'),'Age']=25
frame.loc[(frame.Age.isnull())&(frame.named=='Other'),'Age']=46
#print(frame.Age.isnull().any())
##sns.factorplot('Pclass','Survived',col='named',data=frame)
##plt.show()
frame['Embarked'].fillna('S',inplace=True)

##

sns.heatmap(frame.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) 
fig=plt.gcf()
fig.set_size_inches(14,10)
#plt.show()

##
frame['agegroups']=0


frame.loc[(frame['Age']<=16),'agegroups']=0
frame.loc[(frame['Age']>16)&(frame['Age']<=32),'agegroups']=1
frame.loc[(frame['Age']>32)&(frame['Age']<=48),'agegroups']=2

frame.loc[(frame['Age']>48)&(frame['Age']<=64),'agegroups']=3

frame.loc[(frame['Age']>64),'agegroups']=4

age_group=frame.groupby('Age')['Survived'].count()
age_S=pd.crosstab([frame.Age],[frame.Survived],margins=True).style.background_gradient(cmap='summer_r')

#print age_group
#print frame['agegroups']
##f,ax=plt.subplots(1,1,figsize=(18,8))
##sns.countplot('Survived',data=frame[['Survived','Age']])
##ax.set_title('Survived')
##plt.show()                           

#print frame.head()


frame['Fmembers']=0
frame['individaual']=0
frame['Fmembers']=frame['SibSp']+frame['Parch']
f3=frame[['Pclass','Fmembers']]
group4=frame.groupby('Survived')[['Pclass','Fmembers']]
frame['frange']=pd.qcut(frame['Fare'],4)
#print frame['frange']
frame['flimits']=0
frame.loc[frame['Fare']<=7.91,'flimits']=0
          
frame.loc[(frame['Fare']>7.91)&(frame['Fare']<=14.54),'flimits']=1
frame.loc[(frame['Fare']>14.54)&(frame['Fare']<=31),'flimits']=2
          

frame.loc[(frame['Fare']>31)&(frame['Fare']<=512.329),'flimits']=3
          
#print frame['flimits']
frame['Sex'].replace(['female','male'],[1,0],inplace=True)
frame['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)
frame['named'].replace(['Mr','Mrs','Miss','Master','Other'],[0,1,2,3,4],inplace=True)
frame.drop(['PassengerId','Name','Cabin','Ticket','frange','Fare','Age'],axis=1,inplace=True)
#print frame.head()

features=frame[frame.columns[1:]]
lables=frame['Survived']
#features=np.ravel(features)
#lables=np.ravel(lables)
print features.shape
print lables.shape
#print lables
train,test=train_test_split(frame,test_size=0.3,random_state=0,stratify=frame['Survived'])
#print train
#print test
trainx=train[train.columns[1:]]
#print trainx
trainy=train[train.columns[:1]]
#print trainy
testx=test[test.columns[1:]]
testy=test[test.columns[:1]]
#print testy
trainy=np.ravel(trainy)
testy=np.ravel(testy)
f,ax=plt.subplots(3,3,figsize=(12,8))
#guassian naivebayes


gnaive_model=GaussianNB()
gnaive_model.fit(trainx,trainy)
predict_NB=gnaive_model.predict(testx)
GBN_accuracy=metrics.accuracy_score(predict_NB,testy)
val_pred2=cross_val_predict(GaussianNB(),features,lables,cv=10)
val_pred22=cross_val_score(GaussianNB(),features,lables,cv=kfold,scoring='accuracy')
print ("GNB mean cv accuracy :",val_pred22,"accuracy of",GBN_accuracy)
ax[0,0].set_title("conf_matrix of naive bayes ")
print sns.heatmap(confusion_matrix(lables,val_pred2),ax=ax[0,0],annot=True,fmt='2.0f')

#print ("gaussian naive accuracy :",GBN_accuracy)

#logistic regression

log_model=LogisticRegression()
log_model.fit(trainx,trainy)
predict_log=log_model.predict(testx)
acuraccy_log=metrics.accuracy_score(predict_log,testy)
#print ("logistic regression accuracy",acuraccy_log)
val_pred3=cross_val_predict(LogisticRegression(),features,lables,cv=10)
val_pred33=cross_val_score(LogisticRegression(),features,lables,cv=kfold,scoring='accuracy')
print ("log_model mean cv accuracy :",val_pred33,"accuracy of",acuraccy_log)

ax[0,1].set_title("conf_matrix for logistic regression ")
print sns.heatmap(confusion_matrix(lables,val_pred3),ax=ax[0,1],annot=True,fmt='2.0f')

#knearst neighbours


model_knn=KNeighborsClassifier(n_neighbors=9)
model_knn.fit(trainx,trainy)
predict_knn=model_knn.predict(testx)
accuracy_knn=metrics.accuracy_score(predict_knn,testy)
#print accuracy_knn
val_pred4=cross_val_predict(KNeighborsClassifier(n_neighbors=9),features,lables,cv=10)
val_pred44=cross_val_score(KNeighborsClassifier(n_neighbors=9),features,lables,cv=kfold,scoring='accuracy')
print ("knn model with cv mean acc :",val_pred44 ,"accuracy of :",accuracy_knn)

ax[1,0].set_title("conf_matrix for knn")
print sns.heatmap(confusion_matrix(lables,val_pred4),ax=ax[1,0],annot=True,fmt='2.0f')
#desicion trees
model_tree=DecisionTreeClassifier()
model_tree.fit(trainx,trainy)
predict=model_tree.predict(testx)
accuracy_tree=metrics.accuracy_score(predict,testy)
val_pred6=cross_val_predict(DecisionTreeClassifier(),features,lables,cv=10)
val_pred66=cross_val_score(DecisionTreeClassifier(),features,lables,cv=kfold,scoring='accuracy')
print ("desciom trees with cv mean accuracy :",val_pred66,"accuracy :",accuracy_tree)
##ax[2,0].set_title("conf_matrix for desicion trees")
##print sns.heatmap(confusion_matrix(lables,val_pred66),ax=ax[2,0],annot=True,fmt='2.0f')








#svm model with radial basis kernel

svm_model=svm.SVC(kernel='rbf',C=0.1,gamma=0.1)


svm_model.fit(trainx,trainy)

predict_svm=svm_model.predict(testx)

accuracy_svm=metrics.accuracy_score(predict_svm,testy)
#print testx
##print ("svm accuracy",accuracy_svm)
val_pred5=cross_val_predict(svm.SVC(kernel='rbf',C=0.1,gamma=0.1),features,lables,cv=10)
val_pred55=cross_val_score(svm.SVC(kernel='rbf',C=0.1,gamma=0.1),features,lables,cv=kfold,scoring='accuracy')
print ("svm with mean cv acc :",val_pred55,"accuracy of :",accuracy_svm)
ax[1,1].set_title("conf_matrix for svm")
print sns.heatmap(confusion_matrix(lables,val_pred5),ax=ax[1,1],annot=True,fmt='2.0f')

plt.subplots_adjust(hspace=0.2,wspace=0.2)
plt.show()

## asighment description
##we dropped the name column and replaced names by titles {mr, miss,master, mrs} as we used them later to group by titels to the ages into ranges 
##2-we grouped the sibsp column which was the siblings and the parch column which was "parents " into one group called Fmembers which is the total number of family members beacuse family members number affeacted the survival 
##3-we dropped the ticket number column and cabin because they had alot of missing values and their info was useless if to be one hot encoded
##we replaced the missing values in the age column with averged value of the title according to the title according to the "title grouping " we made before 
##we one hot encoded the female and male into zero and one and we filled the missing values in the gender column by female cause the female gender was domminat on the ship 


## we grouped the fare into ranges also to ease the classification into flimits column
##  																	




