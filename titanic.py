import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
titanic=pd.read_csv(r".csv")
print(titanic.head)
print(titanic.info)
print(titanic.shape)
print(titanic.isna().sum())
titanic.drop(columns=['Cabin'],inplace=True)
titanic.drop(columns=['PassengerId'],inplace=True)
titanic['Age']=titanic['Age'].fillna(titanic['Age'].median())
titanic['Embarked']=titanic['Embarked'].mode()[0]
titanic['FamilySize']=titanic['SibSp']+titanic['Parch']
titanic['IsAlone']=(titanic['FamilySize']==0).astype(int)
titanic.drop(columns=['Name','Ticket'],inplace=True)
titanic['Sex']=titanic['Sex'].map({'male':0,'female':1})
titanic['Embarked']=titanic['Embarked'].map({'S':0,'C':1,'Q':2})
y=titanic['Survived']
X=titanic.drop(columns=['Survived'])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=LogisticRegression()
model.fit(X_train,y_train)
prediction=model.predict(X_test)
model=RandomForestClassifier(n_estimators=500,max_depth=10)
model.fit(X_train,y_train)
prediction=model.predict(X_test)
plt.figure()
sns.countplot(data=titanic,x='Survived',hue='Sex')
plt.show()
plt.figure()
sns.countplot(data=titanic,x='Survived',hue='Sex')
plt.show()
sns.countplot(data=titanic,x='Pclass',hue='Survived')
plt.figure()
plt.show()
print(accuracy_score(y_test,prediction))
print(classification_report(y_test,prediction))
print(confusion_matrix(y_test,prediction))
