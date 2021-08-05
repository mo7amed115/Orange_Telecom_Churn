# importing the Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
#-------------------------------------------------
# importing the data .

data = pd.read_csv('data/Orange_Telecom_Churn_Data.csv')
data.head()

# Test for missing data.

data.isna().sum()

#------------------------------------------------------------

# encode the string Data .

encoder = LabelEncoder()
encode_list = ['state' , 'intl_plan' , 'voice_mail_plan'  , 'area_code' , 'churned' ]
for i in encode_list:
    data[i] = encoder.fit_transform(data[i])
data.head()

#-------------------------------------------------
# Plot the all Features vs Churned .

for i in range(len(data.columns)) :
    if data.columns[i] == 'phone_number':
        continue;
    print(f'Viewing {data.columns[i]} vs Churned ')
    sns.barplot(data = data , y = data.columns[i] , x = 'churned')
    plt.show()

#--------------------------------------------------------------------------
# Drop unnecessary Columns and Splitting the data to Features and Target .

X = data.drop(['state', 'account_length','phone_number',
               'total_day_calls', 'total_day_charge', 'total_eve_minutes',
               'total_eve_calls', 'total_eve_charge', 'total_night_minutes',
               'total_night_calls', 'total_night_charge', 'total_intl_minutes',
               'total_intl_calls', 'total_intl_charge','churned'],axis = 1)
y = data.iloc[: , -1]

X.head()

#-----------------------------------------------------------------------------
# Scalling the data .

from sklearn.preprocessing import StandardScaler
import numpy as np
scaler = StandardScaler()
X['number_vmail_messages'] = scaler.fit_transform(np.array(X['number_vmail_messages']).reshape(-1,1))
X['total_day_minutes'] = scaler.fit_transform(np.array(X['total_day_minutes']).reshape(-1,1))

#-----------------------------------------------------------------------------
# Split the data to train and test . 

from sklearn.model_selection import train_test_split
# Splitting the Data to Training and Test
X_train , X_test , y_train , y_test = train_test_split( X ,
                                                        y ,
                                                        test_size = 0.2 , 
                                                        random_state = 0 )

#-----------------------------------------------------------------------------

# Testing Classification Algorithms for use the best Algorithm.

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

models = {'Logestic_Regression' : LogisticRegression() ,
          'KNN' : KNeighborsClassifier() ,
          'Random_Forest_Classifier' : RandomForestClassifier() ,
          'SVC' : SVC() ,
          'Decision_Tree' : DecisionTreeClassifier()
          }

def fit_and_score(models , X_train , X_test , y_train , y_test) :
    model_scores = {}
    model_confusion = {}
    for name , model in models.items() :
        # fitting the data :
        model.fit(X_train , y_train)
        model_scores[name] = model.score(X_test , y_test)
        y_predict = model.predict(X_test)
        model_confusion[name] = confusion_matrix(y_test , y_predict)
    return model_scores , model_confusion


fit_and_score(models = models ,
              X_train = X_train,X_test = X_test,
              y_train = y_train,y_test = y_test )

#-----------------------------------------------------------------------------

# the Best alogorithm accuracy : SVC

from sklearn.svm import SVC
from sklearn.metrics import (classification_report ,
                             recall_score , f1_score ,
                             r2_score , precision_score )

svc = SVC(C=2)
svc.fit(X_train , y_train)
y_pred = svc.predict(X_test)
print("The Score of SVC :", svc.score(X_test , y_test))
print("The Recall score of SVC :", recall_score(y_test , y_pred))
print("The F1 score of SVC :", f1_score(y_test , y_pred))
print("The R2 score of SVC :", r2_score(y_test , y_pred))
print("The precision score of SVC :", precision_score(y_test , y_pred))
print("The clf report for SVC :\n", classification_report(y_test , y_pred))
print("The cm of SVC : \n", sns.heatmap(confusion_matrix(y_test , y_pred) , annot=True ,
                                     fmt = 'd' , cmap = 'YlGnBu' ))
