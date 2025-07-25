import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression

#Initializing dataset
dataset=pd.read_excel("Dataset.xlsx")
dataset=dataset.drop('ID',axis=1)
dataset=dataset.fillna(dataset.mean())

#Training TEST
y = dataset.iloc[:, 0].values
X = dataset.iloc[:, 1:29].values
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=0,stratify=y)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Risk Model Building
classifier =  LogisticRegression()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print("Accuracy of the model- ",accuracy_score(y_test, y_pred))
predictions = classifier.predict_proba(X_test)

#Writing Output File
df_prediction_prob = pd.DataFrame(predictions, columns = ['prob_0', 'prob_1'])
df_prediction_target = pd.DataFrame(classifier.predict(X_test), columns = ['predicted_TARGET'])
df_test_dataset = pd.DataFrame(y_test,columns= ['Actual Outcome'])
dfx=pd.concat([df_test_dataset, df_prediction_prob, df_prediction_target], axis=1)
dfx.to_csv("Model_Prediction.xlsx", sep=',', encoding='UTF-8')
print("\n")
print(dfx.head())
