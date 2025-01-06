import matplotlib.pyplot as plt
import pandas as pd
from oneHotEncoding import one_hot_encoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier
import seaborn as sns

data = pd.read_csv(r"D:\Project\GradientBoost\dataset\train.csv")

#Just as we did during data analysis
list_of_categorical_cols = data.select_dtypes(include=['object']).columns
for cat in list_of_categorical_cols:
    data[cat] = data[cat].fillna(data[cat].mode().iloc[0])

data, cols = one_hot_encoder(data)
data.drop(columns="Unnamed: 0",inplace=True)
data.drop(columns="Risk_bad",inplace=True)

X = data.drop(columns="Risk_good")
y = data["Risk_good"]

#split into train and test set
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42)

#Creating the classifier
model_xg = XGBClassifier(random_state=2)

model_xg.fit(X_train, y_train)

y_pred = model_xg.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test,y_pred)*100:.2f}%")
sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,fmt='d',cmap="Blues")
plt.show()
