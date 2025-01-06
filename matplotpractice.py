import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from seaborn import rugplot

#reading the data
data = pd.read_csv(r"D:\Project\GradientBoost\dataset\train.csv")

#getting information of the data
print(data.head())
print(data.describe())
print(data.info())

#extracts the numeric and categorical columns
list_of_numeric_cols = data.select_dtypes(include=['int64','float64']).columns
list_of_categorical_cols = data.select_dtypes(include=['object']).columns
print(list_of_numeric_cols)
print(list_of_categorical_cols)

#checking the null values
print(data.isnull().sum())

#filling the null values with the mode
for cat in list_of_categorical_cols:
    print(f"{cat}: {data[cat].unique()}")
    data[cat] = data[cat].fillna(data[cat].mode().iloc[0])

#again checking null values
print(data.isnull().sum())


#plotting the counts of categorical data
fig, axes = plt.subplots(figsize=(50,4), nrows=1, ncols=6)
for i, cat in enumerate(list_of_categorical_cols):
    sns.countplot(data[cat], ax=axes[i])
    axes[i].tick_params(axis='y', labelsize=7)
plt.show()


#Getting distribution of age based on risks
good_df = data.loc[data["Risk"] == "good"]["Age"]
bad_df = data.loc[data["Risk"] == "bad"]["Age"]

#Plotting the distributions
sns.set_style("dark")
fig, axes = plt.subplots(figsize=(10,4),nrows=1,ncols=3,dpi=100)
axes[0].set_title("Good")
sns.distplot(good_df,color="Orange",ax=axes[0])
axes[1].set_title("Bad")
sns.distplot(bad_df,color="Blue",ax=axes[1])
axes[2].set_title("General distribution")
sns.distplot(data["Age"],color="Green",ax=axes[2])
plt.tight_layout()
plt.show()

#Creating a box plot between age and credit amount
interval = (18, 25, 35, 60, 120)
cats = ['Student', 'Young', 'Adult', 'Senior']
data["age_cat"] = pd.cut(data["Age"],interval,labels=cats)
sns.boxplot(x="age_cat", y="Credit amount", hue="Risk", data=data)
plt.show()


#Housing attribute analysis
fig, axes = plt.subplots(figsize=(10,5),nrows=1,ncols=2)
df_good = data.loc[data["Risk"] == 'good']["Housing"]
df_bad = data.loc[data["Risk"] == 'bad']["Housing"]
sns.countplot(data=data,y="Housing",hue="Risk", ax=axes[0])
sns.violinplot(data=data,x="Housing",y="Credit amount",hue="Risk",split=True, ax=axes[1])
plt.show()

#Gender attribute analysis
fig, axes = plt.subplots(figsize=(10,5),nrows=1,ncols=2)
sns.countplot(data=data,y="Sex",hue="Risk",ax=axes[0])
sns.boxplot(data=data,x="Sex",y="Credit amount",hue='Risk',ax=axes[1])
plt.show()

#Job attribute analysis
fig, axes = plt.subplots(figsize=(20,5),nrows=1,ncols=3)
sns.countplot(data=data,y="Job",hue="Risk",ax=axes[0])
sns.boxplot(data=data,x="Job",y="Credit amount",hue="Risk",ax=axes[1])
sns.violinplot(data=data,x="Job",y="Credit amount",hue="Risk",ax=axes[2],split=True)
plt.show()

#distribution of credit amount
good_cred = (data.loc[data["Risk"] == "good"]["Credit amount"])
bad_cred = (data.loc[data["Risk"] == "bad"]["Credit amount"])
sns.distplot(good_cred,hist=False,label="good")
sns.rugplot(good_cred)
sns.distplot(bad_cred,color='Orange',hist=False,label="bad")
sns.rugplot(bad_cred)
plt.legend()
plt.show()

#Saving accounts attribute
fig, axes = plt.subplots(figsize=(20,5),nrows=1,ncols=3)
sns.countplot(data=data,y="Saving accounts",hue="Risk",ax=axes[0])
sns.boxplot(data=data,x="Saving accounts",y="Credit amount",hue="Risk",ax = axes[1])
sns.boxplot(data=data,x="Saving accounts",y="Age",hue="Risk",ax=axes[2])
plt.show()

#purpose
fig, axes = plt.subplots(figsize=(20,5),nrows=1,ncols=2)
sns.countplot(data=data,y="Purpose",hue="Risk",ax=axes[0])
sns.boxplot(data=data,x="Purpose",y="Credit amount",hue="Risk",ax = axes[1])
plt.show()

#duration