import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from seaborn import rugplot
from oneHotEncoding import one_hot_encoder

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
fig, axes = plt.subplots(figsize=(20,10), nrows=2, ncols=3)
colour = ["Red","Blue","Green","Yellow","Orange","Pink"]
k = 0
for i in range(2):
    for j in range(3):
        cat = list_of_categorical_cols[k]
        sns.set_style("dark")
        sns.countplot(data[cat], ax=axes[i][j], color=colour[k])
        k += 1
plt.tight_layout()
fig.savefig(r"D:\Project\GradientBoost\Visualisations\CountPlots.png")


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
fig.savefig(r"D:\Project\GradientBoost\Visualisations\Age.png")

#Creating a box plot between age and credit amount
fig = plt.figure(figsize=(8,8))
axes = fig.add_axes([0.1,0.1,0.9,0.9])
interval = (18, 25, 35, 60, 120)
cats = ['Student', 'Young', 'Adult', 'Senior']
data["age_cat"] = pd.cut(data["Age"],interval,labels=cats)
sns.boxplot(x="age_cat", y="Credit amount", hue="Risk", data=data,ax=axes,color="Blue")
fig.savefig(r"D:\Project\GradientBoost\Visualisations\Cat_age.png")


#Housing attribute analysis
fig, axes = plt.subplots(figsize=(10,5),nrows=1,ncols=2)
df_good = data.loc[data["Risk"] == 'good']["Housing"]
df_bad = data.loc[data["Risk"] == 'bad']["Housing"]
sns.countplot(data=data,y="Housing",hue="Risk", ax=axes[0],color="Blue")
sns.violinplot(data=data,x="Housing",y="Credit amount",hue="Risk",split=True, ax=axes[1],color="Blue")
fig.savefig(r"D:\Project\GradientBoost\Visualisations\Housing.png")

#Gender attribute analysis
fig, axes = plt.subplots(figsize=(10,5),nrows=1,ncols=2)
sns.countplot(data=data,y="Sex",hue="Risk",ax=axes[0],color="Red")
sns.boxplot(data=data,x="Sex",y="Credit amount",hue='Risk',ax=axes[1],color="Red")
fig.savefig(r"D:\Project\GradientBoost\Visualisations\Gender.png")

#Job attribute analysis
fig, axes = plt.subplots(figsize=(15,5),nrows=1,ncols=3)
sns.countplot(data=data,y="Job",hue="Risk",ax=axes[0])
sns.boxplot(data=data,x="Job",y="Credit amount",hue="Risk",ax=axes[1])
sns.violinplot(data=data,x="Job",y="Credit amount",hue="Risk",ax=axes[2],split=True)
fig.savefig(r"D:\Project\GradientBoost\Visualisations\Job.png")

#distribution of credit amount
fig = plt.figure(figsize=(6,6))
good_cred = (data.loc[data["Risk"] == "good"]["Credit amount"])
bad_cred = (data.loc[data["Risk"] == "bad"]["Credit amount"])
sns.distplot(good_cred,hist=False,label="good")
sns.distplot(bad_cred,color='Orange',hist=False,label="bad")
plt.legend()
fig.savefig(r"D:\Project\GradientBoost\Visualisations\Credit_dist.png")

#Saving accounts attribute
fig, axes = plt.subplots(figsize=(15,5),nrows=1,ncols=3)
sns.countplot(data=data,y="Saving accounts",hue="Risk",ax=axes[0],color="Green")
sns.boxplot(data=data,x="Saving accounts",y="Credit amount",hue="Risk",ax = axes[1],color="Green")
sns.boxplot(data=data,x="Saving accounts",y="Age",hue="Risk",ax=axes[2],color="Green")
fig.savefig(r"D:\Project\GradientBoost\Visualisations\Saving_acc.png")

#purpose
fig, axes = plt.subplots(figsize=(15,5),nrows=1,ncols=2)
sns.countplot(data=data,y="Purpose",hue="Risk",ax=axes[0],color="Orange")
sns.boxplot(data=data,x="Purpose",y="Credit amount",hue="Risk",ax = axes[1],color="Orange")
fig.savefig(r"D:\Project\GradientBoost\Visualisations\purpose.png")

#duration
fig, axes = plt.subplots(figsize=(10,20),nrows=3,ncols=1)
sns.countplot(data=data,x="Duration",hue="Risk",ax=axes[0])
sns.pointplot(data=data,x="Duration",y="Credit amount",hue="Risk",ax=axes[1])

good_df = data.loc[data["Risk"] == "good"]["Duration"]
bad_df = data.loc[data["Risk"] == "bad"]["Duration"]
sns.distplot(good_df,label="good",ax=axes[2],hist=False)
sns.distplot(bad_df,label="bad",ax=axes[2],hist=False)
fig.savefig(r"D:\Project\GradientBoost\Visualisations\duration.png")

#checking account
fig, axes = plt.subplots(figsize=(10,20),nrows=3,ncols=1)
sns.countplot(data=data,x="Checking account",hue="Risk",ax=axes[0],color="Yellow")
sns.violinplot(data=data,x="Checking account",y="Age",hue="Risk",split=True,ax=axes[1],color="Yellow")
sns.boxplot(data=data,x="Checking account",y="Credit amount",hue="Risk",ax=axes[2],color="Yellow")
fig.savefig(r"D:\Project\GradientBoost\Visualisations\Checking_acc.png")

data, cols = one_hot_encoder(data)
print(cols)

#correlation
plt.figure(figsize=(14,12))
sns.heatmap(data.astype(float).corr(),linewidths=0.1,vmax=1.0,
            square=True,  linecolor='white', annot=True,annot_kws={"size": 8})
plt.savefig(r"D:\Project\GradientBoost\Visualisations\Heatmap.png")

