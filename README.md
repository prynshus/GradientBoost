# Credit risk prediction using gradient boosting

This project delves into the comprehensive analysis of credit risk data to accurately predict the likelihood of loan defaults. By leveraging the power of gradient boosting algorithms, we aim to model complex relationships within the data, ensuring robust and reliable predictions. Our primary objective is to optimize the model's accuracy, thereby enhancing risk assessment processes and enabling financial institutions to make informed lending decisions with greater confidence.

## Data Analysis
The credit risk data has 1000 rows with 10 columns/features.
The first 5 tuples look like this.

![output_1.png](Outputs/output_1.png)

Further using the describe() and info() method we get this.

![output_2.png](Outputs/output_2.png)

Further we can print out the list of categorical and numerical columns. Print out the number of missing values in each columns.
Also printing out the categories of all the categorical columns.

![output_3.png](Outputs/output_3.png)

Now to deal with the NULL values. We first saw that its a categorical data so filling the nulls with maximum frequency in a particular column is fine to do. In other words getting the mode of the column containing NAN and filling nulls with the mode value.

After doing this we get this.

![output_4.png](Outputs/output_4.png)

## Visualisation of the data
1. First we will see the count of each categories in the categorical columns.

![CountPlots.png](Visualisations/CountPlots.png)

2. Then we get the distribution of age.

![Age.png](Visualisations/Age.png)

3. We will add age categorical column i.e. for 18-25 -> young, 25-35 -> adult, etc and then we create a box plot of this column with the credit amount.

![Cat_age.png](Visualisations/Cat_age.png)

Here we can see some outliers and the median line of each category.

4. Next we will look into the housing attribute.

![Housing.png](Visualisations/Housing.png)

5. The gender/Sex attribute.

![Gender.png](Visualisations/Gender.png)

6. The job attribute.

![Job.png](Visualisations/Job.png)

7. Now we will see the distribution of credit amount.

![Credit_dist.png](Visualisations/Credit_dist.png)

8. Saving account attribute.

![Saving_acc.png](Visualisations/Saving_acc.png)

9. Purpose attribute.

![purpose.png](Visualisations/purpose.png)

10. Duration attribute.

![duration.png](Visualisations/duration.png)

11. Checking account attribute.

![Checking_acc.png](Visualisations/Checking_acc.png)

## One hot encoding

One-Hot Encoding is a method used in machine learning and data preprocessing to convert categorical data into a binary format that can be easily processed by algorithms.
To do this we import a file named as oneHotEncoding.py where we defined our function to convert categories in binary format.
Then we pass our data to this function and the resultant new columns are this.

['Sex_female', 'Sex_male', 'Housing_free', 'Housing_own', 'Housing_rent', 'Saving accounts_little', 'Saving accounts_moderate', 'Saving accounts_quite rich', 'Saving accounts_rich', 'Checking account_little', 'Checking account_moderate', 'Checking account_rich', 'Purpose_business', 'Purpose_car', 'Purpose_domestic appliances', 'Purpose_education', 'Purpose_furniture/equipment', 'Purpose_radio/TV', 'Purpose_repairs', 'Purpose_vacation/others', 'Risk_bad', 'Risk_good', 'age_cat_Student', 'age_cat_Young', 'age_cat_Adult', 'age_cat_Senior']

## HeatMap

Now we will create a heatmap which correlates each column with other columns.

The heatmap looks like this.

![Heatmap.png](Visualisations/Heatmap.png)

## Gradient boost

Gradient Boosting is an ensemble learning technique that combines multiple weak learners (typically shallow decision trees) to build a strong predictive model. It is based on the principle of sequential model optimization and uses gradient descent to minimize a loss function.

In our case we are using XGBoost which is also called extreme gradient boosting.

In the GradientBoost.py file first we create X and y dataframes where X is matrix in which target variable "Risk_good" is dropped and y is vector in which this target variable is added.

Now to split our data X and y in the ratio of 0.75 as training set and 0.25 as testing set we use sklearn module's "train_test_split" function.

Next we import XGBClassifier since in our case we have to classify the dataset.
We create an instance of XGBClassifier with a parameter called random state to ensure reproducibility.
Then we pass/fit our dataset in this instance to train the model.
Next we pass our test dataset to get the required results.

Using the accuracy metrics we get an accuracy of 68.8%

And the confusion matrix is:

![Conf_Matrix.png](Visualisations/Conf_Matrix.png)

Means out of 250 test samples 51 is predicted good instead of bad and 21 is predicted bad instead of good.

## Future improvement
We can use grid search and try for different parameteres to get good accuracy.