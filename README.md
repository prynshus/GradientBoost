# Credit risk prediction using gradient boositng

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
