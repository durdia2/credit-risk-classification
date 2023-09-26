
ANALYSIS----------------

A logistics regression model was used as the best tool for our machine learning model. It is used to widely predict the probability of the target variable in calssification problems.
Whe splitting the data in [Split the Data into Training and Testing] we can see that the data is highly imbalanced. There are way more healthy loans[0] than there are non-healthy loans[1].

#code
y.value_counts()

# output
0    75036
1     2500
Name: loan_status, dtype: int64

In the confusion matrix in [Create a LRM w/ Original Imbalanced Data]:
out of the healthy loans, it predicted 18,663 and healthy loans and 105 as non healthy
out of the non healthy loans it predicted 56 healthy loans and 563 non healthy loans.

We used random over sampler to have a better accuracy score and help the model catch more mistakes. It adds more copies of the minority class to create a more balanced dataset.

# code
y_oversampled.value_counts()

# output
0    56271
1    56271
Name: loan_status, dtype: int64

It used this to create a Logistic Regression Model with the oversampled data that generated an accuracy score of 99% which is higher than the score with the imbalanced data of 95%. 
The oversamples data set works better by predicting actual healthy loans of 18,649 and non healthy at 116.
It predicted 4 healthy loans from the non-healthy loans and 615 that are actually non-healthy.
Logistic Regression Model- Imbalanced

The logisitics regession model predicted The imbalanced data set healthy loans at 100% versus the non-healthy at 85%.
It made the mistake of predicted the healthy loan (low-risk) as non healthy (high-risk)
a high risk non-healthy loan is classified as a healthy loan (low risk). It had an accuracy of 95% and made a mistake 1% of the time when predicting healthy loans and 9% when it predicted non-healthy loans.

Logistic Regression Model- Balanced
The Logistic Regression model fitted with the OverSampled DataSet predicted healthy loans 100% of the time and predicted non-healthy loans 84% of the time.

This model had a lower probability of making a mistake. According to the classification report, it made the mistake of predicting healthy loans wrong 1% and predicting non-healthy loans wrong 1% of the time. It also generated an accuracy score of 99%.

Overall the logistics regression model with the oversampled data performed way better than the model with the imbalanced data. It had a higher accuracy score and lower chances of giving the wrong prediction, so I would recommend to use this one. 
