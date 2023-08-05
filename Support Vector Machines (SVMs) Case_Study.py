#!/usr/bin/env python
# coding: utf-8

# <span style="font-size: 30px; color: Blue; font-weight: bold;">
# Customer churn prediction for a telecom company.
# </span>
# 

# In[2]:


# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

# Load the dataset
df = pd.read_csv('Telco-Customer-Churn.csv')

# Display the first few rows of the dataset
df.head()


# In[3]:


df.shape


# <span style="font-size: 18px;">
# We have a dataset with various information about customers of the telecom company. Some of the features in our dataset include:<br>
# 
# - gender: The customer's gender (Male, Female)
# - SeniorCitizen: Whether the customer is a senior citizen or not (1, 0)
# - Partner: Whether the customer has a partner or not (Yes, No)
# - Dependents: Whether the customer has dependents or not (Yes, No)
# - tenure: Number of months the customer has stayed with the company
# - PhoneService: Whether the customer has a phone service or not (Yes, No)
# - MultipleLines: Whether the customer has multiple lines or not (Yes, No, No phone service)
# - InternetService: Customer’s internet service provider (DSL, Fiber optic, No)
# - OnlineSecurity: Whether the customer has online security or not (Yes, No, No internet service)
# - OnlineBackup: Whether the customer has online backup or not (Yes, No, No internet service)
# - DeviceProtection: Whether the customer has device protection or not (Yes, No, No internet service)
# - TechSupport: Whether the customer has tech support or not (Yes, No, No internet service)
# - StreamingTV: Whether the customer has streaming TV or not (Yes, No, No internet service)
# - StreamingMovies: Whether the customer has streaming movies or not (Yes, No, No internet service)
# - Contract: The contract term of the customer (Month-to-month, One year, Two year)
# - PaperlessBilling: Whether the customer has paperless billing or not (Yes, No)
# - PaymentMethod: The customer’s payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))
# - MonthlyCharges: The amount charged to the customer monthly
# - TotalCharges: The total amount charged to the customer<br>
# And the target variable is Churn, which indicates whether the customer churned or not (Yes or No).<br>
# 
# Let's first check for missing values in our dataset. Then, we'll proceed with preprocessing the data and building the SVM model.
# </span>
# 

# In[4]:


df.isnull().sum()


# In[5]:


df.info()


# <span style="font-size: 18px;">
# The dataset doesn't have any missing values. However, the "TotalCharges" column is of object type, which means it might contain non-numeric values. Let's convert this column to numeric and handle any potential errors.<br>
# 
# Once we've cleaned up the "TotalCharges" column, we can preprocess the data for our SVM. This will involve encoding categorical variables and scaling numerical variables.
# </span>
# 

# In[6]:


# Convert TotalCharges column to numeric and handle errors
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Check if there are any NaN values after conversion
print("Number of NaN values in TotalCharges column:", df['TotalCharges'].isna().sum())

# Replace NaN values with the mean of the column
df['TotalCharges'].fillna(df['TotalCharges'].mean(), inplace=True)

# Verify if the NaN values have been replaced
print("Number of NaN values in TotalCharges column after replacing with mean:", df['TotalCharges'].isna().sum())


# <span style="font-size: 18px;">
# The "TotalCharges" column contained 11 non-numeric values, which we have replaced with the mean of the column. Now, all our columns are in a suitable format for preprocessing.<br>
# 
# Next, we need to convert our categorical variables into numerical ones. Machine learning models, including SVMs, require input data in numerical format. We'll do this using a technique called label encoding, which assigns a unique numeric value to each category in a categorical variable.<br>
# 
# We'll also drop the "customerID" column, as it's a unique identifier for customers and won't contribute to the model's ability to generalize.<br>
# 
# Finally, we'll scale our numerical features to have zero mean and unit variance. This is important because SVMs are not scale-invariant, meaning that the scale of the features can affect the results. Standardizing the features can make the SVM training more stable and the results more reliable.
# </span>
# 

# In[7]:


# Drop the 'customerID' column
df = df.drop('customerID', axis=1)

# Initialize a LabelEncoder
le = LabelEncoder()

# Iterate over all the values of each column and extract their dtypes
for col in df:
    # Compare if the dtype is object
    if df[col].dtypes=='object':
    # Use LabelEncoder to do the numeric transformation
        df[col]=le.fit_transform(df[col])

# Display the first few rows of the dataset
df.head()


# <span style="font-size: 18px;">
# Now, all the categorical variables in our dataset have been converted to numerical ones. For example, the "gender" column now has values of 0 and 1 instead of "Female" and "Male".<br>
# 
# Next, we'll scale the numerical features. For this task, we'll use the StandardScaler from sklearn, which standardizes features by removing the mean and scaling to unit variance.<br>
# 
# After preprocessing our data, we'll split it into a training set and a test set. Then, we'll be ready to train our SVM model.
# </span>
# 

# In[8]:


# Define the features and the target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the SVC model with a linear kernel
model = SVC(kernel='linear')

# Fit the model to the training data
model.fit(X_train, y_train)

# Predict the labels of the test set
y_pred = model.predict(X_test)

# Print the confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Print the classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# <span style="font-size: 18px;">
# The confusion matrix:<br>
# 
# |   | Predicted Not-Churned | Predicted Churned |
# |---|-----------------------|-------------------|
# | Actual Not-Churned | 934 | 102 |
# | Actual Churned | 156 | 217 |
# 
# </span>
# 

# <span style="font-size: 18px;">
# The SVM model achieved an accuracy of 82% on the test set, which is reasonably good.<br> 
# 
# Let's interpret the confusion matrix and classification report:<br>
# 
# - The confusion matrix:<br>
# 
# Actual Not-Churned: Predicted Not-Churned = 934, Predicted Churned = 102<br>
# Actual Churned: Predicted Not-Churned = 156, Predicted Churned = 217<br>
# 
# This tells us that:<br>
# 
# - 934 true negative predictions: the number of negative cases (class 0 - not churned) correctly identified.
# - 217 true positive predictions: the number of positive cases (class 1 - churned) correctly identified.
# - 156 false negative predictions: the number of positive cases incorrectly identified as negative.
# - 102 false positive predictions: the number of negative cases incorrectly identified as positive.<br>
# 
# - The classification report:<br>
# 
# This shows the precision, recall, and F1-score for each class, as well as the overall accuracy of the model.<br> 
# 
# For the not-churned class (class 0), the model has a precision of 86%, recall of 90%, and F1-score of 88%.<br> 
# 
# For the churned class (class 1), the model has a precision of 68%, recall of 58%, and F1-score of 63%.<br> 
# 
# The model has an overall accuracy of 82%.<br>
# 
# The model seems to be doing better at predicting the not-churned class compared to the churned class. This is not uncommon in situations where one class is more prevalent than the other, as is the case with our dataset.<br> 
# 
# This concludes our customer churn prediction task. Please note that in a real-world scenario, we might want to further tune the model or try different models to improve the performance, especially for the minority class.
# </span>
# 

# In[ ]:




