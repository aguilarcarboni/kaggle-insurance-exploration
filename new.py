# %% [markdown]
# # **1. Understanding the Problem and Objective:**
# Before diving into the data, we need understand the problem we are trying to solve and the goals of our analysis. This helps in directing our exploration and identifying relevant patterns.For this Health Insurance data,this dataset is about an Insurance company that has provided Health Insurance to its customers. Now we need build a model to predict whether the policyholders (customers) from past year will also be interested in Vehicle Insurance provided by the company.

# %% [markdown]
# # **2. Importing libraries and Loading the Dataset:**

# %% [code] {"execution":{"iopub.status.busy":"2024-01-06T12:53:36.722565Z","iopub.execute_input":"2024-01-06T12:53:36.722953Z","iopub.status.idle":"2024-01-06T12:53:36.733765Z","shell.execute_reply.started":"2024-01-06T12:53:36.722924Z","shell.execute_reply":"2024-01-06T12:53:36.732339Z"}}
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import os


df_test = pd.read_csv('/data/test.csv')
df = pd.read_csv('/data/train.csv')

df.head()
df.info()
df.dtypes

df.shape

df.duplicated().sum()
df.drop_duplicates(inplace=True)
df.shape

missing_values = df.isnull().sum()
missing_values


"""
# # **5. Exploratory Data Analysis(EDA)**

# %% [code] {"id":"FrmFIRZ1rfhe","execution":{"iopub.status.busy":"2024-01-06T12:53:38.194692Z","iopub.execute_input":"2024-01-06T12:53:38.195197Z","iopub.status.idle":"2024-01-06T12:53:38.203089Z","shell.execute_reply.started":"2024-01-06T12:53:38.195152Z","shell.execute_reply":"2024-01-06T12:53:38.201166Z"}}
sns.set(style="whitegrid")

# %% [code] {"id":"wHHcw18xr6EI","executionInfo":{"status":"ok","timestamp":1703244041934,"user_tz":-330,"elapsed":397757,"user":{"displayName":"Rohit Kumar Shaw","userId":"08344271847417693490"}},"outputId":"a0c6d2e5-7987-41d4-da07-ee1c6ff87e4c","execution":{"iopub.status.busy":"2024-01-06T12:53:43.559231Z","iopub.execute_input":"2024-01-06T12:53:43.559682Z","iopub.status.idle":"2024-01-06T12:56:14.84271Z","shell.execute_reply.started":"2024-01-06T12:53:43.559647Z","shell.execute_reply":"2024-01-06T12:56:14.84136Z"}}
sns.pairplot(df)

# %% [code] {"id":"wNef7MWxsO28","executionInfo":{"status":"ok","timestamp":1703159591426,"user_tz":-330,"elapsed":488,"user":{"displayName":"Rohit Kumar Shaw","userId":"08344271847417693490"}},"outputId":"103efd07-e51d-4242-a1a7-4b89944b33ea","execution":{"iopub.status.busy":"2024-01-06T12:56:14.845105Z","iopub.execute_input":"2024-01-06T12:56:14.845575Z","iopub.status.idle":"2024-01-06T12:56:15.281372Z","shell.execute_reply.started":"2024-01-06T12:56:14.845534Z","shell.execute_reply":"2024-01-06T12:56:15.280026Z"}}
plt.figure(figsize=(7, 4))
sns.boxplot(x=df['Annual_Premium'])
plt.title('Distribution of Annual_Premium')
plt.xlabel('Annual_Premium')
plt.ylabel('Frequency')
plt.show()

# %% [code] {"id":"cFwjaTtbtplz","executionInfo":{"status":"ok","timestamp":1703159596130,"user_tz":-330,"elapsed":1361,"user":{"displayName":"Rohit Kumar Shaw","userId":"08344271847417693490"}},"outputId":"6a2934e6-51f1-45aa-c0da-9991c3387654","execution":{"iopub.status.busy":"2024-01-06T12:56:15.282962Z","iopub.execute_input":"2024-01-06T12:56:15.283611Z","iopub.status.idle":"2024-01-06T12:56:16.236305Z","shell.execute_reply.started":"2024-01-06T12:56:15.283563Z","shell.execute_reply":"2024-01-06T12:56:16.235112Z"}}
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], kde=False, bins=10)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# %% [code] {"id":"Dnm2XmvDvsSz","executionInfo":{"status":"ok","timestamp":1703159601104,"user_tz":-330,"elapsed":1155,"user":{"displayName":"Rohit Kumar Shaw","userId":"08344271847417693490"}},"outputId":"6fe2da4c-b309-4b13-f788-ec28b12a9f39","execution":{"iopub.status.busy":"2024-01-06T12:56:16.239824Z","iopub.execute_input":"2024-01-06T12:56:16.24034Z","iopub.status.idle":"2024-01-06T12:56:17.215484Z","shell.execute_reply.started":"2024-01-06T12:56:16.240293Z","shell.execute_reply":"2024-01-06T12:56:17.214102Z"}}
plt.figure(figsize=(10, 6))
sns.histplot(df['Region_Code'], kde=False, bins=12)
plt.title('Distribution of Region_Code')
plt.xlabel('Region_Code')
plt.ylabel('Frequency')
plt.show()

# %% [code] {"id":"cZndUBV2wM5D","executionInfo":{"status":"ok","timestamp":1703159601830,"user_tz":-330,"elapsed":732,"user":{"displayName":"Rohit Kumar Shaw","userId":"08344271847417693490"}},"outputId":"b00923af-e426-45c0-bd0a-fcb982d31d31","execution":{"iopub.status.busy":"2024-01-06T12:56:17.216877Z","iopub.execute_input":"2024-01-06T12:56:17.217398Z","iopub.status.idle":"2024-01-06T12:56:18.614898Z","shell.execute_reply.started":"2024-01-06T12:56:17.217352Z","shell.execute_reply":"2024-01-06T12:56:18.613571Z"}}
plt.figure(figsize=(10, 6))
sns.histplot(df['Vehicle_Age'], kde=False, bins=10)
plt.title('Distribution of Vehicle_Age')
plt.xlabel('Vehicle_Age')
plt.ylabel('Frequency')
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2024-01-06T12:56:18.616414Z","iopub.execute_input":"2024-01-06T12:56:18.616764Z","iopub.status.idle":"2024-01-06T12:56:18.83372Z","shell.execute_reply.started":"2024-01-06T12:56:18.616733Z","shell.execute_reply":"2024-01-06T12:56:18.832116Z"}}
#Checking If training data is Imbalanced
response_data = df['Response'].value_counts()
plt.figure(figsize=(6,6))
fig, ax = plt.subplots()
ax.pie(response_data, labels = [0,1])
ax.set_title('Checking Imbalance in Training Data Or Response')
"""

# %% [markdown]
# # **6. Feature Engineering**

def veh_a(Vehicle_Damage):
  if Vehicle_Damage == 'Yes':
    return 1
  else:
    return 0

df['Vehicle_Damages'] = df['Vehicle_Damage'].apply(veh_a)
df.drop(['Vehicle_Damage'],axis=1)
df = df.drop(['Vehicle_Damage'],axis=1)

df['Vehicle_Age'] = df['Vehicle_Age'].astype('category')
df = pd.get_dummies(df, columns=['Vehicle_Age'])

df['Gender'] = df['Gender'].astype('category')
df = pd.get_dummies(df, columns=['Gender'],drop_first=True)

def veh_a(Vehicle_Damage):
  if Vehicle_Damage == 'Yes':
    return 1
  else:
    return 0

df_test['Vehicle_Damages'] = df_test['Vehicle_Damage'].apply(veh_a)
df_test.drop(['Vehicle_Damage'],axis=1)
df_test = df_test.drop(['Vehicle_Damage'],axis=1)

df_test['Vehicle_Age'] = df_test['Vehicle_Age'].astype('category')
df_test = pd.get_dummies(df_test, columns=['Vehicle_Age'])

df_test['Gender'] = df_test['Gender'].astype('category')
df_test = pd.get_dummies(df_test, columns=['Gender'],drop_first=True)

# Split data

X_train = df[['Age', 'Driving_License', 'Region_Code', 'Previously_Insured', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage', 'Vehicle_Damages', 'Vehicle_Age_1-2 Year','Vehicle_Age_< 1 Year', 'Vehicle_Age_> 2 Years', 'Gender_Male']]
y_train = df['Response']

X_test = df_test[['Age', 'Driving_License', 'Region_Code', 'Previously_Insured', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage', 'Vehicle_Damages', 'Vehicle_Age_1-2 Year','Vehicle_Age_< 1 Year', 'Vehicle_Age_> 2 Years', 'Gender_Male']]

print(y_train)