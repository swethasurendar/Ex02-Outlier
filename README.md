# Ex02-Outlier

You are given bhp.csv which contains property prices in the city of banglore, India. You need to examine price_per_sqft column and do following,

(1) Remove outliers using IQR 

(2) After removing outliers in step 1, you get a new dataframe.

(3) use zscore of 3 to remove outliers. This is quite similar to IQR and you will get exact same result

(4) for the data set height_weight.csv find the following

    (i) Using IQR detect weight outliers and print them

    (ii) Using IQR, detect height outliers and print them
    
# EXPLANATION
An Outlier is an observation in a given dataset that lies far from the rest of the observations. That means an outlier is vastly larger or smaller than the remaining values in the set. An outlier is an observation of a data point that lies an abnormal distance from other values in a given population. (odd man out).Outliers badly affect mean and standard deviation of the dataset. These may statistically give erroneous results.Most machine learning algorithms do not work well in the presence of outlier. So it is desirable to detect and remove outliers.Outliers are highly useful in anomaly detection like fraud detection where the fraud transactions are very different from normal transactions.

# ALGORITHM
STEP 1
Read the given Data

STEP 2
Get the information about the data

STEP 3
Detect the Outliers using IQR method and Z score

STEP 4
Remove the outliers

STEP 5
Plot the datas using Box Plot

# CODE
(1) & (2) Examine price_per_sqft column and use IQR to remove outliers and create new dataframe

import pandas as pd
import numpy as np
import seaborn as sns

df = pd.read_csv("C:\Users\chief\OneDrive\Documents\Ex02-Outlier\bhp.csv")
df

df.head()

df.describe()

df.info()

df.isnull().sum()

df.shape

sns.boxplot(x="price_per_sqft",data=df)
q1 = df['price_per_sqft'].quantile(0.25)
q3 = df['price_Aper_sqft'].quantile(0.75)
print("First Quantile =",q1,"\nSecond Quantile =",q3)

IQR = q3-q1
ul = q3+1.5*IQR
ll = q1-1.5*IQR

df1 =df[((df['price_per_sqft']>=ll)&(df['price_per_sqft']<=ul))]
df1

df1.shape

sns.boxplot(x="price_per_sqft",data=df1)
(3) Examine price_per_sqft column and use zscore of 3 to remove outliers.
from scipy import stats

z = np.abs(stats.zscore(df['price_per_sqft']))
df2 = df[(z<3)]
df2

print(df2.shape)
sns.boxplot(x="price_per_sqft",data=df2)
(4)(i) For the data set height_weight.csv detect weight outliers using IQR method
df3 = pd.read_csv("C:\Users\chief\OneDrive\Documents\Ex02-Outlier\height_weight.csv")
df3

df3.head()

df3.info()

df3.describe()

df3.isnull().sum()

df3.shape
sns.boxplot(x="weight",data=df3)

q1 = df3['weight'].quantile(0.25)
q3 = df3['weight'].quantile(0.75)
print("First Quantile =",q1,"\nSecond Quantile =",q3)

IQR = q3-q1
ul = q3+1.5*IQR
ll = q1-1.5*IQR

df4 =df3[((df3['weight']>=ll)&(df3['weight']<=ul))]
df4

df4.shape

sns.boxplot(x="weight",data=df4)
(4)(ii) For the data set height_weight.csv detect height outliers using IQR method
sns.boxplot(x="height",data=df3)

q1 = df3['height'].quantile(0.25)
q3 = df3['height'].quantile(0.75)
print("First Quantile =",q1,"\nSecond Quantile =",q3)

IQR = q3-q1
ul = q3+1.5*IQR
ll = q1-1.5*IQR

df5 =df3[((df3['height']>=ll)&(df3['height']<=ul))]
df5

df5.shape

sns.boxplot(x="height",data=df5)

# OUTPUT
(1)(2) Examine price_per_sqft column and use IQR to remove outliers and create new dataframe
 
# Dataset
![image](https://github.com/swethasurendar/Ex02-Outlier/assets/133625914/73e3e205-08e0-4895-a907-6e395da136b7)

# Dataset Head
![image](https://github.com/swethasurendar/Ex02-Outlier/assets/133625914/750d01c5-f767-48f8-9e6d-b712f5667a7f)

# Dataset Info
![image](https://github.com/swethasurendar/Ex02-Outlier/assets/133625914/e38025c1-b3d7-4e23-bce6-a083ec85e737)

# Dataset Describe
![image](https://github.com/swethasurendar/Ex02-Outlier/assets/133625914/0bbda44e-b72e-48e6-9c90-386ab4946d17)

# Null Values!
![image](https://github.com/swethasurendar/Ex02-Outlier/assets/133625914/a0236304-02b3-4d83-b10f-a8199e29a571)

# Dataset Shape
![image](https://github.com/swethasurendar/Ex02-Outlier/assets/133625914/5b9f4837-88d2-4210-8ddc-b6a726610f0c)

# Box plot of price_per_sqft column with outliers
![image](https://github.com/swethasurendar/Ex02-Outlier/assets/133625914/f09f9504-c8e1-4e43-8521-832bdba65325)

# price_per_sqft - Dataset after removing outliers
![image](https://github.com/swethasurendar/Ex02-Outlier/assets/133625914/a46c13c5-fe04-4f32-97d6-6bc05d56062e)

# price_per_sqft - Shape of Dataset after removing outliers
![image](https://github.com/swethasurendar/Ex02-Outlier/assets/133625914/f8049131-e76f-4e57-8ba9-6e8e12b863b8)

# Box Plot of price_per_sqft column without outliers
![image](https://github.com/swethasurendar/Ex02-Outlier/assets/133625914/807d7837-7e3f-43c7-954c-26b2aa00e433)

# (3) Examine price_per_sqft column and use zscore of 3 to remove outliers.
# Dataset after removal of outlier using z score
![image](https://github.com/swethasurendar/Ex02-Outlier/assets/133625914/df67cbfd-0484-4be3-bacf-d2142778ab58)
# Shape of Dataset after removal of outlier using z score
![image](https://github.com/swethasurendar/Ex02-Outlier/assets/133625914/f70b90d8-5303-4895-9baa-163a95ec255d)
# price_per_sqft column after removing outliers
![image](https://github.com/swethasurendar/Ex02-Outlier/assets/133625914/7b49d61d-b907-4fd8-a972-68564111984d)

# (4) For the data set height_weight.csv detect weight and height outliers using IQR method
# Dataset
![image](https://github.com/swethasurendar/Ex02-Outlier/assets/133625914/848b09f4-5ef7-4978-86ef-d64dd8f68727)
# Dataset Heading
![image](https://github.com/swethasurendar/Ex02-Outlier/assets/133625914/e76ab898-b1c5-4835-b193-11b21370916d)
# Null Values
![image](https://github.com/swethasurendar/Ex02-Outlier/assets/133625914/961f1580-f74f-44c6-bceb-19cc68f4a273)

# Datasete Shape
![image](https://github.com/swethasurendar/Ex02-Outlier/assets/133625914/673440c2-4f6a-4882-a424-d5a4983a0b48)
# Weight - With outliers
![image](https://github.com/swethasurendar/Ex02-Outlier/assets/133625914/6cca470f-abb9-4a9a-9a6f-6b71b37ba2c2)
# Weight - Dataset after removing Outliers using IQR method
![image](https://github.com/swethasurendar/Ex02-Outlier/assets/133625914/e3344d01-d47f-42c7-8703-8926bf1f1f83)
# Weight - Shape of Dataset after removing Outliers using IQR method
![image](https://github.com/swethasurendar/Ex02-Outlier/assets/133625914/7a6fa21e-9a4c-4c8f-bb2f-d7927c2ca653)

# Weight - Without Outliers using IQR method
![image](https://github.com/swethasurendar/Ex02-Outlier/assets/133625914/5d96cd43-8167-4bed-874a-cfeb30e5bc8a)
# Height - With outliers
![image](https://github.com/swethasurendar/Ex02-Outlier/assets/133625914/be790971-8e0c-4b46-ab93-6a30a39c3f6f)
# Height - Dataset after removing Outliers using IQR method
![image](https://github.com/swethasurendar/Ex02-Outlier/assets/133625914/aacef06b-5670-4576-8c67-fd014ac21572)
# Height - Shape of Dataset after removing Outliers using IQR method
![image](https://github.com/swethasurendar/Ex02-Outlier/assets/133625914/dfbf6601-c00e-401b-a43b-e14a49637be3)
# Height - Without Outliers using IQR method
![image](https://github.com/swethasurendar/Ex02-Outlier/assets/133625914/f3636bc3-4593-4ac1-8a41-ae9bd0a3f853)
# ![image](https://github.com/swethasurendar/Ex02-Outlier/assets/133625914/13462984-78bd-4dd5-9c13-da97e19c45b2)

# RESULT
The given datasets are read and outliers are detected and are removed using IQR and z-score methods. And print them
