import pandas as pd
from sklearn.model_selection import train_test_split
from app import helpers as help
#from helpers import help.missing_values,unique_values,correlation_df,help.splitnumcat
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import numpy as np
import pymysql.cursors
#path = os.getcwd()
#path = path.strip('decisiontree')
#os.chdir(path)
# Read data
#print(path)

import configparser

config = configparser.ConfigParser()
config.read('/home/mahidhar/FlaskDeployment/src/config.ini')

host=config['MySQL']['host']
user=config['MySQL']['user']
password=config['MySQL']['password']
db=config['MySQL']['db']
charset=config['MySQL']['charset']



connection = pymysql.connect(host=host,
                             user=user,
                             password=password,
                             db=db,
                             charset=charset,
                             cursorclass=pymysql.cursors.DictCursor)
with connection.cursor() as cursor:
    sql = "SELECT * from bank "
    cursor.execute(sql)
    data = cursor.fetchall()


data = pd.DataFrame(data)
data.head()
data.replace(to_replace=['unknown'],value=np.nan,inplace=True)

#data = pd.read_csv(path+"/data/bank-additional-full.csv", header=0, sep=";",quotechar='"')
print("############################################################################################################################################################")
# Reomove the Customer no from analysis
data =data.drop(['customer_no'],axis = 1)
# print the first six rows of the dataframe
print(data.loc[1 ,:])
print("############################################################################################################################################################")
# printing the data types of the columns

print(data.dtypes)

print("############################################################################################################################################################")


# counting the number of rows in the dataset
print("############################################################################################################################################################")


print("Number of rows in the data set = ", data.shape[0])
print("Number of Columns in the data set = ", data.shape[1])

print("############################################################################################################################################################")


# Drop the Monotonically increasing Attribute
print("############################################################################################################################################################")


for col in data.columns:
    if(data[col].is_monotonic):
        print("Column :", col, ":is Monotonically increasing")

print("############################################################################################################################################################")


# Converting the data type of the attributes
print("############################################################################################################################################################")

data[data.select_dtypes(['object']).columns] = data.select_dtypes(['object']).apply(lambda x: x.astype('category'))
print(data.dtypes)

print("############################################################################################################################################################")


# Describin the dataset
print("############################################################################################################################################################")

print(data.describe(include='all').T)

print("############################################################################################################################################################")


# Class Distribution in the entire data set
print("Class distribution \n", data['y'].value_counts()/data.shape[0])

# Correlation between the attributes
print("############################################################################################################################################################")


corr_df = help.correlation_df(data)
print(corr_df)

print("############################################################################################################################################################")


# Splitting the data into train and validation
print("############################################################################################################################################################")

#X = data.drop('customer_no',axis=1)
X = data.drop('y', axis=1)
print(X.columns)

y = np.array(data['y'])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=123, stratify=y)

print("############################################################################################################################################################")


# Shape of the train and validation
print("############################################################################################################################################################")
print("\n The shape of the X_train  after the split are :", X_train.shape)
print("\n The Shape of the X_validation after the split :", X_val.shape)
print("\n The shape of the y_train  after the split are :", y_train.shape)
print("\n The Shape of the y_validation after the split :", y_val.shape)
print("############################################################################################################################################################")



# Missing Values in the dataset
print("############################################################################################################################################################")
missing_data = help.missing_values(X_train)
print(missing_data)
print("############################################################################################################################################################")

# Percentage for the unique values in the attributes
print("############################################################################################################################################################")
unique_df = help.unique_values(X_train)
print(unique_df)
print("############################################################################################################################################################")
# Splitting the data into num and cat attributes
print("############################################################################################################################################################")
numAttr,catAttr = help.splitnumcat(X_train)
print("\n The numerial attributes are = ",list(numAttr))
print("\n The categorical attributes are = ",list(catAttr))
print("############################################################################################################################################################")
# Observing the distribution of the data using box plots
plt.figure(figsize=(18,9))
plt.title("Distribution of all variables")
plt.ylabel("Range of values on Strandard normal scale")
plt.xlabel("Variables")
X_train.boxplot(rot=90)
path = help.path
plt.savefig(path+"/visualization/boxplots.png")

