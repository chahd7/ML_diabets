#importing the dependencies
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm #support vector machine
from sklearn.metrics import accuracy_score


#loading the diabetes dataset to a pandas dataframe
diabetes_dataset = pd.read_csv('diabetes.csv')

#printing the first 5 rows of the data set
diabetes_dataset.head()
#number of rows and columns in data set (row, columns)
diabetes_dataset.shape
#getting the statistical meansures of the data
diabetes_dataset.describe()

#see how many examples there are for diabetes examples and non diabetes ones
diabetes_dataset['Outcome'].value_counts()
diabetes_dataset.groupby('Outcome').mean()

#seperating the data and labels
X = diabetes_dataset.drop(columns = 'Outcome', axis = 1) #axis 1 if dropping column and 0 if row
Y = diabetes_dataset['Outcome']
print(X)
print(Y)

#data standardization
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X) #transforming all data to common range, fit_transform()
print(standardized_data) #all values in range from 0 to 1
X = standardized_data
print(X)
print(Y)

#splitting data into training and test data 
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

#training the model 
classifier = svm.SVC(kernel='linear') #initializes svm classifier with linear kernel 
#training the support vector machine classifier
classifier.fit(X_train, Y_train)

#model evaluation
#accuracy score on the training data 
X_train_predict = classifier.predict(X_train) #predict label for X train 
training_data_accuracy = accuracy_score(X_train_predict, Y_train) #comparing the model to the original Y train
print('Accuracy score of the training data :', training_data_accuracy) #prediction 79 times right
#accuracy score on the test data 
X_test_predict = classifier.predict(X_test) #predict label for X train 
test_data_accuracy = accuracy_score(X_test_predict, Y_test) #comparing the model to the original Y train
print('Accuracy score of the test data :', test_data_accuracy)


#making predictive system 
input_data = (4,110,92,0,0,37.6,.191,30)

#change the input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the array as we prediciting for one instance 
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1) #predict for one instance 

#standardize the input data 
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else : 
  print('The person is diabetic')