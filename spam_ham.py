# importing the necessary libraries
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import pickle

#import the dataset
data = pd.read_csv('spam_ham_dataset.csv')

#see the top 5 entries in dataframe
data.head()

# plotting number of spam and ham labels as histogram
data['label'].hist()
plt.show()

# Please close the window after viewing the histogram for program to resume execution

# checking for na values
data.isna().sum()

# checking for empty text in dataset
index = []
for i in range(0,len(data)):
    if data['text'][i].isspace():
        index.append(i)
print(index)
print(f'length is :{len(index)}')

X = data['text'] # features(raw text)
y = data['label'] # labels
# splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf_model = Pipeline([('feature_extraction',TfidfVectorizer()),('SVC',SVC())])

# training the model
clf_model.fit(X_train,y_train)

predictions = clf_model.predict(X_test)

print(accuracy_score(y_test,predictions))
print('\n')
print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))
# saving the classifier
with open('spam_ham_classifier_model.pkl', 'wb') as f:
    pickle.dump(clf_model, f)
