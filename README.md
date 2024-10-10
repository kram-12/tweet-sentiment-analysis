# Tweet Sentiment Analysis

## Project Overview
This project performs sentiment analysis on tweets using the **Sentiment140** dataset. The dataset consists of 1.6 million tweets labeled with either positive or negative sentiment, which allows training a machine learning model to automatically detect sentiment in text data.

## Features
- **Dataset**: Sentiment140 dataset from Kaggle
- **Machine Learning Model**: Logistic Regression (for classification)
- **Text Preprocessing**: 
  - Cleaning text (e.g., removing URLs, special characters)
  - Stopword removal
  - Stemming
- **Model Evaluation**: Achieves 79.87% accuracy on training data and 77.66% on test data.

### 1. Installing Kaggle Library and Fetching the Dataset

To download the dataset using the Kaggle API:

1. Install the `kaggle` library:
   ```bash
   pip install kaggle
   ```

2. Configure your Kaggle credentials:
   Place your `kaggle.json` (API key) in the appropriate directory and ensure it's secured:
   ```bash
   mkdir -p ~/.kaggle
   cp kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

3. Download the **Sentiment140** dataset:
   ```bash
   kaggle datasets download -d kazanova/sentiment140
   ```

4. Extract the dataset:
   ```bash
   from zipfile import ZipFile
   dataset="/content/sentiment140.zip"
   with ZipFile(dataset, 'r') as zip:
       zip.extractall()
       print("Dataset extracted.")
   ```

### 2. Importing Dependencies

The project uses several Python libraries for data handling, preprocessing, and machine learning. Import them as follows:

```python
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

Make sure to download the NLTK stopwords:
```python
import nltk
nltk.download('stopwords')
```

### 3. Loading the Dataset 

The dataset is available in the csv format, so load the dataset from csv fiel to pandas dataframe:

```python
column_names=['target', 'id', 'date', 'flag', 'user', 'text']
twitter_data=pd.read_csv("/content/training.1600000.processed.noemoticon.csv", names=column_names, encoding="ISO-8859-1")
     
```

### 4. Checking for Null Values

After loading the dataset, it's essential to check for missing data:

```python
twitter_data.isnull().sum()  # Display count of null values per column
```

### 5. Stemming: Reducing Words to Root Form

The **PorterStemmer** is used to reduce words to their root form:

```python
ps = PorterStemmer()

def stemming(content):
    stemmed_content=re.sub('[^a-zA-Z]',' ',content)
    stemmed_content=stemmed_content.lower()
    stemmed_content=stemmed_content.split()
    stemmed_content=[port_stem.stem(word)for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content=' '.join(stemmed_content)

    return stemmed_content

# Apply stemming to the dataset
twitter_data['stemmed_content']=twitter_data['text'].apply(stemming)
```
### 6. Splitting the Dataset: Training (80%) and Testing (20%)

Split the dataset into training and testing sets:

```python
X=twitter_data['stemmed_content'].values
Y=twitter_data['target'].values
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2, stratify=Y, random_state=2)

```

### 7. Converting Text Data to Numerical Data

Convert the cleaned and stemmed text into numerical features using **TF-IDF**:

```python
vectorizer=TfidfVectorizer()

X_train=vectorizer.fit_transform(X_train)
X_test=vectorizer.transform(X_test)
```

### 8. Training the Model using Logistic Regression

Train a logistic regression model on the training set:

```python
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)
```

### 9. Checking the Accuracy Score

After training the model, evaluate its accuracy on both the training and testing data:

```python
# Accuracy on training data
X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(Y_train, X_train_prediction)
print('Accuracy Score on the training data:',training_data_accuracy)

# Accuracy on testing data
X_test_prediction=model.predict(X_test)
test_data_accuracy=accuracy_score(Y_test, X_test_prediction)
print('Accuracy Score on the test data:',test_data_accuracy)
```
- **Training Accuracy**: 79.87%
- **Test Accuracy**: 77.66%

### 10. Saving the Trained Model

Save the trained logistic regression model for later use:

```python
import pickle
filename="trained_model.sav"
pickle.dump(model,open(filename,'wb'))
```

### 11. Checking the Model with Test Dataset

To check the model with unseen data from the test dataset, use the following:
```python
X_new=X_test[200]
print(Y_test[200])

prediction=model.predict(X_new)
print(prediction)

if(prediction[0]==0):
  print("Negative Tweet")
else:
  print("Positive Tweet")
```
### 12. Checking the Model with Dummy Data

To check the model with a dummy tweet, use the following:
```python
tweet_data1 = {
    'target': None,  # We'll predict this
    'id': 9999999999,  # Assign a dummy ID
    'date': '2023-11-17',  # Assign a dummy date
    'flag': 'None',  # Assign a dummy flag
    'user': 'dummy_user',  # Assign a dummy user
    'text': "Had the best time today at the park! The weather was perfect, and the sunset was absolutely stunning. Can't wait to go back! 🌅 #blessed #naturelove"
}

# Create a pandas DataFrame
dummy_tweet1 = pd.DataFrame([tweet_data1])

# Apply stemming function
dummy_tweet1['stemmed_content'] = dummy_tweet1['text'].apply(stemming)

# Vectorize using your existing vectorizer
dummy_tweet1_vectorized = vectorizer.transform(dummy_tweet1['stemmed_content'])

# Predict using your loaded model
prediction = loaded_model.predict(dummy_tweet1_vectorized)

# Assign the prediction to the 'target' column
dummy_tweet1['target'] = prediction[0]

# Print the prediction
if prediction[0] == 0:
    print("Negative Tweet")
else:
    print("Positive Tweet")
```
---

### Project Structure

- **Sentiment_Analysis.ipynb**: The main notebook where the preprocessing, model training, and evaluation take place.
- **Dataset**: The Sentiment140 dataset downloaded from Kaggle.
- **strained_model.sav**: The saved logistic regression model.

---

### License
This project uses the Sentiment140 dataset, available under the terms specified by its creators on [Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140).