# reading the data
import nltk
import pandas as pd

import re
#from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
messages=pd.read_csv("SMSSpamCollection", sep= "\t",names=['label','message'])
lemmatizer=WordNetLemmatizer()
list=[]
for i in range (0,len(messages)):
    review=re.sub('[^a-zA-Z]',' ',messages['message'][i])
    review=review.lower()
    review=review.split()
    #for word in review:
    review=[lemmatizer.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]
    review=' '.join(review)
    list.append(review)
#at this part our data cleaning is done 
#bag of words            

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000)
x=cv.fit_transform(list).toarray()  

y=pd.get_dummies(messages['label'])        #get dummies is used to convert our dependent variable into dummy variable bcoz model cannot understand ham and spam
y=y.iloc[:,1].values                       #instead of having two categorical variables we are having one  
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)

#train model by naive bayes
from sklearn.naive_bayes import MultinomialNB
spam_detect_model=MultinomialNB().fit(x_train,y_train)
y_pred=spam_detect_model.predict(x_test)

#to check the accurace of the model we use confusion matrix
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)

    


