import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import nltk
import seaborn as sns
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# nltk.download('stopwords')  // only for jupyter
# nltk.download('punkt_tab')  //only ofr jupyter

df=pd.read_csv('spam.csv',encoding='latin1')

# ====================
## 1. CLEAN THE DATA 
# ====================

# remove columns with most nullish values and remove them if they are unnecessary

# df.info()
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
# df.head()

# change the name of columns to v1 to target and v2 to text
df.rename(columns={'v1':'target','v2':'text'},inplace=True)
# df.head()


# now convert the first column labels ham spam to 0 1
encoder=LabelEncoder()
df['target']=encoder.fit_transform(df['target'])
# df.head()
df.isnull().sum() # check is there any missing value in our data 
# check duplicate values in our dataset and remove them if they exist
df.duplicated().sum()
df=df.drop_duplicates(keep="first")
df.duplicated().sum()
# df.shape

# =================================
## 2. EDA (Exploratory data analysis
# =================================

# check how much ham and how much spam exist in our dataset
df['target'].value_counts()
# show them in a chart for better viulization 

# chart for target ================
# plt.pie(df['target'].value_counts(),labels=['ham','spam'],autopct='%0.2f')
# plt.show()

# so this upper chart shows that ham are very much and spam are less so data is in balance 
# and now for better analysis create three more columns num_words,num_sentences,num_characters


df['num_characters']=df['text'].apply(len)
df['num_words'] = df['text'].apply(lambda x: len(nltk.word_tokenize(x)))
df['num_sentences']=df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))

# different hams and spams 
hams_messages=df[df['target']==0]
# hams_messages.describe()
spams_messages=df[df['target']==1]
# spams_messages.describe()



# chart for num_characters ================
# plt.figure(figsize=(12,8))
# sns.histplot(hams_messages['num_characters'])
# sns.histplot(spams_messages['num_characters'],color='red')
# chart for num_sentences ================
# plt.figure(figsize=(12,8))
# sns.histplot(hams_messages['num_sentences'])
# sns.histplot(spams_messages['num_sentences'],color='red')
# chart for num_words ================
# plt.figure(figsize=(12,8))
# sns.histplot(hams_messages['num_words'])
# sns.histplot(spams_messages['num_words'],color='red')

# chart for target relation  ================
# sns.pairplot(df,hue='target')

# find core relation between target ================
# df_numeric = df.select_dtypes(include=np.number)
# sns.heatmap(df_numeric.corr(), annot=True)


# =============================
## 3. DATA PREPROCESSING
    # . lower case
    # . tokenization
    # . removing special characters
    # . removing stop words and punctuations
    # . stemming
# =============================


ps=PorterStemmer()
def transform_text(text):
    # make the all text lower case
    text=text.lower() 
    # tokenization
    text=nltk.word_tokenize(text)
    # remove special characters only alpha & numerics remaining 
    modifiedText=[]
    for i in text:
        if (i.isalnum()):
            modifiedText.append(i)
    text=modifiedText[:]
    modifiedText.clear()
    # removing stop words and punctuations
    stop_words= stopwords.words('english')
    punctuations=string.punctuation
    for i in text:
        if (i not in stop_words and i not in punctuations):
            modifiedText.append(i)
    text=modifiedText[:]
    modifiedText.clear()
    # seeming the words
    for i in text:
        modifiedText.append(ps.stem(i))
    return " ".join(modifiedText)
df['transformed_text']=df['text'].apply(transform_text)

## 4. Now make the word cloud for spam and ham messages
wc=WordCloud(width=1000, height=1000, background_color='white', min_font_size=10)

spam_transformed=df[df['target']==1]['transformed_text']
spam_wc=wc.generate(spam_transformed.str.cat(sep=" "))
# chart for showing spam words in word cloud  ================
# plt.figure(figsize=(12,8))
# plt.imshow(spam_wc)

ham_transformed=df[df['target']==0]['transformed_text']
ham_wc=wc.generate(ham_transformed.str.cat(sep=" "))
# chart for showing ham words in word cloud  ================
# plt.figure(figsize=(12,8))
# plt.imshow(ham_wc)


# get the spam corpus and then find 30 most common use words in spam messages

spam_corpus=[]
for word in spam_transformed.tolist():
    for i in word.split():
        spam_corpus.append(i)
# len(spam_corpus)
ham_corpus=[]
for word in ham_transformed.tolist():
    for i in word.split():
        ham_corpus.append(i)
# len(ham_corpus)

# find most 30 spam and ham corpus and show in chart ===========

# spamCountersDF=pd.DataFrame(Counter(spam_corpus).most_common(30))
# sns.barplot(x=spamCountersDF[0],y=spamCountersDF[1])
# plt.xticks(rotation=90)
# plt.show()
# hamCountersDF=pd.DataFrame(Counter(ham_corpus).most_common(30))
# sns.barplot(x=hamCountersDF[0],y=hamCountersDF[1])
# plt.xticks(rotation=90)
# plt.show()

# ======================
## 5. MODEL BUILDING 
# ======================

# convert text_transformed to vectorize text
cv=CountVectorizer()
tfidf=TfidfVectorizer(max_features=2500)


X=tfidf.fit_transform(df['transformed_text']).toarray()
Y=df['target'].values

# split the data for trail and test
x_train,x_test,y_train,y_test=train_test_split(X,Y,train_size=0.3,random_state=2)


# now get the neive bayes model because our data is textual
mnb=MultinomialNB()


mnb.fit(x_train,y_train)
y_predict_mnb=mnb.predict(x_test)
print(accuracy_score(y_test,y_predict_mnb))
print(precision_score(y_test,y_predict_mnb))


pickle.dump(tfidf,open('vectroizer.pk1','wb'))
pickle.dump(mnb,open('model.pk1','wb'))



















