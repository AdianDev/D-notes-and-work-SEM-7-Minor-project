import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer as cv
# from sklearn.feature_extraction.text import CountVectorizer
import re
import string
import pickle

df=pd.read_csv("dataset.csv")
df.columns
df=df[["content","rating"]]
df["rating"]=df["rating"].fillna(df["rating"].mode()[0])
df=df.dropna()

def sentiment(n):
    return 1 if n > 4 else 0
df['sentiment'] = df['rating'].apply(sentiment)

def cleantext(txt):
    txt=str(txt)
    txt=txt.lower()
    txt = re.sub("[,.*!_]","",txt)
    txt = re.sub("\n","",txt)
    txt = re.sub("\t","",txt)
    txt = re.sub("\w*\d\w*","",txt)
    txt = re.sub("^\s+","",txt)
    txt=re.sub("[%s]"%re.escape(string.punctuation),"",txt)
    txt=re.sub("[''""``]","",txt)
    return txt

cle= lambda x:cleantext(x)

df["content"]=pd.DataFrame(df.content.apply(cle))    
x=df.content

y=df["sentiment"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
vec=cv()
vec.fit(x)
vec.get_feature_names()
x_train=vec.transform(x_train)
a=pd.DataFrame(x_train.toarray(), columns=vec.get_feature_names())
x_test=vec.transform(x_test)



model=LogisticRegression()
# model2=KNeighborsClassifier()
# model1=DecisionTreeClassifier()
model.fit(x_train,y_train)

model=LogisticRegression()
model.fit(x_train,y_train)
'''
with open("PRSA.pickle", "wb") as f:
    pickle.dump(model, f)

with open("prsa_pickle", "wb") as f: 
    pickle.dump(model,f)
    
with open("vect_pickle", "wb") as f: 
    pickle.dump(vec,f)'''
#model.score(x_test,y_test)
# pre=model.predict(x_test)
# model.score(x_test,y_test)

#confusion_matrix(y_test,pre)
#model.predict(x_test)

inp = ["Bad"]
vecInp=vec.transform(inp)
ans=model.predict(vecInp)
print(ans)

