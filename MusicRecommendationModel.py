import numpy as np
import pandas as pd
import seaborn as sb

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


df = pd.read_csv('dataset.csv', chunksize=1000)

for chunk in df:
 print(chunk)
 break

print(chunk)
df = chunk
print(df)

print(df.info)
print(df.isna().sum())
df = df.fillna('')

print(df.isna().sum())

songVecto = CountVectorizer()
songVecto.fit(df['track_name'])

df = df.sort_values(by= ['popularity'] , ascending = False)

print("\n" , df , "\n")

def similar(name , data):
  text = songVecto.transform(data[data['track_name']==name]['artists']).toarray()
  num= data[data['track_name']== name].select_dtypes(include= np.number).to_numpy()

  check = []
  for i,row in data.iterrows():
   nm = row['track_name']
   text2 = songVecto.transform(data[data['track_name']==name]['artists']).toarray()
   num2 = data[data['track_name']== name].select_dtypes(include= np.number).to_numpy()

   textcheck = cosine_similarity(text,text2)[0][0]
   numcheck= cosine_similarity(num,num2)[0][0]
   check.append(textcheck + numcheck)
  return check


def recomm(song, data = df):
 
 if df[df['track_name']== song].shape[0] == 0:
    print('Invalid!!!! here below are some Recommendations')
    for songg in data.sample(n=7)['track_name'].values:
       print(songg)
    return

 data['similarity'] = similar(song,data)
 data.sort_values(by=['similarity','popularity'], ascending = [False,False] )
 print(data[['track_name','artists']][1:8])
 
recomm('Pano')
print(df['track_name'])