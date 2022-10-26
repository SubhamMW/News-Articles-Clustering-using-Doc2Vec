# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 10:28:05 2021

@author: user
"""

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import pandas as pd
from pandas import DataFrame
import numpy as np
import nltk as nt
from sklearn.cluster import KMeans
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn import metrics 
from scipy.spatial.distance import cdist 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.manifold import TSNE
from fcmeans import FCM
from sklearn.cluster import DBSCAN

lmt=WordNetLemmatizer()
rt = RegexpTokenizer(r'\w+')
sw = set(stopwords.words('english'))

#Data Preprocessing
news = pd.read_json("F:/Study/NISER/Machine Learning/Project/data_world.json")
newdf=pd.DataFrame()
newdf['title']=news['title']
newdf['content']=news['content']
newdf['t_content']=None
newdf['t_spaced_content']=None

for i in range(len(newdf['content'])):
    #print(i)
    wt1=wt1=rt.tokenize(newdf['content'][i])
    filtered_sentence = []
    spaced_sentence=''
    for w in wt1:
        w=w.lower()
        w=lmt.lemmatize(w)
        if w not in sw:
            filtered_sentence.append(w)
            spaced_sentence=spaced_sentence+w+' '
    newdf['t_content'][i]=filtered_sentence
    newdf['t_spaced_content'][i]=spaced_sentence
    
Tag_documents= [TaggedDocument(doc, ["DOC_"+str(i)]) for i, doc in enumerate(newdf['t_content'])]
 
#Doc2Vec model training   
model_newdf = Doc2Vec(Tag_documents,dm=1, vector_size=150, window=2, min_count=1, workers=4)
model_newdf.save("F:/Study/NISER/Machine Learning/model_newdf")
model_newdf=Doc2Vec.load("F:/Study/NISER/Machine Learning/model_newdf")

newdf_vec=[]
for x in range(len(newdf)):
    newdf_vec.append(model_newdf.docvecs[x])
    
newdf_vec=np.array(newdf_vec)

#t-SNE dimension reductions
tsne=TSNE(n_components=2)
newdf_vec=tsne.fit_transform(newdf_vec)
distortions_newdf = [] 
inertias_newdf = [] 
mapping1_newdf = {} 
mapping2_newdf = {}
plt.scatter(newdf_vec[:, 0], newdf_vec[:, 1],marker='o',edgecolor='black',color='w')
plt.show()

K = range(1,10) 

for k in K: 
    #Building and fitting the model 
    print(k)
    kmeanModel_newdf = KMeans(n_clusters=k)#.fit(art2_Vec) 
    kpred_newdf=kmeanModel_newdf.fit_predict(newdf_vec)   
    
    plt.scatter(newdf_vec[:, 0], newdf_vec[:, 1],s=20, c=kmeanModel_newdf.labels_,marker='o', edgecolor='black',label='cluster 1')
    plt.scatter(kmeanModel_newdf.cluster_centers_[:,0],kmeanModel_newdf.cluster_centers_[:,1],c='r',s=50,marker='D')
    plt.title("After"+str(k)+" Clustering")
    plt.show()
    
    distortions_newdf.append(sum(np.min(cdist(newdf_vec, kmeanModel_newdf.cluster_centers_, 
                      'euclidean'),axis=1)) / newdf_vec.shape[0]) 
    inertias_newdf.append(kmeanModel_newdf.inertia_) 
  
    mapping1_newdf[k] = sum(np.min(cdist(newdf_vec, kmeanModel_newdf.cluster_centers_, 
                 'euclidean'),axis=1)) / newdf_vec.shape[0] 
    mapping2_newdf[k] = kmeanModel_newdf.inertia_
    
    
for key,val in mapping1_newdf.items(): 
    print(str(key)+' : '+str(val)) 
    
for key,val in mapping2_newdf.items(): 
    print(str(key)+' : '+str(val)) 
    
#Distortions and Inertia values    
plt.plot(K, inertias_newdf, 'bx-') 
plt.xlabel('Values of K') 
plt.ylabel('Inertia') 
plt.title('The Elbow Method (Inertia plot) of all AI jazeera') 
plt.show() 
labels_newdf=kmeanModel_newdf

plt.plot(K, distortions_newdf, 'bx-') 
plt.xlabel('Values of K') 
plt.ylabel('Distortions') 
plt.title('The Elbow Method (Distortions plot) of all AI jazeera') 
plt.show()

#Applying DBSCAN
DB=DBSCAN(eps=2.9,min_samples=5).fit(newdf_vec)
DBlabels=DB.labels_
no_clusters = len(np.unique(DBlabels) )
no_noise = np.sum(np.array(DBlabels) == -1, axis=0)
newdf['DBSCAN CLUSTER_LABELS']=DBlabels


print('Estimated no. of clusters: %d' % no_clusters)
print('Estimated no. of noise points: %d' % no_noise)


plt.scatter(newdf_vec[:,0], newdf_vec[:,1], c=DBlabels, marker="o", picker=True, edgecolor='black')
plt.title(' clusters with data')
plt.xlabel('Axis X[0]')
plt.ylabel('Axis X[1]')
plt.show()

    
# after confirmation that best clustering is 3 cluster we do a 3 clustering
kmeanModel_newdf_150vec_3cluster = KMeans(n_clusters=3)#.fit(art2_Vec) 
kpred_newdf_3cluster=kmeanModel_newdf_150vec_3cluster.fit_predict(newdf_vec)
Klabels_150vec_3cluster=kmeanModel_newdf_150vec_3cluster.labels_
print(Klabels_150vec_3cluster)
newdf['Kmeans CLUSTER_LABELS']=Klabels_150vec_3cluster


    
#generation of wordclouds
from wordcloud import WordCloud
from matplotlib import pyplot as plt
from math import ceil

nclust = 3
ncols = 3
nrows = ceil(nclust/ncols)
import string

fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows, figsize=(20,20))
for i in range(nclust) : 
    context = list(newdf[newdf['Kmeans CLUSTER_LABELS'] == i].t_spaced_content)
#     context = list(df2[df2.cluster_no == i].content2)
    data = " ".join(context)
    
    wordcloud = WordCloud(
        background_color='white',
        max_words=100,
        max_font_size=40,
        scale=3,
        random_state=1  # chosen at random by flipping a coin; it was heads
    ).generate(data)

    axeslist.ravel()[i].imshow(wordcloud)
    axeslist.ravel()[i].set_title("Cluster {}".format(i))
    axeslist.ravel()[i].set_axis_off()
        
plt.tight_layout()
plt.show()