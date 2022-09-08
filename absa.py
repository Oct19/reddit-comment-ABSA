import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer 
import stanfordnlp
import glob, os

indir = './result/Comments ABSA/'


# download for first time running
# stanfordnlp.download('en')
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

txt = "The Sound Quality is great but the battery life is very bad. Second sentence goes like this, well said! Third sentence's word count is low. And 4th sentence is short too!"

list_of_files = glob.glob(indir+'*.csv') # * means all, *.csv means csv files only
latest_file = max(list_of_files, key=os.path.getctime)
print('Read file: '+latest_file)

df = pd.read_csv(latest_file)
df = df.reset_index()  # make sure indexes pair with number of rows

df['POS']

for index, comment in df.iterrows():
    txt = comment['Body']
    weight = comment['Upvotes']
    txt = txt.lower()

    # Tokenize txt into sentences
    sentList = nltk.sent_tokenize(txt)

    # Tokenize and POS Tag the sentence.
    for line in sentList:
        txt_list = nltk.word_tokenize(line)
        taggedList = nltk.pos_tag(txt_list)

        # Joining Nouns: sound quality -> soundquality
        newwordList = []
        flag = 0
        for i in range(0,len(taggedList)-1):
            if(taggedList[i][1]=="NN" and taggedList[i+1][1]=="NN"):
                newwordList.append(taggedList[i][0]+taggedList[i+1][0])
                flag=1
            else:
                if(flag==1):
                    flag=0
                    continue
                newwordList.append(taggedList[i][0])
                if(i==len(taggedList)-2):
                    newwordList.append(taggedList[i+1][0])
        finaltxt = ' '.join(word for word in newwordList)
        # print(finaltxt)

        # Tokenize and POS Tag the new sentence.
        stop_words = set(stopwords.words('english'))
        new_txt_list = nltk.word_tokenize(finaltxt)
        wordsList = [w for w in new_txt_list if not w in stop_words]
        taggedList = nltk.pos_tag(wordsList)
        print(taggedList)

# NLP
nlp = stanfordnlp.Pipeline()
doc = nlp(finaltxt)
dep_node = []
for dep_edge in doc.sentences[0].dependencies:
    dep_node.append([dep_edge[2].text, dep_edge[0].index, dep_edge[1]])
for i in range(0, len(dep_node)):
    if (int(dep_node[i][1]) != 0):
        dep_node[i][1] = newwordList[(int(dep_node[i][1]) - 1)]
print(dep_node)