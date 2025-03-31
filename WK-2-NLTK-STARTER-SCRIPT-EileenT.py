'''
CYBV-475
Cyber Deception Week 2
Using NLTK and Pandas
to extract parts of speech
from text

'''
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import re

with open('jules.txt') as julesText:
    julesContent = julesText.read(1000)
with open('doyle.txt') as doyleText:
    doyleContent = doyleText.read(1000)
    
textJules = re.sub("[^a-zA-Z ]", ' ', julesContent)
textDoyle = re.sub("[^a-zA-Z ]", ' ', doyleContent)

julesTokens = nltk.word_tokenize(textJules)
julesTags = nltk.pos_tag(julesTokens)

doyleTokens = nltk.word_tokenize(textDoyle)
doyleTags = nltk.pos_tag(doyleTokens)

posList = [0,0,0,0,0,0,0]  # [0] = Jules-NounCount,  [1] = Jules-VerbCount [2] =Jules- AdjectiveCount [3] Doyle-NounCount  [4] = Doyle-VerbCount [5] Doyle- AdjectiveCount

for eachTag in julesTags:
    if eachTag[1] in ["NN", "NNP", "NNPS"]:
        posList[0] += 1
    elif eachTag[1] in ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]:
        posList[1]+=1 
    elif eachTag[1] in ["JJ", "JJR", "JJS"]:
        posList[2] += 1

        
posList[0] = (posList[0] / len(julesTags)) * 100.0
posList[1] = (posList[1] / len(julesTags)) * 100.0
posList[2] = (posList[2] / len(julesTags)) * 100.0

for eachTag in doyleTags:
    if eachTag[1] in ["NN", "NNP", "NNPS"]:
        posList[3] += 1
    elif eachTag[1] in ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]:
        posList[4]+=1 
    elif eachTag[1] in ["JJ", "JJR", "JJS"]:
        posList[5] += 1

posList[3] = (posList[3] / len(doyleTags)) * 100.0
posList[4] = (posList[4] / len(doyleTags)) * 100.0      
posList[5] = (posList[5] / len(doyleTags)) * 100.0

print(posList)

df = pd.DataFrame(posList, index = ['JulesNoun', 'JulesVerb', 'JulesAdjective', 'DoyleNoun', 'DoyleVerb', 'DoyleAdjective' ])

df.plot(kind='barh', color=['blue', 'pink', 'orange', 'red', 'purple', 'yellow'], figsize=(6, 4))

plt.title('Comparing Parts of Speech in Texts by Jules Verne and Sir Arthur Conan Doyle')
plt.xlabel('Percentage of Total Words')
plt.ylabel('Parts of Speech')
plt.grid(True, axis='x', linestyle='--', color='gray', linewidth=0.5)

print(df)
plt.show()

