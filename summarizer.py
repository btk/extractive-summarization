import os
import nltk
import string

from bs4 import BeautifulSoup
import pandas as pd
import sklearn as sk
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from pathlib import Path
from string import digits
import pickle

def read_documents():
	documentNames = os.listdir("./documents")
	documents = [];
	for documentName in documentNames:
		document = open("./documents/"+documentName, "r", encoding='utf-8', errors='ignore').read()
		sentences = parse_document(document);
		documents += [(documentName, sentences)]
	return documents

def read_document(documentName):
	document = open("./documents/"+documentName, "r", encoding='utf-8', errors='ignore').read()
	sentences = parse_document(document);
	return sentences;

def parse_document(document):
	bs = BeautifulSoup(document, features="html.parser")
	sentences = bs.case.sentences.findAll("sentence")
	return list(map(lambda sentence: sentence.text, sentences))

def create_document_collection(documentTuples):
	corpus = []
	for documentTuple in documentTuples:
		for line in documentTuple[1]:
			corpus += [line]
	return corpus

# Taken from my first assignment
def preprocess(lines):

	processed = [];
	lineCounter = 0;
	bookName = "";

	for line in lines:
		line = line.strip(); # Strip all extra white spaces
		line = line.translate({ord(k): None for k in digits}) # Remove all numbers in the string
		# I will turn all letters to lower case since our goal is to classify depending on topic
		line = line.lower();

		# Expand contaracted words.
		line = decontracted(line);

		# remove punctuations
		line = line.translate(str.maketrans('', '', string.punctuation));

		# remove the stop words
		line = remove_stopwords(line);

		# instead of stemming words, I will lemmatize them with wordnet article database
		line = lemmatize_words(line);

		# remove single letter words in the text like: j. f. kennedy => kennedy
		line = ' '.join( [w for w in line.split() if len(w)>1] )

		processed += [line];

		lineCounter+=1;


	return processed 	# you may change the return value if you need.

# Taken from my first assignment
def decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

stop_words = set(stopwords.words('english'))
def remove_stopwords(data):
    temp_list=[]
    for word in data.split():
        if word.lower() not in stop_words:
            temp_list.append(word)
    return ' '.join(temp_list)



lemma = nltk.wordnet.WordNetLemmatizer()
lemma.lemmatize('article')
def lemmatize_words(text):
    return " ".join([lemma.lemmatize(word) for word in text.split()])

def calculate_tfidf(load_from_pickle):
	if(load_from_pickle):
		transformer = pickle.load(open("tfidf.pickle", "rb" ))
	else:
		documents = read_documents()
		corpus = create_document_collection(documents)

		transformer = TfidfVectorizer()
		tfidf = transformer.fit_transform(corpus)
		with open("tfidf.pickle", 'wb') as handle:
			pickle.dump(transformer, handle)

	return transformer

def get_summary_by_document(transformer, documentName):
	sentences = read_document(documentName)
	new_tfidf = transformer.transform(sentences)
	cs = cosine_similarity(new_tfidf, new_tfidf)
	sentenceIndexes = cs[0].argsort()[:-6:-1]
	print(sentenceIndexes);
	for index in sentenceIndexes:
		print(sentences[index])

if __name__ =="__main__":
	LOAD_FROM_PICKLE = 1;
	FILE_NAME = "08_3.xml"

	transformer = calculate_tfidf(LOAD_FROM_PICKLE)
	get_summary_by_document(transformer, FILE_NAME)



#
