import os
from bs4 import BeautifulSoup

def read_documents():
	documentNames = os.listdir("./documents")[:10]
	for documentName in documentNames:
		document = open("./documents/"+documentName, "r", encoding='utf-8', errors='ignore').read()
		sentences = parse_document(document);
		print(documentName, len(sentences));

def parse_document(document):
	bs = BeautifulSoup(document, features="html.parser")
	sentences = bs.case.sentences.findAll("sentence")
	return list(map(lambda sentence: sentence.text, sentences))



if __name__ =="__main__":
	read_documents()
