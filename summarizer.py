# -*- coding: utf-8 -*-

import spacy
from spacy import displacy
from pathlib import Path

if __name__ =="__main__":

	# Load the language model
	nlp = spacy.load("./training/model-best")

	sentence = "Burak, bizim manavdan 10 adet elma satın aldı"

	doc = nlp(sentence)
	print(doc);

	print ("{:<15} | {:<8} | {:<15} | {:<20}".format('Token','Relation','Head', 'Children'))
	print ("-" * 70)

	for token in doc:
  	# Print the token, dependency nature, head and all dependents of the token
	  print ("{:<15} | {:<8} | {:<15} | {:<20}"
	         .format(str(token.text), str(token.dep_), str(token.head.text), str([child for child in token.children])))

	# Use displayCy to visualize the dependency
	# And save it as SVG file

	# for jupiter, uncomment the next line
	# displacy.render(doc, style='dep', jupyter=True, options={'distance': 120})

	svg = displacy.render(doc, style='dep', options={'distance': 120})

	output_path = Path("./dependency_plot.svg")
	output_path.open("w", encoding="utf-8").write(svg)
