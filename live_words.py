from textblob import TextBlob as tb
import string
import math
import urllib
from bs4 import BeautifulSoup
import nltk
import os
import gensim
import numpy as np
import csv
import redis
import re 
from sklearn.externals import joblib

import json

r = redis.StrictRedis(host='localhost', port=6379, db=0)
image_keywords = r.lrange('food3', 0, -1)



texts = []
stopwords = nltk.corpus.stopwords.words()
# 
def main(url):
	page = urllib.urlopen(url).read()
	soup = BeautifulSoup(page)

	# kill all script and style elements
	for script in soup(["script", "style"]):
		script.extract()    # rip it out

	# get text
	text = soup.get_text()

	# break into lines and remove leading and trailing space on each
	lines = (line.strip() for line in text.splitlines())
	# break multi-headlines into a line each
	chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
	# drop blank lines
	text = '\n'.join(chunk for chunk in chunks if chunk)

	# print((text.split()))

	text_split = text.split()
	text_clean = [w.lower() for w in text_split if w.isalnum() and w.lower() not in stopwords]
	# texts.append(text_clean)

	# print (texts)
	print "KEYWORDS FROM NEWSLETTER: " ,text_clean

	print ("model loading .............")
	model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
	# joblib.dump(model, 'model.pkl') 
	model.save('my')
	# model = joblib.load('model.pkl') 

	# new_model = gensim.models.Word2Vec.load('my')
	print ("model loaded")

	featureVec = np.zeros(300,dtype="float32")
	nwords = 0




	for word in set(text_clean):
	    try:
	    	featureVec = np.add(featureVec,model[word])
	    	nwords = nwords + 1
	    except KeyError:
	    	pass

	featureVec = np.divide(featureVec,nwords)


	from sklearn.metrics.pairwise import cosine_similarity



	with open("hadoop.csv", 'r') as f:
	    data = [row for row in csv.reader(f.read().splitlines())]

	d = {} 
	for row in data[1:]:
		# print "advertiser id: ", row[0], "keywords: ", row[2], ",", row[3], ",", row[4]
		score_keyword = model[row[2]]
		score_keyword1 = model[row[3]]
		score_keyword2 = model[row[4]]
		score_keyword = score_keyword.reshape(1, -1)
		score_keyword1 = score_keyword1.reshape(1, -1)
		score_keyword2 = score_keyword2.reshape(1, -1)
		featureVec = featureVec.reshape(1, -1)
		d[row[2]] = (cosine_similarity(featureVec, score_keyword ))
		d[row[3]] = (cosine_similarity(featureVec, score_keyword1 ))
		d[row[4]] =(cosine_similarity(featureVec, score_keyword2 ))
	





	featureVec_image = np.zeros(300,dtype="float32")
	nwords = 0
	for word in set(image_keywords):
	    try:
	    	featureVec_image = np.add(featureVec_image,model[word])
	    	nwords = nwords + 1
	    except KeyError:
	    	pass

	featureVec_image = np.divide(featureVec_image,nwords)


	d_image = {} 
	for row in data[1:]:
		# print "advertiser id: ", row[0], "keywords: ", row[2], ",", row[3], ",", row[4]
		score_keyword = model[row[2]]
		score_keyword1 = model[row[3]]
		score_keyword2 = model[row[4]]
		score_keyword = score_keyword.reshape(1, -1)
		score_keyword1 = score_keyword1.reshape(1, -1)
		score_keyword2 = score_keyword2.reshape(1, -1)
		featureVec_image = featureVec_image.reshape(1, -1)
		d_image[row[2]] = (cosine_similarity(featureVec_image, score_keyword ))
		d_image[row[3]] = (cosine_similarity(featureVec_image, score_keyword1 ))
		d_image[row[4]] =(cosine_similarity(featureVec_image, score_keyword2 ))
	# print "image", d_image
	# print d_image
	d_final = {}

	for k, v in d.items():
		d_final[k] = d[k] + 0.25*d_image[k]
	# print "final", d
	results = []
	j = {}
	max = 0.0
	winner_url = ""
	for row in data[1:]:
		j["advertiser_id"] = row[0]
		j["url"] = row[1]
		j["cpm"] = row[5]
		j["keywords"]=[]
		j["keywords"].append(row[2])
		j["keywords"].append(row[3])
		j["keywords"].append(row[4])
		j["score_context"] = []
		j["score_context"].append("{0:.2f}".format(float(d[row[2]][0])))
		j["score_context"].append(" ")
		j["score_context"].append("{0:.2f}".format(float(d[row[3]][0])))
		j["score_context"].append("{0:.2f}".format(float(d[row[4]][0])))
		j["score_image"] = []
		j["score_image"].append("{0:.2f}".format(float(d_image[row[2]][0])))
		j["score_image"].append("{0:.2f}".format(float(d_image[row[3]][0])))
		j["score_image"].append("{0:.2f}".format(float(d_image[row[4]][0])))
		j["score_final"] = []
		j["score_final"].append("{0:.2f}".format(float(d_final[row[2]][0])))
		j["score_final"].append("{0:.2f}".format(float(d_final[row[3]][0])))
		j["score_final"].append("{0:.2f}".format(float(d_final[row[4]][0])))
		final_score = (d_final[row[2]][0] + d_final[row[3]][0] + d_final[row[4]][0])/3.0
		# if final_score > max:
		# 	max = final_score
		# 	winner_url = row[1]
		jk = json.dumps(j)
		results.append(jk)
	# results.append(winner_url)
	return results
	




# j = main('http://link.food.com/view/560151201acbcd60208b4c8e5wkpp.ucxn/b0c2c481')
