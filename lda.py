#
#	From a command line, download the English language model
#	for spacy.
#
#	python -m spacy download en

#	Retrieve stopwords list from nltk.
#	Stopwords are words that add little meaning
#	to a text.  This includes function words 
#	such as articles (a, an, the);
#

#import nltk

#nltk.download('stopwords')

#
#	----- Package imports -----
#

#	Import web browser interface.

import webbrowser

#	Import regular expressions package.

import re

#	Import numpy which supports fundamental scientific
#	computing operations for Python.

import numpy as np

#	Import pandas which provides data structures
#	and data analysis tools for Python.
	
import pandas as pd

#	Import pprint which provides for "pretty printing"
#	Python data structures.

from pprint import pprint

#	Import Gensim which provides a wide variety
#	of topic modeling methods for Python.

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel, HdpModel, LdaModel, LsiModel

#	Import Sklearn which provides convenient grid search
#	methods for finding the number of topics to use.

from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV

#	Import random sampler.

from random import sample

#	Import spacy which provides basic natural
#	language processing methods for Python.
#	We could also use nltk, but spacy is more
#	up-to-date.

import spacy

#	Import default English stop words from spacy.

from spacy.lang.en.stop_words import STOP_WORDS

#	Import graphical display tools.
#	
#	pyLDAvis is a popular way to display the
#	results of a topic modeling process. 
#	You'll see this type of output in many
#	articles displaying the results of a topic
#	modeling process.

import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
#%matplotlib inline

#	Enable logging for gensim.

import logging
import warnings

#	Import the wordcloud library.

from wordcloud import WordCloud

#	Import collections.

import collections

#	----- Start main program. -----

def main():

#	Fix collections import warning.

#	try:
#		collectionsAbc = collections.abc
#	except AttributeError:
#		collectionsAbc = collections

	try:
		from collections.abc import Iterable
	except ImportError:
		from collections import Iterable

#	Enable logging.

	logging.basicConfig( format='%(asctime)s : %(levelname)s : %(message)s' , 
		level=logging.ERROR )

	warnings.filterwarnings( "ignore" , category=DeprecationWarning )

#	----- Get English stopwords list from spacy. -----

	stop_words = STOP_WORDS

#	----- Extend the stopword list by adding any others we want here. -----

	stop_words.add( 'from') 
	stop_words.add( 'subject' ) 
	stop_words.add( 're' ) 
	stop_words.add( 'edu' ) 
	stop_words.add( 'use' ) 
	stop_words.add( '_' ) 

#	----- Import data to analyze using pandas. -----

	df = pd.read_json( 'https://raw.githubusercontent.com/selva86/datasets/master/newsgroups.json' )

	print( df.target_names.unique() )

	df.head()

	def clean_text( data ):

#		Remove email addresses.

		data = [ re.sub( '\S*@\S*\s?', '', sent ) for sent in data ]

#		Remove newline characters and normalize whitespace.

		data = [ re.sub( '\s+', ' ', sent ) for sent in data ]

#		Remove single quotes.

		data = [ re.sub( "\'", "", sent ) for sent in data ]

		return data
		
#	-----	Clean text using a series of regular expressions. -----

#		Convert data values to a list.

	data = df.content.values.tolist()

#		Extract a random sample of 1000 entries from the data values.

	data = sample( data , 1000 )
	
	data = clean_text( data )
	
	pprint( data[:1] )

#		Convert text to list of tokens using gensim's
#		simple_preprocess utility.  Specifying deacc=True
#		removes punctuation.
		
	def sent_to_words( sentences ):
		for sentence in sentences:
			yield( gensim.utils.simple_preprocess( str( sentence ), deacc=True ) )

	data_words = list( sent_to_words( data ) )

	print( data_words[:1] )

#		Build the bigram and trigram models.
#		Set higher threshold to generate fewer phrases.

	bigram = gensim.models.Phrases( data_words, min_count=5, threshold=100 ) 
	trigram = gensim.models.Phrases( bigram[data_words], threshold=100 )  

#		Faster way to get a sentence clubbed as a trigram/bigram.

	bigram_mod = gensim.models.phrases.Phraser( bigram )
	trigram_mod = gensim.models.phrases.Phraser( trigram )

#		Display trigram example.

	print( trigram_mod[ bigram_mod[ data_words[ 0 ] ] ] )

#		Define functions for stopwords, bigrams, trigrams and lemmatization.

	def remove_stopwords( texts ):
		return [ [ word for word in simple_preprocess( str( doc ) ) if word not in stop_words ] for doc in texts ]

	def make_bigrams( texts ):
		return [ bigram_mod[ doc ] for doc in texts ]

	def make_trigrams( texts ):
		return [ trigram_mod[ bigram_mod[ doc ] ] for doc in texts ]

	def lemmatize( texts , allowed_postags=[ 'NOUN', 'ADJ', 'VERB', 'ADV' ] ):
		"""https://spacy.io/api/annotation"""
		texts_out = []
		for sent in texts:
			doc = nlp( " ".join( sent ) ) 
			texts_out.append( [ token.lemma_ for token in doc if token.pos_ in allowed_postags ] )
		return texts_out

#	Remove stop words.

	data_words_nostops = remove_stopwords( data_words )

#	Construct bigrams.

	data_words_bigrams = make_bigrams( data_words_nostops )

#	Initialize spacy 'en' model, keeping only tagger component
#	by disabling the parser and named entity extractor.

	nlp = spacy.load( 'en' , disable=[ 'parser' , 'ner' ] )
#	nlp = spacy.load( "en_core_web_sm" , disable=[ 'parser' , 'ner' ] )

#	Lemmatize words, keeping only nouns, adjectives, verbs, and adverbs.

	data_lemmatized = lemmatize(
						data_words_bigrams, 
						allowed_postags=[ 'NOUN', 'ADJ', 'VERB', 'ADV' ] )

	print( data_lemmatized[:1] )

#	Create Dictionary.

	id2word = corpora.Dictionary( data_lemmatized )

#	Create Corpus.

	texts = data_lemmatized

#	Compute term document frequencies.

	corpus = [ id2word.doc2bow( text ) for text in texts ]

#	View start of processed corpus.

	print( corpus[:1] )

#	Build LDA model.

	lda_model = gensim.models.ldamodel.LdaModel(
					corpus = corpus,
					id2word = id2word,
					num_topics = 20, 
					random_state = 100,
					update_every = 1,
					chunksize = 100,
					passes = 10,
					alpha = 'auto',
					per_word_topics = True )

#	Print keywords in the 10 topics.

	pprint( lda_model.print_topics() )
	doc_lda = lda_model[ corpus ]

#	Compute perplexity, a measure of how good the model is. 
#	The lower the perplexity value the better.

	print( '\nPerplexity: ', lda_model.log_perplexity( corpus ) )

#	Compute Coherence Score.
#	The higher the coherence value the better.

	coherence_model_lda = CoherenceModel(
							model = lda_model, 
							texts = data_lemmatized, 
							dictionary = id2word, 
							coherence = 'c_v' )

	coherence_lda = coherence_model_lda.get_coherence()

	print( '\nCoherence Score: ' , coherence_lda )

#	Visualize the topics.

#	pyLDAvis.enable_notebook()		
	vis = pyLDAvis.gensim.prepare( lda_model, corpus, id2word )
#	vis
	
	pyLDAvis.save_html( vis , 'ldavis.html' )
#	webbrowser.open_new_tab( 'lda.html' )	
	
	def compute_coherence_values(
		dictionary, corpus, texts, limit, 
		start = 2, step = 3, sort = True ):

		"""
		Compute c_v coherence for various number of topics

		Parameters:
		----------
		dictionary : Gensim dictionary
		corpus : Gensim corpus
		texts : List of input texts
		limit : Max num of topics

		Returns:
		-------
		model_list : List of LDA topic models
		coherence_values : Coherence values corresponding to the LDA model with respective number of topics
		"""

		coherence_values = []
		model_list = []
		
		for num_topics in range( start, limit, step ):
#			model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
			lda_model = gensim.models.ldamodel.LdaModel( corpus=corpus,
											num_topics=num_topics,
											id2word=id2word,
											random_state=100,
											update_every=1,
											chunksize=100,
											passes=10,
											alpha='auto',
											per_word_topics=True )

			model_list.append( lda_model )

			coherencemodel = CoherenceModel(	model = lda_model, 
												texts = texts, 
												dictionary = dictionary, 
												coherence = 'c_v' )

			coherence_values.append( coherencemodel.get_coherence() )

		return model_list, coherence_values

#	Compute coherence values for sequence of LDA models.
#	Can take a long time to run.

	model_list, coherence_values = compute_coherence_values( 	
										dictionary = id2word, 
										corpus = corpus, 
										texts = data_lemmatized, 
										start = 2, 
										limit = 40, 
										step = 6 )

#	Display plot of coherence values for number of topics.

	limit = 40; 
	start = 2; 
	step = 6;

	x = range( start, limit, step )

	plt.plot( x , coherence_values )
	plt.xlabel( "Number of Topics" )
	plt.ylabel( "Coherence" )
	plt.legend( ( "coherence_values" ) , loc='best' )
#	plt.show()
	plt.savefig( "coherences.png" )

#	Print the coherence scores.

	for m, cv in zip( x , coherence_values ):
		print( "Num Topics =" , m , " has Coherence Value of" , round( cv , 4 ) )

#	Select the model and print the topics.

	optimal_model = model_list[ 3 ]
	model_topics = optimal_model.show_topics( formatted = False )
	pprint( optimal_model.print_topics( num_words = 10 ) )

	def format_topics_sentences( ldamodel = lda_model, corpus = corpus, texts = data ):
    	# Initialize output.
		sent_topics_df = pd.DataFrame()

		# Get main topic in each document.
		
		for i, row_list in enumerate( ldamodel[ corpus ] ):
			row = row_list[ 0 ] if ldamodel.per_word_topics else row_list 
			row = sorted( row, key = lambda x: ( x[ 1 ] ) , reverse = True )

        # Get the Dominant topic, Percent Contribution and Keywords
        # for each document.
        
			for j, ( topic_num , prop_topic ) in enumerate( row ):
				if j == 0:  # => dominant topic
					wp = ldamodel.show_topic( topic_num )
					topic_keywords = ", ".join( [ word for word, prop in wp ] )
					sent_topics_df = sent_topics_df.append(	
						pd.Series(	
							[	int( topic_num ), 
								round( prop_topic , 4 ) , 
								topic_keywords
							]
						) , 
						ignore_index = True
					)
				else:
					break

		sent_topics_df.columns = [ 
			'Dominant_Topic' , 
			'Perc_Contribution' , 
			'Topic_Keywords' ]

		#	Add original text to the end of the output.
		
		contents = pd.Series( texts )
		sent_topics_df = pd.concat( [ sent_topics_df , contents ] , axis = 1 )
		return( sent_topics_df )

	df_topic_sents_keywords = format_topics_sentences(
			ldamodel = optimal_model, 
			corpus = corpus, 
			texts = data
		)

#	Format.

	df_dominant_topic = df_topic_sents_keywords.reset_index()
	df_dominant_topic.columns = [	
			'Document_#' , 
			'Dominant_Topic', 
			'Topic_%_Contribution', 
			'Keywords', 
			'Text'
		]

#	Show.

	df_dominant_topic.head( 10 )

#	Group top 5 sentences under each topic.

	sent_topics_sorteddf_mallet = pd.DataFrame()

	sent_topics_outdf_grpd = df_topic_sents_keywords.groupby( 'Dominant_Topic' )

	for i, grp in sent_topics_outdf_grpd:
		sent_topics_sorteddf_mallet = pd.concat( [ sent_topics_sorteddf_mallet, 
             grp.sort_values( [ 'Perc_Contribution' ], ascending = [ 0 ] ).head( 1 ) ], 
             axis = 0 )

#	Reset Index.

	sent_topics_sorteddf_mallet.reset_index( drop=True , inplace=True )

#	Format.

	sent_topics_sorteddf_mallet.columns = [ 
		'Topic_Num', 
		"Topic % Contrib", 
		"Keywords", 
		"Text"
	]

#	Show.

	sent_topics_sorteddf_mallet.head()

#	Number of Documents for Each Topic.

	topic_counts = df_topic_sents_keywords[ 'Dominant_Topic' ].value_counts()

#	Percentage of Documents for Each Topic.

	topic_contribution = round( topic_counts/topic_counts.sum() , 4 )

#	Topic Number and Keywords.

	topic_num_keywords = df_topic_sents_keywords[ 
		[ 'Dominant_Topic' , 'Topic_Keywords' ] ]

#	Concatenate Column wise.

	df_dominant_topics = pd.concat(
			[ topic_num_keywords, topic_counts, topic_contribution ] , 
			axis=1
		)

#	Change column names.

	df_dominant_topics.columns = [ 
		'Dominant_Topic', 
		'Topic_Keywords', 
		'Num_Documents', 
		'Perc_Documents'
	]

#	Show dominant topics.

	df_dominant_topics

if __name__== "__main__":
  main()
 