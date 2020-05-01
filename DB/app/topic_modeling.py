import os
import re

import wordcloud
import json
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd

import spacy
from spacy.lang.en.stop_words import STOP_WORDS as en_stopwords
from spacy.lang.pt.stop_words import STOP_WORDS as pt_stopwords
from spacy.lang.es.stop_words import STOP_WORDS as es_stopwords

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import ngrams

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

import pyLDAvis.gensim


def get_stopwords_caio(lang):
    if lang == 'en':
        # df_stopwords = set(nltk.corpus.stopwords.words('english'))
        df_stopwords = set(en_stopwords)
        custom_stopwords = set(['olympics', 'olympic', 'london',
                                'rio', 'legacy', '2012', '2016',
                                'said', 'caption', 'image', 'copyright',
                                'getty', 'bbc', 'theguardian', 'dailymail',
                                'telegraph', 'images', 'bookmark', 'reddit',
                                'd','s','t','m','n','ve', 'll'])
        return df_stopwords | custom_stopwords
    elif lang == 'pt':
        # df_stopwords = set(nltk.corpus.stopwords.words('english'))
        df_stopwords = set(pt_stopwords)
        custom_stopwords = set(['legado', 'olimpico', 'london',
                                'olímpica', 'londres', 'rio', 'legado',
                                'olímpico', '2012', '2016'])
        return df_stopwords | custom_stopwords


def get_stopwords_daniela(lang):
    if lang == 'en':
        df_stopwords = set(en_stopwords)
        custom_stopwords = set(['eurosceptic', 'euroscept', 'eurosceptics',
                                'euroscepticism', 'said', 'caption', 'image', 'copyright',
                                'getty', 'theguardian', 'dailymail',
                                'bookmark', 'reddit',
                                'images', 'd', 'll', 'm', 'n',
                                's', 't', 've'])
        return df_stopwords | custom_stopwords
    elif lang == 'es':
        df_stopwords = set(es_stopwords)
        custom_stopwords = set(['euroesceptic', 'euroescept', 'euroesceptics',
                                'euroescepticismo'])
        return df_stopwords | custom_stopwords

def topic_model_lda_auto(data_all, lemmatize, use_bigrams, use_trigrams, df_stopwords, nlp):
    def sent_to_words(docs):
        for text in docs:
            text = re.sub("\'", "", re.sub('\s+', ' ', text))  ## Remove newlines and single quotes
            yield (gensim.utils.simple_preprocess(str(text), deacc=True))

    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in df_stopwords and len(word) > 2] for doc in
                texts]

    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]

    def make_trigrams(texts):
        return [trigram_mod[bigram_mod[doc]] for doc in texts]

    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent))
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out

    def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
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
        for num_topics in range(start, limit, step):
            model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                    id2word=dictionary,
                                                    num_topics=num_topics,
                                                    random_state=100,
                                                    update_every=1,
                                                    chunksize=20,
                                                    passes=10,
                                                    alpha='auto',
                                                    per_word_topics=True)
            model_list.append(model)
            coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
            coherence_values.append(coherencemodel.get_coherence())

        return model_list, coherence_values

    ## Text Preprocessing
    data_words = list(sent_to_words(data_all))

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)  # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    data_words = remove_stopwords(data_words)

    if use_trigrams:
        data_words = make_trigrams(data_words)
    elif use_bigrams:
        data_words = make_bigrams(data_words)

    if lemmatize:
        data_lemmatized = lemmatization(data_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    else:
        data_lemmatized = data_words

    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)

    # Create Corpus
    texts = data_lemmatized

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_lemmatized,
                                                            start=2, limit=8, step=1)
    print(coherence_values)
    max_i = np.argmax(coherence_values)

    # Compute Perplexity
    print('\nPerplexity: ',
          model_list[max_i].log_perplexity(corpus))  # a measure of how good the model is. lower the better.

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=model_list[max_i], texts=data_lemmatized, dictionary=id2word,
                                         coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)

    return model_list[max_i], corpus, id2word, coherence_values, max_i


def topic_model_lda_k(data_all, lemmatize, use_bigrams, use_trigrams, df_stopwords, nlp, num_k):
    def sent_to_words(docs):
        for text in docs:
            text = re.sub("\'", "", re.sub('\s+', ' ', text))  ## Remove newlines and single quotes
            yield (gensim.utils.simple_preprocess(str(text), deacc=True))

    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in df_stopwords and len(word) > 2] for doc in
                texts]

    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]

    def make_trigrams(texts):
        return [trigram_mod[bigram_mod[doc]] for doc in texts]

    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent))
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out

    ## Text Preprocessing
    data_words = list(sent_to_words(data_all))

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)  # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    data_words = remove_stopwords(data_words)

    if use_trigrams:
        data_words = make_trigrams(data_words)
    elif use_bigrams:
        data_words = make_bigrams(data_words)

    if lemmatize:
        data_lemmatized = lemmatization(data_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    else:
        data_lemmatized = data_words

    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)

    # Create Corpus
    texts = data_lemmatized

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=num_k,
                                            random_state=100,
                                            update_every=1,
                                            chunksize=20,
                                            passes=10,
                                            alpha='auto',
                                            per_word_topics=True)


    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=model, texts=data_lemmatized, dictionary=id2word,
                                         coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)


    return model, corpus, id2word