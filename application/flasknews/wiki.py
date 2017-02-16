''''
(c) 2017 Roberto Gobbetti
'''

import pandas as pd
import numpy as np
from mwviews.api import PageviewsClient
import wikipedia
from datetime import date
import time
import datetime
from unidecode import unidecode

import nyt
from anomalies import find_anom

'''Under here importing things for nlp (although doing all of this in nyt now, leaving it as an old function uses these packages)'''
import nltk
import string
import os
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer


def download_pageviews(entities=None, start='20150701', end=None, access='desktop', agent='user', limit=10000):

    """
    Download pageviews from Wikipedia

    :param entities: A list of entities (Wikipedia pages) to get pageview data for
    :param start: The start date of the range over which to collect data;
        2015-07-01 is the earliest supported by the API
    :param end: The end date of the range, or None for today
    :param access: The method by which Wikipedia was accessed (default: desktop)
    :param agent: The user agent accessing Wikipedia (default: user)
    :param limit: The number of most-trafficked entities to return data for, if no entities are specified in the call
    :return: A DataFrame of entities x pageviews by day
    """
    
    if end is None:
        end = datetime.date.today().strftime('%Y%m%d')
    
    p = PageviewsClient()
    dates = pd.date_range(start=start, end=end)

    #str -> list
    if type(entities) is str:
        
        entities = [entities]
    
    # if entities aren't passed in, get the daily top entities for the period
    if entities is None:
        df_pvs = None
    
        for d in dates:
            try:
                df = pd.DataFrame(p.top_articles('en.wikipedia', year=d.year, month=d.month,\
                                                 day=d.day, limit=limit, access=access))
            except:
                continue

            df = df.set_index('article').rename(columns={'views': d})[[d]]

            if df_pvs is None:
                df_pvs = df
            else:
                df_pvs = df_pvs.join(df, how='outer')

        entities = df_pvs.index.values.tolist()
    
    for i in range(len(entities)):
        try:
            entities[i] = unidecode(wikipedia.page(entities[i]).title)
        except wikipedia.exceptions.DisambiguationError as e:
            print 'I could not understand that, please check your spelling or be more specific'
            print 'Error: {0}'.format(e)
            avere = pd.DataFrame(columns=['NONE'])
            return avere
        except wikipedia.exceptions.PageError as e:
            print 'I could not understand that, please check your spelling or be more specific'
            print 'Error: {0}'.format(e)
            avere = pd.DataFrame(columns=['NONE'])
            return avere
        
    search = p.article_views('en.wikipedia', entities, start=start, end=end, access=access, agent=agent)
    df = pd.DataFrame.from_dict(search, orient='index')
    
    return df



def wiki_to_words(entities=None, url=search_url, apikey=api_key, start='20150701', end=None, window=30, tolerance=1.5):

    '''
    NOT USING IT ANYMORE: leaving it here as it might be useful anyway for other projects
    All the relevant stuff that is done here, is now performed partially directly in the view.py file and partially in the nyt
    module by the query and NLP functions.

    Performs a few functions automatically:
    _ uses download_pageviews to get the time series of wiki pageviews of the entities requested
    _ uses anom.find_anom to find anomalies in the time series
    _ uses nyt.nyt_search to find the articles +-1 day from the anomalies within start and end, if any

    :entities: list or string of search terms
    :url: search url for nyt search
    :apikey: api key for nyt
    :start: starting point for looking for articles in nyt
    :end: ending point for looking for articles in nyt
    :window: window for rolling mean in anom.find_anom
    :tolerance: tolerance for anomaly detection (# of \sigma away from mean) as in anom.find_anom

    :return: _ top_w: dict {entity: {anomaly timestamp: [list of top words]}}
             _ ds: dataframe of all wiki pageviews data from start to end
             _ all_anom: dict {entity: {anomaly timestamp: # of pageviews}}
    '''
    
    if end == None:
        end = datetime.date.today().strftime('%Y%m%d')
    
    ds = download_pageviews(entities=entities, end=None)

    tokenizer = RegexpTokenizer(r'\w+')
    # NOTICE: I am looking only for bi- and tri- grams as I believe them to be the most desriptive for events
    tfidf = TfidfVectorizer(tokenizer=tokenizer.tokenize, stop_words='english', ngram_range=(2,3))

    
#    all_entities = {}
    all_anom = {}
    top_w = {}
    
    tot_interval = [pd.Timestamp(start) <= d <= pd.Timestamp(end) for d in ds.index] #dates of interest

#    print ds.columns
    
    for column in ds.columns:
        anom, norm, anom_all = find_anom(ds[column], window=window, tolerance=tolerance)
        
        dates_interval = [pd.Timestamp(start) <= d <= pd.Timestamp(end) for d in anom.index] #anomaly dates in interval
        anoms_interval = anom[dates_interval]
        
        topic = column.replace('_',',').lower()

        df = nyt.search_and_organize(topic, start, end)

        token_dict={}
        for i in range(len(df.all_text)):
            token_dict[i] = ' '.join(nyt.preprocess(df.all_text[i]))

        bow = tfidf.fit_transform(token_dict.values()) # Bag of Words
        bow_array = bow.toarray()

        vocabulary = tfidf.get_feature_names()
        counts = np.sum(bow_array, axis=0)
        
        top_words = pd.DataFrame({'words': vocabulary, 'counts': counts}).sort_values(by='counts',\
            ascending=False).reset_index().drop('index', axis=1)

        top_w[column] = top_words
        all_anom[column] = anoms_interval.to_dict()
        
    return top_w, ds[tot_interval], all_anom 

