'''
(c) 2017 Roberto Gobbetti
'''

import pandas as pd
import numpy as np
import time
import json
import urllib2

'''Under here importing things for nlp'''
import nltk
import string
import os
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.metrics.pairwise import linear_kernel # for cosine similarity



'''QUERY FUNCTION HERE'''

from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2

dbname = 'your_db_name'
username = 'your_username'

def query_from_name(wiki_name, input_name, beg, end):
    '''
    Performs a query in the database I previously built, where are stored historical data from nyt
    (all the articles from a certain date)

    :wiki_name: name as wikipedia extension
    :input_name: input name from user. Generally these two names are kept in case I possibly have two different recognizers
        for the same entity
    :beg: beginning time of query as string
    :end: anding time of query as string

    :return:
        :ds_from_sql: all the dataset queried with headline, lead paragraph, pulication date, all the text available, wub url,
            subsetcion name, as a pandas dataframe
        :top_words: list of top words with their tf-idf score, as a pandas dataframe
        :il: list of indices ordered as the index of the most relevant article and downward, as a python list
    '''


    name = wiki_name.replace('_',' ')

    '''
    Keeping the commented out text here and down, in case one wants to change to a different query with any of the 
    words in the wiki_name
    '''
#    name_list = preprocess(wiki_name.replace('_', ' '))
#    name = str(map(str,name_list))
#    name = name.replace('[', '{').replace(']', '}').replace('\'', '\"')
    
    #connect
    con = None
    con = psycopg2.connect(database = dbname, user = username)

    # query:
#    sql_query ="""
#    SELECT main_hl, lead_paragraph, pub_date, all_text, web_url, subsection_name
#    FROM nyt_archiv_accents
#    WHERE pub_date>='%s'
#        AND pub_date<='%s'
#        AND section_name != 'Briefing'
#        AND section_name != 'Paid Death Notices'
#        AND (
#            SELECT COUNT(DISTINCT word)
#            FROM
#                UNNEST(
#                    '%s'::text[]
#                ) s(word)
#            INNER JOIN
#            REGEXP_SPLIT_TO_TABLE(LOWER(nyt_archiv_accents.all_text), ' ') v (word) using (word)
#        ) > 0;
#    """ %(beg, end, name)

    # query:
    sql_query ="""
    SELECT main_hl, lead_paragraph, pub_date, all_text, web_url, subsection_name
    FROM nyt_archiv_accents
    WHERE pub_date>='%s'
        AND pub_date<='%s'
        AND section_name != 'Briefing'
        AND section_name != 'Paid Death Notices'
        AND (
                LOWER(all_text) ~* '%s'
                OR
                LOWER(all_text) ~* '%s'
            );
    """ %(beg, end, name, input_name)
    
    ds_from_sql = pd.read_sql_query(sql_query,con)
    top_words, il = generate_top_words(ds_from_sql.all_text, name.split('|'))
    
    return ds_from_sql, top_words, il



'''NLP functions under here'''

def stem_tokens(tokens, stemmer):
    stemmer = PorterStemmer()
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def lemm_tokens(tokens, lemmer):
    lemmer  = WordNetLemmatizer()
    lemmed = []
    for item in tokens:
        lemmed.append(lemmer.lemmatize(item))
    return lemmed

def preprocess(sentence):
    lemmer  = WordNetLemmatizer()
    sentence = sentence.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sentence)
    lems = lemm_tokens(tokens, lemmer)
    filtered_words = [w for w in lems if not w in stopwords.words('english')]
#    return ' '.join(filtered_words)
    return filtered_words

def generate_top_words(ser, search_list):
    '''
    takes df column with text
    returns most relevant words and most index list for the most relevant articles
    '''
    tokenizer = RegexpTokenizer(r'\w+')
    tfidf = TfidfVectorizer(tokenizer=tokenizer.tokenize, stop_words='english')#, ngram_range=(1,1))

    token_dict={}
    for i in range(len(ser)):
        token_dict[i] = ' '.join(preprocess(ser[i]))
#    print len(token_dict)
    
    if len(token_dict)==0:
        top_words = pd.DataFrame({'words':['' for i in range(10)], 'counts':[0 for i in range(10)]})
        index_list = []
    else:
        bow = tfidf.fit_transform(token_dict.values()) # Bag of Words
        bow_array = bow.toarray()

        vocabulary = tfidf.get_feature_names()
        counts = np.sum(bow_array, axis=0)
        
        top_words = pd.DataFrame({'words': vocabulary, 'counts': counts}).sort_values(by='counts',\
                            ascending=False).reset_index().drop('index', axis=1)

        new_words = any_in(list(top_words.words),search_list)
        top_words = top_words.loc[top_words.words.isin(new_words)]

        cos_tot = np.array([0.0 for i in range(bow.shape[0])])
        for i in range(bow.shape[0]):
            cos_tot += linear_kernel(bow[i], bow).flatten()
        
        index_list = list(cos_tot.argsort()) # 10 (or less) indices relative to the most relevant articles

    return top_words, index_list

def any_in(a,b):
    '''
    I need this to remove words that are part of my search and that contain number. Use it in generate_top_words
    '''
    c = [x for x in a if set(x.split()).isdisjoint(b)]
    c = [x for x in c if not any(y.isdigit() for y in x)]
    c = [x for x in c if len(x)>2]
    return c



'''NEW YORK TIMES stuff here'''

api_key = 'your_nyt_api_key'
search_url = 'https://api.nytimes.com/svc/search/v2/articlesearch.json'
archiv_url = 'https://api.nytimes.com/svc/archive/v1/2016/1.json'

def nyt_search(url, topic, begin, end, apikey):

    '''
    search articles in nyt

    :url: nyt endpoint
    :topic: search topic, string of comma separated topics 'barack,obama' (NO SPACES)
    :begin: start search this date, string '20160101'
    :end: end search this date, string '20160101'
    :apikey: api key for nyt

    :return: query result in dict form
    '''

    if topic == '':
        try:
            query = url+'?begin_date='+begin+'&end_date='+end+'&api_key='+apikey
            result = urllib2.urlopen(query)
            return json.loads(result.read())
        except urllib2.HTTPError as e:
            if e.code == 429:
                time.sleep(.1);
                return nyt_search(url, topic, begin, end, apikey)
            raise
    else:
        try:
            query = url+'?fq='+topic+'&begin_date='+begin+'&end_date='+end+'&api_key='+apikey
            result = urllib2.urlopen(query)
            return json.loads(result.read())
        except urllib2.HTTPError as e:
#            print e.code
#            print e.read()
            if e.code == 429:
#                print begin+'\n'
                time.sleep(.1);
                return nyt_search(url, topic, begin, end, apikey)
            raise
        
#    return json.loads(result.read())

def nyt_archiv(url, year, month, apikey):
    '''
    search nyt articles month by month. I used this to download my database

    :url: nyt endpoint
    :year: search in this year, as string
    :month: search in this month, as string
    :apikey: api key for nyt

    :return: query result in dict form
    '''
    query = url+'?year='+year+'&month='+month+'&api_key='+apikey

    return json.loads(urllib2.urlopen(query).read())

def search_and_organize(topic, begin, end, url=search_url, apikey=api_key):
    
    '''
    :DEPRECATED: really, I just stopped using this, but useful for API calls on the fly
    :topic: search topic, string of comma separated topics 'barack,obama' (NO SPACES)
    :begin: start search this date, string '20160101'
    :end: end search this date, string '20160101'
    :url: nyt endpoint
    :apikey: api key for nyt

    :return: pandas.DataFrame with relevant data. columns: headline, lead_paragraph, abstract, keyword, pub_date,
             section_name, subsection_name, web_url, all_text
             (this last one entry is headline + lead_paragraph + abstract + keyword, space separated)
    '''

    df = pd.DataFrame()
    day = pd.Timestamp(begin)
    split_topic = topic.lower().split(',')
    
    while day <= pd.Timestamp(end):
        day_s = day.strftime('%Y%m%d')
        res = nyt_search(url, topic, day_s, day_s, apikey)
        if res['response']['meta']['hits'] > 0:
            df_day = pd.DataFrame(res['response']['docs'])
            df_day['main_hl'] = df_day.headline.apply(lambda x: x['main'])
            df_day['keyw'] = df_day.keywords.apply(lambda x: ' '.join([key['value'] for key in x]))
            df_day = df_day[['main_hl','lead_paragraph', 'abstract', 'keyw',\
                             'pub_date', 'section_name', 'subsection_name', 'web_url']]
            df_day = df_day.rename(columns={'main_hl':'headline', 'keyw':'keyword'})

            df_day = df_day.fillna('')
            df_day['all_text'] =\
                (df_day['headline'] +' '+ df_day['lead_paragraph'] +' '+ df_day['abstract'] +' '+ df_day['keyword']).map(lambda x: x.lower())
            
            df_day = df_day[df_day.all_text.str.contains(topic.replace(',','|'))]
            df = df.append(df_day)
            day = day + pd.Timedelta('1d')
        else:
            day = day + pd.Timedelta('1d')


#    l=range(len(df))
    df = df.reset_index(drop=True)
    return df


'''the GUARDIAN stuff here'''

GUAapi = 'your_guardian_api_key'
GUAurl = 'http://content.guardianapis.com/search'

def search_guaapi(topic, st, en, url=GUAurl, apikey=GUAapi):
    '''
    saerch for articles in the Guardian. I did not end up using it, but explored possibilities with it.
    '''

    topic = topic.replace(' ','%20')
#    print topic
    query = url+'?from-date='+st+'&to-date='+en+'&q='+topic+'&api-key='+apikey
    result = urllib2.urlopen(query)
    return json.loads(result.read())

def s_and_o_guardian(topic, begin, end, url=GUAurl, apikey=GUAapi):
    '''
    Like search_and_organize function above. Only returns title, pub_date and url from the Guardian
    '''
    df = pd.DataFrame()
    day = pd.Timestamp(begin)
    
    while day <= pd.Timestamp(end):
        day_s = day.strftime('%Y-%m-%d')
        res = search_guaapi(topic, day_s, day_s, url=url, apikey=apikey)
        if res['response']['total'] > 0:
            df_day = pd.DataFrame(res['response']['results'])
            df_day = df_day[['webTitle', 'webPublicationDate', 'webUrl']]
            df_day = df_day.rename(columns={'webTitle':'all_text', 'webPublicationDate':'pub_date', 'webUrl':'url'})

            df = df.append(df_day)
            day = day + pd.Timedelta('1d')
        else:
            day = day + pd.Timedelta('1d')

    df = df.reset_index(drop=True)
    return df



