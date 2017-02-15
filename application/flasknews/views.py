from flasknews import app
from flask import Flask, render_template, request, make_response
import wiki
import nyt
import anomalies
import re


from io import BytesIO
import urllib
from matplotlib import pyplot as plt
import pandas as pd

import datetime as dt
import json

from wordcloud import WordCloud

@app.route('/')
@app.route('/index')
def index():
	return render_template('input.html')


@app.route('/input')
def send_input():
    return render_template("input.html")

@app.route('/output')
@app.route('/uncertainty')
def show_plot():
	entities = request.args.get('entity')
	if entities == '':
		entities = 'Barack Obama'
	ds = wiki.download_pageviews(entities=str(entities))

	if len(ds) == 0:
		return render_template("uncertainty.html")

	wiki_name = ds.columns[0]
	a, n, aa = anomalies.find_anom(ds[wiki_name], window=30, tolerance=6, threshold=1000)

	ds = ds.fillna(0)

	ds = ds.reset_index()
	ds = ds.rename(columns={'index':'x', wiki_name:'y'})
	ds['color'] = '#3D3DFF'
	ds.loc[ds['x'].isin(aa.index),'color'] = '#E62E00'

	#Getting day preceding 1970/1/1 in order to count full first day as well (off by 1 day in js otherwise)
	ds['x'] = ds['x'].apply(lambda t: 1000 * int( ( t - dt.datetime(1969,12,31) ).total_seconds() ))

	the_data = []
	the_data.append(dict(values=[]))
	for i in range(len(ds)):
		the_data[0]['values'].append(dict(x=ds.x[i], y=ds.y[i], color=ds.color[i]))

	the_dates = []
	
	the_words = []

	for i in range(len(a)):
		if i == len(a)-1:
			dates = [x for x in aa.index if x >= a.index[i]]
			beg = dates[0]  - pd.Timedelta(days=1)
			end = dates[-1] + pd.Timedelta(days=2)
			nyt_ds, top_words, il = nyt.query_from_name(wiki_name, entities, beg, end)

			dd = a.index[i].strftime('%Y-%m-%d')
			if len(il)>0:

				all_the_words = ' '.join(list(nyt_ds.all_text))
				wordcloud = WordCloud(background_color="white", max_words=50, width = 200, height = 200).generate(all_the_words)
				plt.imshow(wordcloud)
				plt.axis('off')
				imgdata = BytesIO()
				plt.savefig(imgdata, format='svg')
				svg_data = '<svg' + imgdata.getvalue().split('<svg')[1]  # this is svg data

				the_dates.append(dict(date = dd, wordcloud=svg_data))

				for j in reversed(il[-10:]):
					the_words.append(dict(date=dd, article=nyt_ds.main_hl[j], url=nyt_ds.web_url[j]))
			else:
				the_words.append(dict(date=dd, article='No articles found', url=''))
				the_dates.append(dict(date = dd, wordcloud=''))

		else:
			dates = [x for x in aa.index if x >= a.index[i] and x<a.index[i+1]]
			beg = dates[0]  - pd.Timedelta(days=1)
			end = dates[-1] + pd.Timedelta(days=2)
			nyt_ds, top_words, il = nyt.query_from_name(wiki_name, entities, beg, end)

			dd = a.index[i].strftime('%Y-%m-%d')
			if len(il)>0:

				all_the_words = ' '.join(list(nyt_ds.all_text))
				wordcloud = WordCloud(background_color="white", max_words=50, width = 250, height = 250).generate(all_the_words)
				plt.imshow(wordcloud)
				plt.axis('off')
				imgdata = BytesIO()
				plt.savefig(imgdata, format='svg')
				svg_data = '<svg' + imgdata.getvalue().split('<svg')[1]  # this is svg data

				the_dates.append(dict(date = dd, wordcloud=svg_data))

				for j in reversed(il[-10:]):
					the_words.append(dict(date=dd, article=nyt_ds.main_hl[j], url=nyt_ds.web_url[j]))
			else:
				the_words.append(dict(date=dd, article='No articles found', url=''))
				the_dates.append(dict(date = dd, wordcloud=''))


	return render_template('output.html',\
		the_data=the_data, the_dates = the_dates, the_words=the_words, entities=wiki_name.replace('_', ' '))

@app.route('/slides')
def slides():
	return render_template('slides.html')
