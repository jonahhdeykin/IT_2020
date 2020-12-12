import pandas as pd
import requests
import json
import csv
import time
import datetime
import pytz
import pickle
import os
import bz2
import paq
import random

def getPushshiftData(search_type, after, before, sub):
	url = 'https://api.pushshift.io/reddit/search/'+ str(search_type) +'/?size=500&after='+str(after)+'&before='+str(before)+'&subreddit='+str(sub)
	r = requests.get(url)
	data = json.loads(r.text)
	return data['data']


def get_day(search_type, sub, date):
	timez = pytz.timezone("America/New_York")
	date =  timez.localize(date).replace(hour=16, minute=0)
	timed = datetime.timedelta(days=1)
	before_date = date - timed
	fetched = False

	while not fetched:
		try:
			data = getPushshiftData(search_type, before_date.strftime('%s'), date.strftime('%s'), sub)
			fetched = True
		except:
			pass
	return data

def score_zip(x, y):
	x = bytes(x, encoding='utf8')
	y = bytes(y, encoding='utf8')
	c_x = len(bz2.compress(x))
	c_y = len(bz2.compress(y))
	cat_xy = x + y
	c_xy = len(bz2.compress(cat_xy))
	cat_yx = y + x
	c_yx = len(paq.compress(cat_yx))

	return (min(c_xy, c_yx) - min(c_x, c_y))/max(c_x, c_y)
def score_paq(x, y):
	x = bytes(x, encoding='utf8')
	y = bytes(y, encoding='utf8')
	c_x = len(paq.compress(x))
	c_y = len(paq.compress(y))
	cat_xy = x + y
	c_xy = len(paq.compress(cat_xy))
	cat_yx = y + x
	c_yx = len(paq.compress(cat_yx))

	return (min(c_xy, c_yx) - min(c_x, c_y))/max(c_x, c_y)

def scrape_data():
	text_data = []
	with open('prices.csv', 'r') as f:
		reader = csv.reader(f)
		title = next(reader)
		for row in reader:
			date_info = row[0].split('-')
			date = datetime.datetime(year=int(date_info[0]), month= int(date_info[1]), day=int(date_info[2]))
			text_list = []
			for sub_r in ['SecurityAnalysis', 'Finance', 'WallStreetBets', 'Options', 'Forex', 'Investing', 'Stocks']:

				for post in get_day('submission', sub_r, date):
					text = ''
					try:
						text = post['title'] + ' '
					except:
						pass

					try:
						text = text + post['selftext']
					except:
						pass
					text_list.append((text , post['created_utc']))

				for post in get_day('comment', sub_r, date):
					text_list.append((post['body'] , post['created_utc']))

				print(sub_r)

			text_list.sort(key = lambda x: x[1])
			total_text = ''
			for text in text_list:
				total_text = total_text + ' ' + text[0]
			text_data.append((row[0], total_text))
			print(row[0])
	with open('data.txt', 'wb') as f:
		pickle.dump(text_data, f)

def gen_mat_vals(t_list, z_val, p_vals, dates):
	texts = []
	count = 0
	for day in t_list:
		t_count = 0
		if day[0] not in dates:

			for t in texts:
				z_vals.append(score_zip(day[1], t))
				p_vals.append(score_paq(day[1], t))
				t_count += 1
				if t_count % 5 == 0:
					print(count, t_count)
			z_vals.append(0)
			p_vals.append(0)
			dates.append(day[0])

		texts.append(day[1])
		count += 1


		if count % 5 == 0:
			with open('dates.txt', 'wb') as f:
				pickle.dump(dates, f)

			with open('z_vals.txt', 'wb') as f:
				pickle.dump(z_vals, f)

			with open('p_vals.txt', 'wb') as f:
				pickle.dump(p_vals, f)



if __name__ == '__main__':
	with open('data.txt', 'rb') as f:
		text_list = pickle.load(f)

	text_list = iter(text_list)
	with open('dates.txt', 'rb') as f:
		dates = pickle.load(f)

	with open('z_vals.txt', 'rb') as f:
		z_vals = pickle.load(f)

	with open('p_vals.txt', 'rb') as f:
		p_vals = pickle.load(f)


	gen_mat_vals(text_list, z_vals, p_vals, dates)



	gen_mat_vals(text_list)







