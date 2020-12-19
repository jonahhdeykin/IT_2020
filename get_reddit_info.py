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
import lzma
import paq
import random
import math
import numpy as np
import sklearn.cluster as SKCL
import deflate

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

def score_zip(x, y, c_x, c_y):
	x = bytes(x, encoding='utf8')
	y = bytes(y, encoding='utf8')

	cat_xy = x + y
	c_xy = len(bz2.compress(cat_xy))
	cat_yx = y + x
	c_yx = len(bz2.compress(cat_yx))

	return (min(c_xy, c_yx) - min(c_x, c_y))/max(c_x, c_y)

def score_def(x, y, c_x, c_y):
	x = bytes(x, encoding='utf8')
	y = bytes(y, encoding='utf8')

	cat_xy = x + y
	c_xy = len(deflate.gzip_compress(cat_xy))
	cat_yx = y + x
	c_yx = len(deflate.gzip_compress(cat_yx))

	return (min(c_xy, c_yx) - min(c_x, c_y))/max(c_x, c_y)

def score_lzma(x, y, c_x, c_y):
	x = bytes(x, encoding='utf8')
	y = bytes(y, encoding='utf8')

	cat_xy = x + y
	c_xy = len(lzma.compress(cat_xy))
	cat_yx = y + x
	c_yx = len(lzma.compress(cat_yx))

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

def gen_mat_vals(t_list, z_vals, p_vals, m_vals, dates, precomputed, path):
	texts = []
	count = 0
	for day in t_list:
		t_count = 0
		day_l = precomputed[day[0]]
		if day[0] not in dates:

			for t in texts:
				z_vals.append(score_zip(day[1], t[1], day_l[1], precomputed[t[0]][1]))
				p_vals.append(score_paq(day[1], t[1], day_l[0], precomputed[t[0]][0]))
				m_vals.append(score_lzma(day[1], t[1], day_l[2], precomputed[t[0]][2]))
				t_count += 1
				if t_count % 5 == 0:
					print(count, t_count)

			z_vals.append(0)
			p_vals.append(0)
			m_vals.append(0)
			dates.append(day[0])

		texts.append(day)
		count += 1

		if count % 5 == 0:
			with open('dates_' + path + '.txt', 'wb') as f:
				pickle.dump(dates, f)

			with open('z_vals_' + path + '.txt', 'wb') as f:
				pickle.dump(z_vals, f)

			with open('p_vals_' + path + '.txt', 'wb') as f:
				pickle.dump(p_vals, f)

			with open('m_vals_'  + path + '.txt', 'wb') as f:
				pickle.dump(m_vals, f)

def build_distance_matrix(t_list,  p_vals, z_vals, m_vals):
	dim = int(math.sqrt(len(p_vals)*2))

	p_mat = np.zeros((dim, dim))
	z_mat = np.zeros((dim, dim))
	m_mat = np.zeros((dim, dim))

	index = 0
	for i in range(0, dim):
		for j in range(0, i + 1):
			p_mat[dim-1-i][dim-1-j] = p_vals[index]
			p_mat[dim-1-j][dim-1-i] = p_vals[index]

			z_mat[dim-1-i][dim-1-j] = z_vals[index]
			z_mat[dim-1-j][dim-1-i] = z_vals[index]

			m_mat[dim-1-i][dim-1-j] = m_vals[index]
			m_mat[dim-1-j][dim-1-i] = m_vals[index]
			index += 1

	return(p_mat, z_mat, m_mat)

def compute_silhouette_pre(labels, dist):
	unique_ls = np.unique(labels)
	s_score = 0
	for l in unique_ls:
		if l != -1:
			locs = np.where(labels == l)
			locs = locs[0]
			if locs.size > 1:
				for loc_1 in locs:
					a = 0
					for loc_2 in locs:
						if loc_1 != loc_2:
							a += dist[loc_1][loc_2]
					a = a/(locs.size-1)
					b = 100
					for l2 in unique_ls:
						if l != l2 and l2 != -1:
							t_val = 0
							o_locs = np.where(labels == l2)
							o_locs = o_locs[0]
							for loc_3 in o_locs:
								t_val += dist[loc_1][loc_3]
							t_val = t_val/o_locs.size

							if t_val < b:
								b = t_val
					print(a, b)
					s_score += (b-a)/max(a, b)

	try:
		s_score = s_score/(labels.size - np.where(labels == -1)[0].size)
	except:
		return 0

	return s_score

def compute_silhouette(labels, points):
	unique_ls = np.unique(labels)
	s_score = 0
	for l in unique_ls:
		if l != -1:
			locs = np.where(labels == l)
			locs = locs[0]
			if locs.size > 1:
				for loc_1 in locs:
					a = 0
					for loc_2 in locs:
						if loc_1 != loc_2:
							a += np.linalg.norm(points[loc_1] - points[loc_2])
					a = a/(locs.size-1)
					b = 100
					for l2 in unique_ls:
						if l != l2 and l2 != -1:
							t_val = 0
							o_locs = np.where(labels == l2)
							o_locs = o_locs[0]
							for loc_3 in o_locs:
								t_val += np.linalg.norm(points[loc_1] - points[loc_3])
							t_val = t_val/o_locs.size

							if t_val < b:
								b = t_val
					print(a, b)
					s_score += (b-a)/max(a, b)

	s_score = s_score/(labels.size - np.where(labels == -1)[0].size)
	return s_score

def transform_metric(dist, epochs):
	inverted_dist = np.ones(dist.shape) - dist - np.diag(np.ones((dist.shape[0]),))
	states = np.diag(np.ones((dist.shape[0]),))
	for i in range(0, epochs):
		c = np.einsum('ij,...j', states, inverted_dist).transpose()
		t = c/c.sum(axis=0,keepdims=1)
		states = t
		print('epoch {}/{}'.format(i+1, epochs))

	return states.transpose()


def precompute(t_list, path):
	t_dict = dict()

	for day in t_list:
		t_dict[day[0]] = (len(deflate.gzip_compress(bytes(day[1], encoding='utf8'))), len(bz2.compress(bytes(day[1], encoding='utf8'))), len(lzma.compress(bytes(day[1], encoding='utf8'))))
		print(day[0])

	with open('precomputed_' + str(path) + '.txt', 'wb') as f:
		pickle.dump(t_dict, f)

def evaluate_clustering(dates, labels, s_p_returns):

	returns_dict = dict()

	for i in range(0, len(labels)):
		date = dates[i]
		date_f = dates[i+1]
		label = labels[i]
		if label != -1:
			if label in returns_dict:
				returns_dict[label].append((s_p_returns[date_f] - s_p_returns[date])/s_p_returns[date])
			else:
				returns_dict[label] = [s_p_returns[date]]

	return returns_dict

def compute_all_clusters(path):
	with open('data.txt', 'rb') as f:
		text_list = pickle.load(f)

	with open('dates_'+ path +'.txt', 'rb') as f:
		dates = pickle.load(f)

	with open('p_vals_'+ path +'.txt', 'rb') as f:
		p_vals = pickle.load(f)

	with open('z_vals_'+ path +'.txt', 'rb') as f:
		z_vals = pickle.load(f)

	with open('m_vals_'+ path +'.txt', 'rb') as f:
		m_vals = pickle.load(f)

	text_list = iter(text_list)

	p_mat, z_mat, m_mat = build_distance_matrix(text_list, p_vals, z_vals, m_vals)


	split = int(2*p_mat.shape[0]/3)

	p_mat_test = p_mat[:split, :split]
	z_mat_test = z_mat[:split, :split]
	m_mat_test = m_mat[:split, :split]

	p_km_full = transform_metric(p_mat, 1000)
	z_km_full = transform_metric(z_mat, 1000)
	m_km_full = transform_metric(m_mat, 1000)

	p_km_test = p_km_full[:split]
	z_km_test = z_km_full[:split]
	m_km_test = m_km_full[:split]



	p_s = []
	z_s = []
	m_s = []


	top = 3
	bottom = 2
	for i in range(0, 100):
		scan = SKCL.DBSCAN(eps=((top-bottom) * i/99) + bottom, metric = 'precomputed')
		p_s.append(scan.fit(p_mat_test).labels_)
		z_s.append(scan.fit(z_mat_test).labels_)
		m_s.append(scan.fit(m_mat_test).labels_)
		print('DBSCAN {}/100'.format(i+1))


	with open('p_scan_'+ path +'.txt', 'wb') as f:
		pickle.dump(p_s, f)

	with open('z_scan_'+ path +'.txt', 'wb') as f:
		pickle.dump(z_s, f)

	with open('m_scan_'+ path +'.txt', 'wb') as f:
		pickle.dump(m_s, f)

	p_a = []
	z_a = []
	m_a = []

	for i in range(2, 21):
		agg = SKCL.AgglomerativeClustering(n_clusters= i, affinity = 'precomputed', linkage='average')
		p_a.append(agg.fit(p_mat_test).labels_)
		z_a.append(agg.fit(z_mat_test).labels_)
		m_a.append(agg.fit(m_mat_test).labels_)
		print('Agglomerative Clustering {}/19'.format(i-1))

	with open('p_agg_'+ path +'.txt', 'wb') as f:
		pickle.dump(p_a, f)

	with open('z_agg_'+ path +'.txt', 'wb') as f:
		pickle.dump(z_a, f)

	with open('m_agg_'+ path +'.txt', 'wb') as f:
		pickle.dump(m_a, f)

	p_km = []
	z_km = []
	m_km = []

	for i in range(2, 21):
		km = SKCL.KMeans(n_clusters= i)
		p_km.append(km.fit(p_km_test).labels_)
		z_km.append(km.fit(z_km_test).labels_)
		m_km.append(km.fit(m_km_test).labels_)
		print('K Means {}/19'.format(i-1))

	with open('p_km_'+ path +'.txt', 'wb') as f:
		pickle.dump(p_km, f)

	with open('z_km_'+ path +'.txt', 'wb') as f:
		pickle.dump(z_km, f)

	with open('m_km_'+ path +'.txt', 'wb') as f:
		pickle.dump(m_km, f)


def compute_all_silhouette(path):
	with open('p_scan_'+ path +'.txt', 'rb') as f:
		p_sc = pickle.load(f)

	with open('z_scan_'+ path +'.txt', 'rb') as f:
		z_sc = pickle.load(f)

	with open('m_scan_'+ path +'.txt', 'rb') as f:
		m_sc = pickle.load(f)

	with open('p_agg_'+ path +'.txt', 'rb') as f:
		p_agg = pickle.load(f)

	with open('z_agg_'+ path +'.txt', 'rb') as f:
		z_agg = pickle.load(f)

	with open('m_agg_'+ path +'.txt', 'rb') as f:
		m_agg = pickle.load(f)

	with open('p_km_'+ path +'.txt', 'rb') as f:
		p_km = pickle.load(f)

	with open('z_km_'+ path +'.txt', 'rb') as f:
		z_km = pickle.load(f)

	with open('m_km_'+ path +'.txt', 'rb') as f:
		m_km = pickle.load(f)

	with open('data.txt', 'rb') as f:
		text_list = pickle.load(f)
		text_list = iter(text_list)

	with open('dates_'+ path +'.txt', 'rb') as f:
		dates = pickle.load(f)

	with open('p_vals_'+ path +'.txt', 'rb') as f:
		p_vals = pickle.load(f)

	with open('z_vals_'+ path +'.txt', 'rb') as f:
		z_vals = pickle.load(f)

	with open('m_vals_'+ path +'.txt', 'rb') as f:
		m_vals = pickle.load(f)


	p_mat, z_mat, m_mat = build_distance_matrix(text_list, p_vals, z_vals, m_vals)

	split = int(2*p_mat.shape[0]/3)

	p_mat_test = p_mat[:split, :split]
	z_mat_test = z_mat[:split, :split]
	m_mat_test = m_mat[:split, :split]

	p_km_full = transform_metric(p_mat, 1000)
	z_km_full = transform_metric(z_mat, 1000)
	m_km_full = transform_metric(m_mat, 1000)

	p_km_test = p_km_full[:split]
	z_km_test = z_km_full[:split]
	m_km_test = m_km_full[:split]

	p_sc_score = []
	for sc in p_sc:
		p_sc_score.append(compute_silhouette_pre(sc, p_mat_test))

	z_sc_score = []
	for sc in z_sc:
		z_sc_score.append(compute_silhouette_pre(sc, z_mat_test))

	m_sc_score = []
	for sc in m_sc:
		m_sc_score.append(compute_silhouette_pre(sc, m_mat_test))

	p_agg_score = []
	for sc in p_agg:
		p_agg_score.append(compute_silhouette_pre(sc, p_mat_test))

	z_agg_score = []
	for sc in z_agg:
		z_agg_score.append(compute_silhouette_pre(sc, z_mat_test))

	m_agg_score = []
	for sc in m_agg:
		m_agg_score.append(compute_silhouette_pre(sc, m_mat_test))

	p_km_score = []
	for sc in p_km:
		p_km_score.append(compute_silhouette(sc, p_km_test))

	z_km_score = []
	for sc in z_km:
		z_km_score.append(compute_silhouette(sc, z_km_test))

	m_km_score = []
	for sc in m_km:
		m_km_score.append(compute_silhouette(sc, m_km_test))

	with open('p_sc_score_'+ path +'.txt', 'wb') as f:
		pickle.dump(p_sc_score, f)

	with open('z_sc_score_'+ path +'.txt', 'wb') as f:
		pickle.dump(z_sc_score, f)

	with open('m_sc_score_'+ path +'.txt', 'wb') as f:
		pickle.dump(z_sc_score, f)

	with open('p_agg_score_'+ path +'.txt', 'wb') as f:
		pickle.dump(p_agg_score, f)

	with open('z_agg_score_'+ path +'.txt', 'wb') as f:
		pickle.dump(z_agg_score, f)

	with open('m_agg_score_'+ path +'.txt', 'wb') as f:
		pickle.dump(m_agg_score, f)

	with open('p_km_score_'+ path +'.txt', 'wb') as f:
		pickle.dump(p_km_score, f)

	with open('z_km_score_'+ path +'.txt', 'wb') as f:
		pickle.dump(z_km_score, f)

	with open('m_km_score_'+ path +'.txt', 'wb') as f:
		pickle.dump(m_km_score, f)


def compute_dists():
	with open('precomputed_1.txt', 'rb') as f:
		precomputed = pickle.load(f)

	with open('precomputed_5.txt', 'rb') as f:
		precomputed_5 = pickle.load(f)

	with open('precomputed_20.txt', 'rb') as f:
		precomputed_20 = pickle.load(f)

	with open('data.txt', 'rb') as f:
		t_list_1 = pickle.load(f)

	t_list_5 = []
	for i in range(0, int(len(t_list_1)/5)):
		s = ''
		for j in range(0, 5):
			s += t_list_1[5*i +j][1]
		t_list_5.append((t_list_1[5*i +4][0], s))

	t_list_20 = []
	for i in range(0, int(len(t_list_1)/20)):
		s = ''
		for j in range(0, 20):
			s += t_list_1[20*i +j][1]
		t_list_20.append((t_list_1[20*i +19][0], s))

	count = 0
	count_5 = 0
	count_20 = 0

	z_vals = []
	p_vals = []
	m_vals = []
	z_vals_5 = []
	p_vals_5 = []
	m_vals_5 = []
	z_vals_20 = []
	p_vals_20 = []
	m_vals_20 = []
	dates = []
	dates_5 = []
	dates_20 = []
	texts = []
	texts_5 = []
	texts_20 = []

	i = 0
	j = 0
	for day in t_list_1:
		t_count = 0
		day_l = precomputed[day[0]]
		if day[0] not in dates:

			for t in texts:
				z_vals.append(score_zip(day[1], t[1], day_l[1], precomputed[t[0]][1]))
				p_vals.append(score_def(day[1], t[1], day_l[0], precomputed[t[0]][0]))
				m_vals.append(score_lzma(day[1], t[1], day_l[2], precomputed[t[0]][2]))
				t_count += 1
				if t_count % 5 == 0:
					print(count, t_count)

			z_vals.append(0)
			p_vals.append(0)
			m_vals.append(0)
			dates.append(day[0])

		texts.append(day)
		count += 1

		if count % 5 == 0:
			with open('dates_1.txt', 'wb') as f:
				pickle.dump(dates, f)

			with open('z_vals_1.txt', 'wb') as f:
				pickle.dump(z_vals, f)

			with open('p_vals_1.txt', 'wb') as f:
				pickle.dump(p_vals, f)

			with open('m_vals_1.txt', 'wb') as f:
				pickle.dump(m_vals, f)

		if count % 5 == 0:
			day = t_list_5[count_5]
			t_count_5 = 0
			day_l = precomputed_5[day[0]]
			if day[0] not in dates_5:

				for t in texts_5:
					z_vals_5.append(score_zip(day[1], t[1], day_l[1], precomputed_5[t[0]][1]))
					p_vals_5.append(score_def(day[1], t[1], day_l[0], precomputed_5[t[0]][0]))
					m_vals_5.append(score_lzma(day[1], t[1], day_l[2], precomputed_5[t[0]][2]))
					t_count_5 += 1

					print(5, count_5, t_count_5)

				z_vals_5.append(0)
				p_vals_5.append(0)
				m_vals_5.append(0)
				dates_5.append(day[0])

			texts_5.append(day)
			count_5 += 1

			with open('dates_5.txt', 'wb') as f:
				pickle.dump(dates_5, f)

			with open('z_vals_5.txt', 'wb') as f:
				pickle.dump(z_vals_5, f)

			with open('p_vals_5.txt', 'wb') as f:
				pickle.dump(p_vals_5, f)

			with open('m_vals_5.txt', 'wb') as f:
				pickle.dump(m_vals_5, f)

		if count % 20 == 0:
			day = t_list_20[count_20]
			t_count_20 = 0
			day_l = precomputed_20[day[0]]
			if day[0] not in dates_20:

				for t in texts_20:
					z_vals_20.append(score_zip(day[1], t[1], day_l[1], precomputed_20[t[0]][1]))
					p_vals_20.append(score_def(day[1], t[1], day_l[0], precomputed_20[t[0]][0]))
					m_vals_20.append(score_lzma(day[1], t[1], day_l[2], precomputed_20[t[0]][2]))
					t_count_20 += 1

					print(20, count_20, t_count_20)

				z_vals_20.append(0)
				p_vals_20.append(0)
				m_vals_20.append(0)
				dates_20.append(day[0])

			texts_20.append(day)
			count_20 += 1

			with open('dates_20.txt', 'wb') as f:
				pickle.dump(dates_20, f)

			with open('z_vals_20.txt', 'wb') as f:
				pickle.dump(z_vals_20, f)

			with open('p_vals_20.txt', 'wb') as f:
				pickle.dump(p_vals_20, f)

			with open('m_vals_20.txt', 'wb') as f:
				pickle.dump(m_vals_20, f)


if __name__ == '__main__':
	paths = ['1', '5', '20']
	for p in paths:
		compute_all_clusters(p)
		compute_all_silhouette(p)

