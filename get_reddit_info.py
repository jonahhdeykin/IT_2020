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
import statistics
from copy import deepcopy
import scipy

#get reddit data for a given sub and a given day
def getPushshiftData(search_type, after, before, sub):
	url = 'https://api.pushshift.io/reddit/search/'+ str(search_type) +'/?size=500&after='+str(after)+'&before='+str(before)+'&subreddit='+str(sub)
	r = requests.get(url)
	data = json.loads(r.text)
	return data['data']

#Pull the correct time bounds for a trading day
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

#Compute bzip2 NCD
def score_zip(x, y, c_x, c_y):
	x = bytes(x, encoding='utf8')
	y = bytes(y, encoding='utf8')

	cat_xy = x + y
	c_xy = len(bz2.compress(cat_xy))
	cat_yx = y + x
	c_yx = len(bz2.compress(cat_yx))

	return (min(c_xy, c_yx) - min(c_x, c_y))/max(c_x, c_y)

#Compute deflate NCD
def score_def(x, y, c_x, c_y):
	x = bytes(x, encoding='utf8')
	y = bytes(y, encoding='utf8')

	cat_xy = x + y
	c_xy = len(deflate.gzip_compress(cat_xy))
	cat_yx = y + x
	c_yx = len(deflate.gzip_compress(cat_yx))

	return (min(c_xy, c_yx) - min(c_x, c_y))/max(c_x, c_y)

#Compute LZMA NCD
def score_lzma(x, y, c_x, c_y):
	x = bytes(x, encoding='utf8')
	y = bytes(y, encoding='utf8')

	cat_xy = x + y
	c_xy = len(lzma.compress(cat_xy))
	cat_yx = y + x
	c_yx = len(lzma.compress(cat_yx))

	return (min(c_xy, c_yx) - min(c_x, c_y))/max(c_x, c_y)

#Scrape the reddit data
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

#Precompute the distance matricies
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

#Construct a distace matrix given list of precomputed values
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

#Embed the points in euclidean space using the distance matrix
def transform_metric(dist, epochs):
	inverted_dist = np.ones(dist.shape) - dist - np.diag(np.ones((dist.shape[0]),))
	states = np.diag(np.ones((dist.shape[0]),))
	for i in range(0, epochs):
		c = np.einsum('ij,...j', states, inverted_dist).transpose()
		t = c/c.sum(axis=0,keepdims=1)
		states = t
		print('epoch {}/{}'.format(i+1, epochs))

	return states.transpose()

#run a single split with n as the split ratio
def split_n(train, test, distance, n, returns):
	lower = max(1, int(len(test)*n))
	upper = min(int(len(test)*(1-n)), len(test) - 1)

	test_returns = []
	for index in train:
		test_returns.append((index, returns[index]))
	test_returns.sort(key= lambda x: x[1])
	lower_comp = [i[0] for i in test_returns[:lower]]
	upper_comp = [i[0] for i in test_returns[upper:]]

	neg_test = []
	pos_test = []
	lower_index = []
	upper_index = []
	for index in test:
		l_c = np.mean(distance[index][lower_comp])
		u_c = np.mean(distance[index][upper_comp])

		if l_c < u_c:
			neg_test.append(returns[index])
			lower_index.append(index)
		else:
			pos_test.append(returns[index])
			upper_index.append(index)

	l_m = sum(neg_test)/len(neg_test)
	l_sd = statistics.pstdev(neg_test)

	u_m = sum(pos_test)/len(pos_test)
	u_sd = statistics.pstdev(pos_test)

	return(l_m, l_sd, len(neg_test), u_m, u_sd, len(pos_test), lower_index, upper_index)

#Run all different splits for step_size trials
def run_splits(step_size, path_list, seed=None):
	full_results = dict()
	vals = []
	p_mats = []
	z_mats = []
	m_mats = []
	tests = []
	rets = []
	trains = []
	pos_dict = dict()
	if seed is not None:
		np.random.seed(seed)
	for path in path_list:
		with open('data.txt', 'rb') as f:
			text_list = pickle.load(f)

		with open('dates_'+ path +'.txt', 'rb') as f:
			dates = pickle.load(f)
			dates.reverse()

		with open('p_vals_'+ path +'.txt', 'rb') as f:
			p_vals = pickle.load(f)

		with open('z_vals_'+ path +'.txt', 'rb') as f:
			z_vals = pickle.load(f)

		with open('m_vals_'+ path +'.txt', 'rb') as f:
			m_vals = pickle.load(f)

		with open('s_p.txt', 'rb') as f:
			s_p = pickle.load(f)

		returns = dict()
		for i in range(0, len(dates)-1):
			returns[len(dates)-1-i] = (s_p[dates[len(dates)-2-i]] - s_p[dates[len(dates)-1-i]])/s_p[dates[len(dates)-1-i]]

		rets.append(returns)

		text_list = iter(text_list)

		p_mat, z_mat, m_mat = build_distance_matrix(text_list, p_vals, z_vals, m_vals)
		p_mats.append(p_mat)
		z_mats.append(z_mat)
		m_mats.append(m_mat)

		l = []
		u = []

		order = np.random.permutation([i for i in range(1, len(dates))])

		train = order[:int(len(order)/2)]
		trains.append(train)
		test = order[int(len(order)/2):int(len(order)/4)*3]
		tests.append(test)
		vals.append(order[int(len(order)/4)*3:])

		p = []
		z = []
		m = []
		for i in range(1, step_size+1):
			try:
				p.append(split_n(train, test, p_mat, (i/step_size) * 0.5, returns))
			except:
				p.append((1, 10, 10, 0, 10, 10, [], []))
			try:
				z.append(split_n(train, test, z_mat, (i/step_size) * 0.5, returns))
			except:
				z.append((1, 10, 10, 0, 10, 10, [], []))
			try:
				m.append(split_n(train, test, m_mat, (i/step_size) * 0.5, returns))
			except:
				m.append((1, 10, 10, 0, 10, 10, [], []))
		full_results[path] = (p, z, m)

		row = []
		try:
			row.append(pos_split(train, test, p_mat, returns))
		except:
			row.append(('NA',))
		try:
			row.append(pos_split(train, test, z_mat, returns))
		except:
			row.append(('NA',))
		try:
			row.append(pos_split(train, test, m_mat, returns))
		except:
			row.append(('NA',))

		pos_dict[path] = row


	with open('full_results.txt', 'wb') as f:
		pickle.dump(full_results, f)



	best_configs = []
	for key in full_results:
		res = full_results[key]
		for c in res:
			best = 1
			index = 0
			temp = 1
			for i in range(0, len(c)):
				if c[i][5] > 1 and c[i][2] > 1:
					temp = one_tailed_t_test(c[i][3], c[i][4], c[i][5], c[i][0], c[i][1], c[i][2])
				if temp < best:
					index = i
					best = temp
			best_configs.append((best, index))

	index = 0
	val_configs = []
	for key in full_results:
		res = full_results[key]
		try:
			val_configs.append((key+'_p', split_n(trains[index], vals[index], p_mats[index], ((best_configs[index*3][1]+1)/step_size) * 0.5, rets[index])))
		except:
			val_configs.append(('NA', 0))
		try:
			val_configs.append((key+'_z', split_n(trains[index], vals[index], z_mats[index], ((best_configs[index*3+1][1]+1)/step_size) * 0.5, rets[index])))
		except:
			val_configs.append(('NA', 0))
		try:
			val_configs.append((key+'_m',split_n(trains[index], vals[index], m_mats[index], ((best_configs[index*3+2][1]+1)/step_size) * 0.5, rets[index])))
		except:
			val_configs.append(('NA', 0))
		index += 1

	return val_configs, full_results, pos_dict

#Run the one taikled Welch's T-test
def one_tailed_t_test(m1, s1, l1, m2, s2, l2):
	t = (m1 - m2)/math.sqrt((s1/l1)+(s2/l2))
	df = (((s1/l1)+(s2/l2))**2)/(((s1/l1)**2)/(l1-1)+((s2/l2)**2)/(l2-1))
	return(scipy.stats.t.sf(t, df=df))

#Run the simple positive negqtive split test
def pos_split(train, test, distance, returns):
	pos = []
	neg = []
	for index in train:
		if returns[index] > 0:
			pos.append(index)
		else:
			neg.append(index)

	neg_test = []
	pos_test = []
	lower_index = []
	upper_index = []
	for index in test:
		l_c = np.mean(distance[index][neg])
		u_c = np.mean(distance[index][pos])

		if l_c < u_c:
			neg_test.append(returns[index])
			lower_index.append(index)
		else:
			pos_test.append(returns[index])
			upper_index.append(index)

	l_m = sum(neg_test)/len(neg_test)
	l_sd = statistics.pstdev(neg_test)

	u_m = sum(pos_test)/len(pos_test)
	u_sd = statistics.pstdev(pos_test)

	return l_m, u_m

if __name__ == '__main__':

	#Compute all results

	average_performance = dict()
	for i in ['1', '5', '20']:
		for j in ['_p', '_z', '_m']:
			average_performance[i+j] = []
	t_f = []
	t_r = []
	for i in range(0, 500):
		val, full, rand = run_splits(100, ['1', '5', '20'], seed = i)
		t_f.append(full)
		t_r.append(rand)
		for v in val:
			if v[0] != 'NA':
				average_performance[v[0]].append((v[1][0], v[1][3]))

		print(i)

	for key in average_performance:
		l = []
		u = []
		for i in average_performance[key]:

			l.append(i[0])
			u.append(i[1])
		s = one_tailed_t_test(np.mean(u), math.sqrt(np.var(u)), len(u), np.mean(l), math.sqrt(np.var(l)), len(l))
		print(key, np.mean(l), np.mean(u), s)


	average_performance_2 = dict()
	for i in ['1', '5', '20']:
		for j in ['_p', '_z', '_m']:
			average_performance_2[i+j] = []

	with open('s_p.txt', 'rb') as f:
			s_p = pickle.load(f)

	for f in t_f:
		for key in f:
			with open('dates_'+ key +'.txt', 'rb') as fl:
				dates = pickle.load(fl)
				dates.reverse()

			returns = dict()
			for i in range(0, len(dates)-1):
				returns[len(dates)-1-i] = (s_p[dates[len(dates)-2-i]] - s_p[dates[len(dates)-1-i]])/s_p[dates[len(dates)-1-i]]

			for c in [(0, '_p'), (1, '_z'), (2, '_m')]:
				results = f[key][c[0]]
				classify = [0 for _ in range(0, len(dates))]
				for r in results:
					for i in r[6]:
						classify[i] -= 1
					for i in r[7]:
						classify[i] += 1

				lower = []
				upper = []

				for i in range(1, len(dates)):
					if classify[i] < 0:
						lower.append(returns[i])
					if classify[i] > 0:
						upper.append(returns[i])
				if len(lower) > 0 and len(upper) > 0:

					average_performance_2[key+c[1]].append((np.mean(lower), np.mean(upper)))


	for key in average_performance_2:
		l = []
		u = []
		for i in average_performance_2[key]:

			l.append(i[0])
			u.append(i[1])
		s = one_tailed_t_test(np.mean(u), math.sqrt(np.var(u)), len(u), np.mean(l), math.sqrt(np.var(l)), len(l))
		print(key, np.mean(l), np.mean(u), s)

	average_performance_3 = dict()
	for i in ['1', '5', '20']:
		for j in ['_p', '_z', '_m']:
			average_performance_3[i+j] = []


	for r in t_r:
		for key in r:
			for c in [(0, '_p'), (1, '_z'), (2, '_m')]:
				average_performance_3[key+c[1]].append(r[key][c[0]])

	for key in average_performance_3:
		l = []
		u = []
		for i in average_performance_3[key]:
			if i[0] != 'NA':
				l.append(i[0])
				u.append(i[1])
		s = one_tailed_t_test(np.mean(u), math.sqrt(np.var(u)), len(u), np.mean(l), math.sqrt(np.var(l)), len(l))
		print(key, np.mean(l), np.mean(u), s)

