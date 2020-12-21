import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

def format_data(data):
    lz_X=[i for i in range(len(data))]
    lz_X=np.array(lz_X).reshape(-1, 1)
    lz_Y=[X_ray_data[i]['label'] for i in range(len(data))]
    lz_Y=np.array(lz_Y)
    
    return lz_X, lz_Y

def func(a,b):
    x=int(a[0])
    y=int(b[0])
    global lz_NCD
    
    return lz_NCD[x][y]


if __name__ == '__main__':

	f=open("X_ray_data_labeled",'rb')
	X_ray_data = pickle.load(f)
	f.close()

	lz_X, lz_Y=format_data(X_ray_data)
	lz_X_train, lz_X_test, lz_y_train, lz_y_test = train_test_split(lz_X, lz_Y, test_size=0.70, random_state=30)

	f=open("lz_NCD",'rb')
	lz_NCD = pickle.load(f)
	f.close()

	lz_model = KNeighborsClassifier(n_neighbors=1,algorithm='ball_tree',metric=func)
	lz_model.fit(lz_X_train,lz_y_train)

	predictions = lz_model.predict(lz_X_test.reshape(-1, 1))
	acc = metrics.accuracy_score(lz_y_test, predictions)