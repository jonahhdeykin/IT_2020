import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

def format_data(data):
    zip_X=[i for i in range(len(data))]
    zip_X=np.array(zip_X).reshape(-1, 1)
    zip_Y=[X_ray_data[i]['label'] for i in range(len(data))]
    zip_Y=np.array(zip_Y)
    
    return zip_X, zip_Y

def func(a,b):
    x=int(a[0])
    y=int(b[0])
    global zip_NCD
    
    return zip_NCD[x][y]


if __name__ == '__main__':

	f=open("X_ray_data_labeled",'rb')
	X_ray_data = pickle.load(f)
	f.close()

	zip_X, zip_Y=format_data(X_ray_data)
	zip_X_train, zip_X_test, zip_y_train, zip_y_test = train_test_split(zip_X, zip_Y, test_size=0.70, random_state=30)

	f=open("zip_NCD_total",'rb')
	zip_NCD = pickle.load(f)
	f.close()

	zip_model = KNeighborsClassifier(n_neighbors=1,algorithm='ball_tree',metric=func)
	zip_model.fit(zip_X_train,zip_y_train)

	predictions = zip_model.predict(zip_X_test.reshape(-1, 1))
	acc = metrics.accuracy_score(zip_y_test, predictions)

