import bz2
import lz4.frame
import cv2
import os
import numpy as np
import pickle

def load_images_from_folder(folder_list,Label_list):
    Ind=0
    data={}
    for i in range(len(folder_list)):
        count=0
        for filename in os.listdir(folder_list[i]):
            if(count>1000):
                break
            data[Ind]={}
            img = cv2.imread(os.path.join(folder_list[i],filename))
            if img is not None:
                data[Ind]['data']=img
                data[Ind]['label']=Label_list[i]
            Ind+=1
            count+=1

    with open("X_ray_data_labeled",'wb') as f:
        pickle.dump(data,f)
        f.close()

def get_c_zip(data):

    c_zip_data=[]
    for i in range(len(data)):
        c_lz_data.append(len(bz2.compress(data[i]['data'])))
        
    with open('c_zip_data','wb') as f:
        pickle.dump(c_zip_data,f)
        f.close()

def get_c_lz(data):
    c_lz_data=[]
    for i in range(len(data)):
        c_lz_data.append(len(lz4.frame.compress(data[i]['data'])))
        
    with open('c_lz_data','wb') as f:
        pickle.dump(c_lz_data,f)
        f.close()

def score_zip(x, y,c_x,c_y):
    x = bytes(x)
    y = bytes(y)
    cat_xy = x+y
    c_xy = len(bz2.compress(cat_xy))
    cat_yx = y+x
    c_yx = len(bz2.compress(cat_yx))
    

    return (min(c_xy, c_yx) - min(c_x, c_y))/max(c_x, c_y)

def score_lz(x, y,c_x,c_y):
    x=bytes(x)
    y=bytes(y)
    cat_xy = x + y
    c_xy = len(lz4.frame.compress(cat_xy))
    cat_yx = y + x
    c_yx = len(lz4.frame.compress(cat_yx))

    return (min(c_xy, c_yx) - min(c_x, c_y))/max(c_x, c_y)

def get_zip_NCD(data_dic,c_zip_data,zip_NCD,Ical):

        
    for i in range(len(data_dic)-1):
        if i not in Ical:
            print(i)
            for j in range(i+1,len(data_dic)):
                zip_NCD[i][j]=score_zip(data_dic[i]['data'], data_dic[j]['data'],c_zip_data[i],c_zip_data[j])
                zip_NCD[j][i]=zip_NCD[i][j]
            
            if i % 5==0:
                with open('zip_NCD','wb') as f:
                    pickle.dump(zip_NCD,f)
                    f.close()
                with open('Icalculated_zip','wb') as f:
                    pickle.dump(Ical,f)
                    f.close()

            Ical.append(i)

def get_lz_NCD(c_lz_data,data_dic,lz_NCD,Ical):

    for i in range(len(data_dic)-1):
        if i not in Ical:
            for j in range(i+1,len(data_dic)):
                lz_NCD[i][j]=score_lz(data_dic[i]['data'], data_dic[j]['data'],c_lz_data[i],c_lz_data[j])
                lz_NCD[j][i]=lz_NCD[i][j]

            if i % 5==0:
                with open('lz_NCD','wb') as f:
                    pickle.dump(lz_NCD,f)
                    f.close()
                with open('Icalculated_lz','wb') as f:
                    pickle.dump(Ical,f)
                    f.close()

            Ical.append(i)

if __name__ == '__main__':

	#image data 
	f=open("X_ray_data_labeled",'rb')
	X_ray_data = pickle.load(f)
	f.close()

	f=open("zip_NCD",'rb')
	zip_NCD = pickle.load(f)
	f.close()

    f=open("lz_NCD",'rb')
    lz_NCD = pickle.load(f)
    f.close()
	#zip compressed data
	f=open("c_zip_data",'rb')
	c_zip_data = pickle.load(f)
	f.close()

    f=open("c_lz_data",'rb')
    c_lz_data = pickle.load(f)
    f.close()

	Ical=[]

	get_zip_NCD(X_ray_data,c_zip_data,zip_NCD,Ical)

    get_lz_NCD(X_ray_data,c_lz_data,lz_NCD,Ical)






