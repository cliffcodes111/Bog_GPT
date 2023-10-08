#!/usr/bin/env python
# coding: utf-8

# In[3]:


#Hackathon Code Interactive

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import geemap
import ee
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn import linear_model
import scipy
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

ee.Initialize()


# In[5]:


#important functions
def unique(list1):
 
    # initialize a null list
    unique_list = []
     
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
            
    return unique_list

#Functions to clean data and sort it by datetime

def to_datetime(dt):
    datetimevec= []
    for i in range (0,len(dt)):
        if isinstance(dt[i], str) == True:
            datetime_object = datetime.strptime(dt[i],'%Y-%m-%d %H:%M:%S')
            #print(datetime_object)
            str_time = datetime_object.strftime('%d/%m/%Y' )
            datetimevec.append(str_time)
    return datetimevec

def to_date(dt):
    datetimevec= []
    for i in range (0,len(dt)):
        if isinstance(dt[i], str) == True:
            datetime_object = datetime.strptime(dt[i],'%Y-%m-%d')
            str_time = datetime_object.strftime('%d/%m/%Y' )
            datetimevec.append(str_time)
    return datetimevec

def to_date_done(dt):
    datetimevec= []
    for i in range (0,len(dt)):
        if isinstance(dt[i], str) == True:
            datetime_object = datetime.strptime(dt[i],'%d/%m/%Y')
            datetimevec.append(datetime_object)
    return datetimevec

def pad(data):
    bad_indexes = np.isnan(data)
    good_indexes = np.logical_not(bad_indexes)
    good_data = data[good_indexes]
    interpolated = np.interp(bad_indexes.nonzero()[0], good_indexes.nonzero()[0], good_data)
    data[bad_indexes] = interpolated
    return data

def choppeddf(df, date_fix, date_int):
    chopped_data = []
    for i in date_int:
        index = date_fix.index(i)
        chopped_data.append(df['Data'][index])
    return chopped_data

def choppeddf2(df, date_fix, date_int):
    chopped_data = []
    for i in date_int:
        index = date_fix.index(i)
        chopped_data.append(df['Depth'][index])
    return chopped_data

def choppeddf3(vector, date_fix, date_int):
    chopped_data = []
    for i in date_int:
        index = date_fix.index(i)
        chopped_data.append(vector[index])
    return chopped_data


# In[6]:


#data classes
class moddata:
    def __init__(self, name, data, date):
        self.name = name
        self.data = data
        self.date = date
class satdata:
    def __init__(self, name, data, date):
        self.name = name
        self.data = data
        self.date = date


# In[7]:


# Date vector
def degrade(time_steps, vector):
    newvector = []
    for i in range(0, len(vector)-1):
        if i % time_steps == 0:
            newvector.append(vector[i])
    return newvector

#creating a range of dates (under an associated vector) for the training data
start_date = input("Please enter the start date in the format %Y-%m-%d: ")
end_date = input("Please enter the end date in the format %Y-%m-%d: ")

date_vec_init = list(pd.date_range(start=start_date,end=end_date))
date_vec_sen = degrade(6,date_vec_init)
date_vec_mod = degrade(1, date_vec_init)

for i in range(0,len(date_vec_sen)):
    date_vec_sen[i] = str(date_vec_sen[i].strftime("%Y-%m-%d"))
for i in range(0,len(date_vec_mod)):
    date_vec_mod[i] = str(date_vec_mod[i].strftime("%Y-%m-%d"))


# In[8]:


#User input for longitude and latitude
lon = input("Please enter the longitudinal coordinate of your borehole of interest: ")
lat = input("Please enter the latitude coordinate of your borehole of interest: ")


# In[32]:


#Creating geometries for GEEMAPS to read in specific bog areas
#rectangle half size (in coordinates)

half = 0.05
point = ee.Geometry.Point([lon, lat])
rec = ee.Geometry.Rectangle([lon + half, lat + half, lon - half, lat - half])


# In[63]:


#Retrieving Sentinel 1 data
#Note that sentinel 1 data is spaced apart by 6 days
AscImg = []
DescImg = []
points_list_sen = []
empty_dates = list()

for j in range(0, len(date_vec_sen)-1):
    imgVV = ee.ImageCollection('COPERNICUS/S1_GRD').filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))    .filter(ee.Filter.eq('instrumentMode', 'IW')).filterBounds(point).select('VV')    .filterDate(date_vec_sen[j], date_vec_sen[j+1])
    desc = imgVV.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING')) 
    descfirst = desc.first()
    data = geemap.ee_to_numpy(ee_object = descfirst, region = point)
    if isinstance(data, np.ndarray) == True:
        points_list_sen.append(data)
    else:
        points_list_sen.append('None %d' %(j))
        empty_dates.append(date_vec_sen[j+1])
    print("retrieved " + str(j))
    
    new_points_listsen = []
    for i in range(0, len(points_list_sen)):
        if isinstance(points_list_sen[i], np.ndarray) == True:
            new_points_listsen.append(float(points_list_sen[i].reshape(-1)))
        elif isinstance(points_list_sen[i], str) == True:
            new_points_listsen.append(None)
    
        


# In[35]:


#Retrieving MODIS data
    
AscImg = []
DescImg = []
#Map.addLayer(polygon1)
points_list_mod = []
empty_dates = list()

for j in range(0, len(date_vec_mod)-1):
    img = ee.ImageCollection("MODIS/061/MOD11A1").select('LST_Day_1km').filterDate(date_vec_mod[j], date_vec_mod[j+1]).filterBounds(rec)
    #desc = imgVV.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING')) 
    descfirst = img.first()
    data = geemap.ee_to_numpy(ee_object = descfirst, region = rec)
    if isinstance(data, np.ndarray) == True:
        points_list_mod.append(data)
    else:
        points_list_mod.append('None %d' %(j))
        empty_dates.append(date_vec_mod[j+1])
    print("retrieved" + str(j))
    


# In[64]:


new_points_listmod = []
for i in range(0, len(points_list_mod)):
    if isinstance(points_list_mod[i], np.ndarray):
        av = np.average(points_list_mod[i])
        new_points_listmod.append(av)
    elif isinstance(points_list_mod[i], str):
        new_points_listmod.append(new_points_listmod[i-1])


# In[69]:


sendata = new_points_listsen 
moddata = new_points_listmod
datesen = date_vec_sen[1:]
datemod = date_vec_mod[1:]


# In[74]:


#Intersection of two datasets 
intersection_set = set.intersection(set(datesen), set(datemod))
date_int = list(intersection_set)
date_int.sort(key=lambda date: datetime.strptime(date, "%Y-%m-%d"))
data_mod = choppeddf3(moddata, datemod, date_int)
data_sen = choppeddf3(sendata, datesen, date_int)


# In[81]:


get_ipython().run_line_magic('store', '-r linregmodel')
get_ipython().run_line_magic('store', '-r neural_model')


# In[82]:


model_requested = input("Choose either linreg or neural: ")
if model_requested == linreg:
    model = linregmodel
    test = model.pred([sendata, moddata])
    plt.plot(datesen, test, label = "Predicted Linear Regression Model")
if model_requested == neural:
    model = neural_model
    model = model.pred([data])
    plt.plot(datesen, model, label = "Predicted Neural Network Model")


# In[ ]:




