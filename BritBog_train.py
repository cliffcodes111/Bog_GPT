#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#The goal of this code snippet is to create a neural network model
#that predicts the water table depth of a British peat bog with satellite data.

#The model is trained from existing data from a the Boltonfellend Bog in Carlisle.
#This bog was chosen due to it having a proportion of area undergoing restoration
#and a portion of it still experiencing damage from anthropogenic activities. 

#Using GRD data from Sentinel 1 and LST-day data from MODIS (previously shown
#to accurately predict water table levels), a neural network is developed.
#The initial neural network will train on data from water table depth measurements
#from 13 boreholes in the bog alongside sentinel and MODIS data for the exact
#coordinates of the peat bog for 3 years. Later additions will train the model 
#on more borehole point within the bog and within other surrounding bogs to get
#wide variety of bogs within the training dataset. Further satellite data from
#MODIS, Sentinel1, Sentinel2, and LANDSAT can be considered. 


# In[12]:


#Importing Relevant Modules

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


#Developing functions that will be used to clean the data throughout the code

#Ensuring that inputs into the Sentinel data gathering system are every 6 days to accomodate for satellite
def degrade(time_steps, vector):
    newvector = []
    for i in range(0, len(vector)-1):
        if i % time_steps == 0:
            newvector.append(vector[i])
    return newvector

#creating a range of dates (under an associated vector) for the training data
start_date = "2017-01-01"
end_date = "2019-06-27"
date_vec_init = list(pd.date_range(start=start_date,end=end_date))
date_vec_sen = degrade(6,date_vec_init)
date_vec_mod = degrade(1, date_vec_init)

for i in range(0,len(date_vec_sen)):
    date_vec_sen[i] = str(date_vec_sen[i].strftime("%Y-%m-%d"))
for i in range(0,len(date_vec_mod)):
    date_vec_mod[i] = str(date_vec_mod[i].strftime("%Y-%m-%d"))


# In[6]:


def unique(list1):
 
    # initialize a null list
    unique_list = []
     
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
            
    return unique_list


# In[8]:


#Sentinel Points
dbh8 = [ -2.780614, 55.01578]
abh1 = [-2.810805556, 55.014325]
abh2 = [-2.809936111, 55.01245556]
abh3 = [-2.813558333, 55.01241944]
rbh1 = [-2.80215, 55.01413333]
rbh3 = [-2.805869444, 55.01449167]
rbh4 = [-2.807677778, 55.01149444]
ombh4 = [-2.801505556, 55.01446389]
nbh1 = [-2.796633333, 55.01230556]
nbh2 = [-2.800097222, 55.01194167]
sobh2 = [-2.798338889, 55.00924722]
sobh3 = [ -2.797319444, 55.00581667]
sbh1 = [-2.789194444, 55.018475]
dbh5 = [-2.785811111, 55.01290833]

bolton_dict = { "dbh8":dbh8,                "abh1":abh1, "abh3": abh3, "rbh1":rbh1, "rbh3":rbh3,              "rbh4":rbh4, "ombh4":ombh4, "nbh1":nbh1, "nbh2": nbh2, "sobh2":sobh2, "sobh3":sobh3,               "sbh1":sbh1, "dbh5":dbh5}
boltonnamelist = ["dbh8", "abh1",  "abh3", "rbh1", "rbh3", "rbh4", "ombh4", "nbh1", "nbh2", "sobh2", "sobh3", "sbh1", "dbh5"]

#At some point need to make this a user input (widget?) Also need a user input of time
#pointpoint = bolton_dict[str(choice_name)]
#point1 = ee.Geometry.Point(pointpoint1) 
#point2 = ee.Geometry.Point(pointpoint2) 
points = [ee.Geometry.Point(bolton_dict[str(a)]) for a in boltonnamelist]
boltonnamelist = boltonnamelist


# In[11]:


#Retrieving Sentinel 1 data
#Note that sentinel 1 data comes every 6 days
sen_dataframe = pd.DataFrame()
sen_dataframe.insert(0, "Date", date_vec_sen[0:-1], True)
#sen_dataframe = sen_dataframe.set_index("Date")
for point in points:
    
    #determining name
    indexpoint = points.index(point)
    name = boltonnamelist[indexpoint]
    
    AscImg = []
    DescImg = []
    #Map.addLayer(polygon1)
    points_list_sen = []
    empty_dates = list()

    for j in range(0, len(date_vec_sen)-1):
        imgVV = ee.ImageCollection('COPERNICUS/S1_GRD').filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))        .filter(ee.Filter.eq('instrumentMode', 'IW')).filterBounds(point).select('VV')        .filterDate(date_vec_sen[j], date_vec_sen[j+1])
        desc = imgVV.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING')) 
        descfirst = desc.first()
        data = geemap.ee_to_numpy(ee_object = descfirst, region = point)
        if isinstance(data, np.ndarray) == True:
            points_list_sen.append(data)
        else:
            points_list_sen.append('None %d' %(j))
            empty_dates.append(date_vec_sen[j+1])
        print("retrieved" + str(j))
    
    new_points_listsen = []
    for i in range(0, len(points_list_sen)):
        if isinstance(points_list_sen[i], np.ndarray) == True:
            new_points_listsen.append(float(points_list_sen[i].reshape(-1)))
        elif isinstance(points_list_sen[i], str) == True:
            new_points_listsen.append(None)
    print("done")
            
    sen_dataframe.insert(0, name, new_points_listsen)
display(sen_dataframe)
        


# In[137]:


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


# In[138]:


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


# In[139]:


#importing hydro data
class hydrodata:
    def __init__(self, name, data, date):
        self.name = name
        self.data = data
        self.date = date
class satdata:
    def __init__(self, name, data, date):
        self.name = name
        self.data = data
        self.date = date
class linreg:
    def __init__(self, bogname, model):
        self.name = name
        self.model = model
        
class neural:
    def __init__(self,bogname, model):
        self.name = name
        self.model = model

hydro_object_list = []
sat_object_list = []
for name in boltonnamelist:
    #importing in hydro_data
    boreholenamecsv = "/Users/sarahcliff/Desktop/durham management/Environmental physics Year 4/Masters project/project/Sentinel/New Bolton fell moss better data/raw data/" + str(name) + "raw@2022-02-27.csv"
    df_hydro = pd.read_csv(boreholenamecsv)
    df_hydrodata = pad(np.array(df_hydro['Data']))
    date_hy = to_date(df_hydro['Date'])
    obj = hydrodata(name, df_hydrodata, date_hy)
    hydro_object_list.append(obj)
    
    #importing in sat data
    df_satdata = sen_dataframe[str(name)].tolist()
    datesat = to_date(sen_dataframe["Date"].tolist())
    objsat = satdata(name, df_satdata, datesat)
    sat_object_list.append(objsat)
    

#Land Surface temperature data



#Determining the intersection of all three data-sets to fit within machine learning model
for i in range(0, len(sat_object_list)):
    df_lst = pd.read_csv("/Users/sarahcliff/Desktop/durham management/Environmental physics Year 4/Masters project/project/MODIS data/Raw MODIS/lstraw@2022-02-17.csv")
    date_lst = to_date(df_lst['Date'])
    data_lst = pad(np.array(df_lst['Data']))
    intersection_set = set.intersection(set(hydro_object_list[i].date), set(sat_object_list[i].date), set(date_lst))
    date_int = list(intersection_set)
    #sorting dates to fit within machine learning model.
    date_int.sort(key=lambda date: datetime.strptime(date, "%d/%m/%Y"))
    data_lst = choppeddf3(data_lst, date_lst, date_int)
    datahyint = choppeddf3(hydro_object_list[i].data, hydro_object_list[i].date, date_int)
    hydro_object_list[i].data = datahyint
    datasatint = choppeddf3(sat_object_list[i].data, sat_object_list[i].date, date_int)
    sat_object_list[i].data = datasatint


# In[176]:


#Multi-Regression model 
train_listlist = pd.DataFrame()
linregmodellist = []

itt= 1

for index in range(0, len(hydro_object_list)-1):
    if len(sat_object_list[index].data) != len(date_int):
        diff = np.abs(len(sat_object_list[index].data) - len(date_int))
        datedate = date_int[diff:]
        datalstdatalst = data_lst[diff:]
    else:
        datedate = date_int
        datalstdatalst = data_lst
    data = {'Date': datedate, 'lst': datalstdatalst, 'Sat': sat_object_list[index].data, 'Bog' : hydro_object_list[index].data}  
    df = pd.DataFrame(data) 
    X = df[['lst', 'Sat']]
    y = df['Bog']

    model_linreg = linear_model.LinearRegression()
    model_linreg.fit(X, y)
    score = model_linreg.score(X, y, sample_weight=None)
    coefvalx, coefvaly = model_linreg.coef_
    #print([score, coefvalx, coefvaly])
    print(model_linreg.coef_, score)
    
    linregmodel = linreg(sat_object_list[index].name, model_linreg)
    linregmodellist.append(linregmodel)


# In[144]:


#Neural Network model

X = np.array([])
y = np.array([])
for index in range(0, len(hydro_object_list)):
    if len(hydro_object_list[index].data) != len(date_int):
        pass
    else:
        data = {'Date': date_int, 'lst': data_lst, 'Sat': sat_object_list[index].data, 'Bog' : hydro_object_list[index].data}  
        df = pd.DataFrame(data) 
        xx1 = preprocessing.normalize([np.array(sat_object_list[index].data)])
        xx2int = preprocessing.normalize([np.array(datalstdatalst)])
        xx2 = xx2int * len(hydro_object_list)
        yy = preprocessing.normalize([np.array(hydro_object_list[index].data)])
        for value in xx1:
            X = np.append(X, value, axis = 0)
        for value in xx2:
            X = np.append(X, value, axis = 0)
        for value in yy:
            y = np.append(y, value, axis = 0)   
            y = np.append(y, value, axis = 0)
X = X.reshape(-1,1)


# In[150]:


#Neural Network


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1/24), random_state=200)

nn = MLPRegressor(
    activation='relu',
    solver = 'lbfgs',
    hidden_layer_sizes=(10,1),
    alpha=0.01,
    max_iter = 2000000,
    random_state=20,
    early_stopping=True,
    learning_rate_init = 0.5,
    learning_rate = 'adaptive'
)
nn.fit(X_train, y_train)
#predicting new values
pred = nn.predict(X_test)
nn.score(X_test, y_test)

neural_model = neural(sat_object_list[-1].name, nn)


# In[163]:


print(len(y_test))


# In[174]:


#Plotting Neural Network Model
plt.plot(date_vec_sen[-127:-1], y_test, label = "Recorded")
plt.plot(date_vec_sen[-127:-1], pred, label = "Predicted")

plt.gcf().autofmt_xdate()
#plt.xaxis.set_major_locator(plt.MaxNLocator(6))
plt.legend()
plt.ylabel('Normalized Water Table Depth', fontsize = 10, font = "Times New Roman")
plt.xlabel('Date', fontsize = 10)
plt.xticks(["June 2019"])
plt.title('Neural Network prediction of DBH5 Water Table (Sentinel and LST)', fontsize = 16, font = "Times New Roman")


# In[526]:


#Utilizing regression model to predict groundwater levels
#predicting dataset

from matplotlib import dates as mpl_dates
predict_vec = []
res_vec = []
for i in range (0, len(test_sat)):
    data = {'Date': date_int[i], 'lst': data_lst[i], 'Sat': test_sat[i], 'Bog' : test_hy[i]}  
    predicted = model.predict([[data_lst[i], test_sat[i]]])
    predicted = float(predicted.reshape(-1))
    predict_vec.append(predicted)

plt.figure()
plt.plot(to_date_done(date_int), test_hy, label = 'Recorded')
plt.plot(to_date_done(date_int), predict_vec, label = 'Predicted')
plt.gcf().autofmt_xdate()
#plt.xaxis.set_major_locator(plt.MaxNLocator(6))
plt.legend()
plt.ylabel('Water Table Depth (cm)', fontsize = 10)
plt.xlabel('Date', fontsize = 10)
#plt.xticks([])
plt.title('Multiregressive Prediction of ' + str(choice_name) +  ' Water Table (Sentinel and LST)', fontsize = 10)


# In[177]:


get_ipython().run_line_magic('store', 'linregmodel')
get_ipython().run_line_magic('store', 'neural_model')


# In[ ]:




