# britbogs_gpt

The goal of this code snippet is to create a neural network model
that predicts the water table depth of a British peat bog with satellite data.

The model is trained from existing data from a the Boltonfellend Bog in Carlisle.
This bog was chosen due to it having a proportion of area undergoing restoration
#and a portion of it still experiencing damage from anthropogenic activities. 

Using GRD data from Sentinel 1 and LST-day data from MODIS (previously shown
to accurately predict water table levels), a neural network is developed.
The initial neural network will train on data from water table depth measurements
from 13 boreholes in the bog alongside sentinel and MODIS data for the exact
coordinates of the peat bog for 3 years (early 2017 to late 2019). Later additions will train the model 
on more borehole point within the bog and within other surrounding bogs to get
wide variety of bogs within the training dataset. Further satellite data from
MODIS, Sentinel1, Sentinel2, and LANDSAT can be considered. 

For simpler analysis of predicted water table depth from a bog with data past 2019
can be done with a simple multi-regression analysis.

