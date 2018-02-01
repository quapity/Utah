Binary LSTM Event Classification on UUSS Data
===============================================
</p>

<p align="center">
<b><a href="#overview">Overview</a></b>
|
<b><a href="#set-up">Set-up</a></b>
|
<b><a href="#set-up">Usage</a></b>

</p>

Overview
-----

Use a model (CNN,LSTM or combined) trained on data within Utah (2012-2017) to make predictions on new events. 
The model takes event spectrograms (1-3 channel) and classifies events as either quarry blasts or local earthquakes based on the spectral content.

MODEL INPUT
![ScreenShot](https://github.com/quapity/event_cluster/raw/master/screen1.png)

Set-Up
------------

### Dependencies
* Relies on Numpy,Scipy,Pandas,and Geopy. Most can be installed with pip or ship with Anaconda
    - http://pandas.pydata.org
    - http://scipy.org
    - https://github.com/geopy/geopy
* Waveform data from Obspy  
    - https://github.com/obspy/obspy/wiki
* Database files from UUSS 
    - eventlist, filename: dbcat 
    - picktables by event type, filenames: le,qb
* I use the super handy adjustText(https://github.com/Phlya/adjustText) to plot non-overlapping text on map (esp for Mark)
    - pip install adjustText
* And I gave you come clumsy plotting functions that use Basemap
    - mpl_toolkit
    - 


Other Stuff
-----
* Model Training occured on ~ 13k events
 

![ScreenShot](https://github.com/quapity/UUSS_LSTM_classification/raw/master/screen3.png)

Useage
----------

### General Usage

* Once you have db files [dbcat,le,qb] you can call any function in the pipeline. The following will get you a map and table for the first 20 events in your database:
data = preproc_data(parse_dbselect(),num_events=20)
plt_events(data)
make_stats(data)


Credits
------------

*tensorflow
*Keras
*obspy




