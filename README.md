Seismic Event Clustering using Affinity Propagation
===============================================
</p>

<p align="center">
<b><a href="#overview">Overview</a></b>
|
<b><a href="#set-up">Set-up</a></b>
|
<b><a href="#set-up">Other Stuff</a></b>

</p>

![ScreenShot](https://github.com/quapity/event_cluster/raw/master/mineral_zoom1.png)


Overview
-----

Uses Affinity Propogation (sci-kit implementation) to cluster a catalog of seismic events.  


Set-Up
------------

### Dependencies
* Relies on Numpy,Scipy,Pandas,and Geopy. Most can be installed with pip or ship with Anaconda
    - http://pandas.pydata.org
    - http://scipy.org
    - https://github.com/geopy/geopy
* Requires Obspy for seismic routines and data fetch 
    - https://github.com/obspy/obspy/wiki
* ANF catalog import stolen from old version of detex, a python code for subspace detection. Check it out at:
    - https://github.com/dchambers/detex.git 
    - or to install the latest: git+git://github.com/d-chambers/detex


Other Stuff
-----
* Affinity Propogation clustering for event catalog
* Input is local seismic data OR date range and station list for  obspy fetched waveform data
* Output is a picktable (as a dataframe), and templates for each detection -organized in day directories
* Current support for ANF catalog 

![ScreenShot](https://github.com/quapity/event_cluster/raw/master/review_clusters-02.png)

Tutorial
----------

### General Usage

* LDK.detection_function.detect('YYYY','MM','DD','HH','MM','SS',duration=7200,ndays=1,wb=1)
    - The first 6 args are start date/time. These pipe to obspy UTCDatetime
    - duration: number of seconds (2 hours is the smallest allowable increment)
    - ndays:    number of days to process from start date/time
    - wb:       the station list to use. Referenced from lists in the last half of Util.py

### Basic Tutorial

Credits
------------




