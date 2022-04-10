Binary Event Classification on UUSS Data
===============================================
</p>

<p align="left">
<b><a href="#overview">Overview</a></b>
|
<b><a href="#Data">Data</a></b>
|
<b><a href="#Models">Models</a></b>
|
<b><a href="#credits">Credits</a></b>

</p>


Overview
-----

The University of Utah Seismograph Stations ([UUSS]) has been monitoring seismicity in Utah for decades. This work uses just a few years (2012-2017) of the UUSS seismicity catalog to test how well deep learning can assign event type using waveform data from individual stations. Although many event types are observed during routine processing, discrimination of quarry blasts and earthquakes can be among the most time consuming. This work focuses on binary event discrimination for quarry blasts and earthquakes at local distances (< 350 km).

What we found is that we are able to match event types in Utah very well using a variety of architectures when we rely on event spectrograms (90 seconds in duration). Each quarry site produces many examples, and even though examples from an individual mine site can vary significantly, identifying new examples from these mine sites is successful. Additionally, we were able to make successful predictions using both vertical and triaxial sensors using one model, which makes this a practical, flexible and often highly accurate alternative or compliment to waveform matching methods.

<p align="center"><img src="https://github.com/quapity/UUSS_LSTM_classification/raw/master/figures/screen1.png"></p>

Data
-----

In data directory includes a csv of the catalog metadata we used. This does not include the data arrays, but gives users the first arrival pick times for stations and includes some event data (mag, type, etc). Using [obspy] users can download data arrays for use in whatever format is desirable. Alternative formats and additional Utah event catalogs through 2020 can be found at https://doi.org/10.31905/RDQW00CT.


Models
-----

The LSTM models we used in this study were built with Keras and are provided here. Subsequent studies that leverage this dataset use pytorch, for example [semi-supervised learning] for event discrimination. 

* Load the models (HDF5, built with Tensorflow) using Keras
    - [Tensorflow]
    - [Keras]
    
  

Credits
------------
Reference: 

Linville, L., Pankow, K., & Draelos, T. (2019). Deep learning models augment analyst decisions for event discrimination. Geophysical Research Letters, 46(7), 3643-3651.


* [Tensorflow]
* [Keras]
* [Obspy]

[Keras]:https://keras.io/
[Tensorflow]:https://www.tensorflow.org/
[Obspy]:https://github.com/obspy/obspy/wiki
[UUSS]:https://quake.utah.edu
[semi-supervised learning]:https://github.com/sandialabs/shadow


