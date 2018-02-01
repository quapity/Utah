#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 08:26:12 2018

@author: lisa
"""
import pandas as pd
import pickle
import numpy as np

import matplotlib.pyplot as plt
#from mpl_toolkits.basemap import Basemap

plt.switch_backend("nbagg")
plt.style.use('seaborn-white')
plt.rcParams.update({'font.size': 18})
plt.rcParams['savefig.dpi'] = 80
from obspy.clients.fdsn import Client
client = Client("IRIS")

from scipy.signal import spectrogram
from obspy.signal.rotate import rotate_ne_rt
from obspy.geodetics.base import gps2dist_azimuth
from obspy.core import UTCDateTime

from keras.models import load_model
model = load_model('models/LSTM_40_1445.h5')

#%%
def rotater(tr,inv,evlo,evla):
    lat= inv[0][0].latitude
    lon = inv[0][0].longitude
    dist,baz,_ = gps2dist_azimuth(lat,lon,evla,evlo)
    e = rotate_ne_rt(tr[0].data,tr[1].data,baz)
    tmptr = tr.copy()
    tmptr[0].data = e[0]
    tmptr[1].data = e[1]
    return tmptr,lat,lon,dist

def rotater_single(tr,inv,evlo,evla):
    lat= inv[0][0].latitude
    lon = inv[0][0].longitude
    dist,baz,_ = gps2dist_azimuth(evla,evlo,lat,lon)
    #e = rotate_ne_rt(tr[0].data,tr[1].data,baz)
    #tmptr = tr.copy()
    #tmptr[0].data = e[0]
    #tmptr[1].data = e[1]
    return lat,lon,dist
    
def scale_transform(x):
    x = np.vstack(np.nan_to_num(x/np.max(x)))
    return np.transpose(1+(-np.log(x+.0002)/np.median(np.log(x+.0002))))

def scale_transform_view(x):
    x = np.vstack(np.nan_to_num(x/np.max(x)))
    return 1+(-np.log(x+.0002)/np.median(np.log(x+.0002)))

def parse_dbselect(filename = 'dbcat'):
    ddict ={'id':[],'type':[],'station':[],'net':[],'chan':[],'impulsive':[],
        'datetime':[],'mag':[],'lat':[],'lon':[],'depth':[],'weight':[]}
    ecat = pd.read_fwf(filename)
        
    for file,wt in zip(['le','qb'],[1,2]):
        with open(file,'r')as f:
            for line in f.readlines():
                line = line.strip()
              
                if line.startswith('201'):
                    ecount= int(line[138:146])
                    idx =ecat[ecat['DbId'] == ecount].index[0]
                    
                if line.endswith(' 01') and int(line[16:17]) <=wt:
                    lline = line.replace(' ','0')
                    ddict['id'].append(int(ecount))
                    ddict['type'].append(ecat['Typ'][idx])
                    ddict['station'].append(line[:5].strip())
                    ddict['net'].append(line[5:7])
                    ddict['chan'].append(line[7:12].strip())
                    ddict['impulsive'].append(line[12:16].strip())
                    ddict['weight'].append(line[16:17])
                    if int(lline[30:34]) > 99:
                        sec = int(lline[30:34])/100.00
                    else:
                        sec = int(lline[30:34])
                    ddict['datetime'].append(UTCDateTime(int(lline[17:21]),int(lline[21:23]),
                          int(lline[23:25]),int(lline[25:27]),
                          int(lline[27:29])) + sec)                        
                    ddict['mag'].append(float(ecat['Mag'][idx]))
                    ddict['lat'].append(ecat['Lat'][idx])
                    ddict['lon'].append(ecat['Lon'][idx])
                    ddict['depth'].append(ecat['Depth'][idx])
    return pd.DataFrame(ddict)
#%%
pre_filt = [0.001, 0.005, 45, 50]
def preproc_data(filename ='out_from_parse_dbselect', to_file=False, num_events=2, **ev_id):
    #df = pd.read_pickle(filename)
    if ev_id:
        df = filename[filename['id'] == ev_id['ev_id']]
    else:
        df = filename.iloc[:num_events]
    
    df = df.reset_index(drop=True)
    pdf= None
    count = 0
    duration = 5*60
    for i in range(len(df)):
        data = {'x':[],'y':[],'chan':[],'net':[],'station':[],'datetime':[],'depth':[],'dsize':[],
            'id':[],'impulsive':[],'evlo':[],'evla':[],'mag':[],'type':[],'weight':[],
            'stlo':[],'stla':[],'lbl':[],'dist':[]}
        
        chan = df['chan'][i]
        evlo = df['lon'][i]
        evla = df['lat'][i]
        #chan = str(chan[0]+'H?')
        #print(count)
        try:
            tr = client.get_waveforms(station=df['station'][i],network=df['net'][i],
                              channel='*',location='*',
                              starttime=UTCDateTime(df['datetime'][i])-10,
                              endtime=UTCDateTime(df['datetime'][i]) + duration,attach_response=True)
            inv = client.get_stations(starttime=UTCDateTime(df['datetime'][i])-10,
                                  endtime=UTCDateTime(df['datetime'][i]) + duration,
                                  station=df.station[i],network=df.net[i],location='*',
                                  channel='*',level='response')
            tr =tr.select(channel = '[E,H,B]H?')
            if len(tr) > 3:
                tr = tr.select(channel='H??')
            tr.merge(fill_value=0)
            #tr = tr.remove_sensitivity()
            tr.remove_response(inventory=inv,pre_filt=pre_filt)
            tr.resample(100)
            tr.detrend()
            tr.taper(.01)
            
            if len(tr) == 1:
                stla,stlo,dist = rotater_single(tr,inv,evlo,evla)
                tr.filter('highpass',freq=1.0)
                f, t, S0 = spectrogram(tr[0].data, tr[0].stats.sampling_rate)
                S1 = np.zeros_like(S0)
                data['x'].append(np.array([S0[3:51,1:41]/np.max(S0[3:51,1:41]),S1[3:51,1:41],S1[3:51,1:41]]))
                data['dsize'] = 'single'
            elif len(tr) == 3:
                trn,stla,stlo,dist = rotater(tr,inv,evlo,evla)
                trn.filter('highpass',freq=1.0)
                f, t, S0 = spectrogram(trn[0].data, trn[0].stats.sampling_rate)
                f, t, S1 = spectrogram(trn[1].data, trn[1].stats.sampling_rate)
                f, t, S2 = spectrogram(trn[2].data, trn[2].stats.sampling_rate)
                data['x'].append(np.array([S0[3:51,1:41]/np.max(S0[3:51,1:41]),
                                 S1[3:51,1:41]/np.max(S1[3:51,1:41]),S2[3:51,1:41]/np.max(S2[3:51,1:41])]))
                data['dsize'] = 'tri'
            tmp =df.loc[i]
            tmp.stla = stla
            tmp.stlo = stlo
            tmp.dist = dist
            data['station'].append(tmp.station)
            data['chan'].append(tmp.chan)
            data['net'].append(tmp.net)
            data['evlo'].append(tmp.lon)
            data['evla'].append(tmp.lat)
            data['datetime'].append(tmp.datetime)
            data['depth'].append(tmp.depth)
            data['mag'].append(tmp.mag)
            data['id'].append(tmp.id)
            data['type'].append(tmp.type)
            data['weight'].append(tmp.weight)
            data['impulsive'].append(tmp.impulsive)
            data['stla'].append(tmp.stla)
            data['stlo'].append(tmp.stlo)
            data['dist'].append(tmp.dist)
            if tmp.type == 'le':
                data['y'].append([1,0])
                data['lbl'].append(0)
            elif tmp.type == 'qb':
                data['y'].append([0,1])
                data['lbl'].append(1)             
            if pdf is None:
                pdf = pd.DataFrame(data)
            else:
                pdf = pdf.append(pd.DataFrame(data))
            
            count+=1
        except: #ValueError :
            print(i)
        if num_events > 50:
            print(i)
    pdf = pdf.reset_index(drop=True)
    predictions1 = model.predict(np.array([scale_transform(x) for x in pdf['x']]))
    pred1 = np.argmax(predictions1,axis=1)
    pdf['predictions'] = [np.array(x) for x in predictions1]
    pdf['preds'] = pred1
    #pdf[pdf['pred'] != pdf['lbl']]
    print('sucessfully processed ' + str(count) + ' events')
    if to_file == True:   
        stg = 'proc_'+filename
        pdf.to_pickle(stg)
        pdf= None
    else:
        return pdf
        
#%%
            

#%%

 
pred = model.predict(np.array([scale_transform(x) for x in pdf['x']]))
def eval_pred(method='viz'):
    '''look at spectrograms type = 'viz', or look a table'''
    if method == 'viz':
        m = Basemap(llcrnrlat=-35,urcrnrlat=43,llcrnrlon=-115.5,urcrnrlon=-106,resolution='h',epsg = 32142)
        m.drawstates()
        for each in range(len(pdf)):
            plt.subplot(211)
            title = 'ID: ' + str(pdf.iloc[each]['id'])+' Label: '+ str(pdf.iloc[each]['type'])+ ' Prediction: ' + str(pred[each])
            plt.imshow(scale_transform_view(pdf.iloc[each]['x']),origin='lower',
               cmap = 'RdGy_r',aspect='auto',interpolation='gaussian');plt.colorbar()
            plt.title(title)
            plt.subplot(212)
            x,y = m(pdf.iloc[each]['evlo'],pdf.iloc[each]['evla'])
            plt.scatter(x,y,s=100)
            plt.show()



