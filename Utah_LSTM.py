#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 08:26:12 2018

@author: lisa
"""
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import adjustText

plt.switch_backend("nbagg")
plt.style.use('seaborn-white')
plt.rcParams.update({'font.size': 18})
plt.rcParams['savefig.dpi'] = 80
plt.rcParams['figure.figsize']=14,14
from obspy.clients.fdsn import Client
client = Client("IRIS")

from scipy.signal import spectrogram
from obspy.signal.rotate import rotate_ne_rt
from obspy.geodetics.base import gps2dist_azimuth
from obspy.core import UTCDateTime

from keras.models import load_model
model = load_model('models/LSTM_40_1445.h5')

#%matplotlib inline
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

def plot_txt(x,y,text):
    
    font = {'family': 'Arial',
        'color':  'black',
        'weight': 'normal',
        'size': 12,
        }
    texts = []
    for xt, yt, s in zip(x, y, text):
        texts.append(plt.text(xt, yt, s,bbox=dict(facecolor='white', edgecolor='white',
                     boxstyle='round,pad=.2', alpha = .4),fontdict = font))
    return texts

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
def preproc_data(filename ='out_from_parse_dbselect', to_file=False, num_lines=2, **ev_id):
    #df = pd.read_pickle(filename)
    if ev_id:
        df = filename[filename['id'] == ev_id['ev_id']]
    else:
        df = filename.iloc[:num_lines]
    
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
        
    pdf = pdf.reset_index(drop=True)
    predictions1 = model.predict(np.array([scale_transform(x) for x in pdf['x']]))
    pred1 = np.argmax(predictions1,axis=1)
    pdf['predictions'] = [np.array(x) for x in predictions1]
    pdf['preds'] = pred1
    num_events = len(set(pdf['id']))
    #pdf[pdf['pred'] != pdf['lbl']]
    print('sucessfully processed ' + str(count) + ' examples from ' + str(num_events))
    if to_file == True:   
        stg = 'proc_'+filename
        pdf.to_pickle(stg)
        pdf= None
    else:
        return pdf
        
#%%
            

#%%

 
#pred = model.predict(np.array([scale_transform(x) for x in pdf['x']]))
#def eval_pred(method='viz'):
#    '''look at spectrograms type = 'viz', or look a table'''
#    if method == 'viz':
#        m = Basemap(llcrnrlat=-35,urcrnrlat=43,llcrnrlon=-115.5,urcrnrlon=-106,resolution='h',epsg = 32142)
#        m.drawstates()
#        for each in range(len(pdf)):
#            plt.subplot(211)
#            title = 'ID: ' + str(pdf.iloc[each]['id'])+' Label: '+ str(pdf.iloc[each]['type'])+ ' Prediction: ' + str(pred[each])
#            plt.imshow(scale_transform_view(pdf.iloc[each]['x']),origin='lower',
#               cmap = 'RdGy_r',aspect='auto',interpolation='gaussian');plt.colorbar()
#            plt.title(title)
#            plt.subplot(212)
#            x,y = m(pdf.iloc[each]['evlo'],pdf.iloc[each]['evla'])
#            plt.scatter(x,y,s=100)
#            plt.show()


#%%


def plt_events(data):
    fig= plt.figure()
    latmin = min(data['stla']) -.1
    latmax = max(data['stla'])+.1
    lomin = min(data['stlo']) -.1
    lomax = max(data['stlo'])+.1
    m = Basemap(llcrnrlat=latmin,urcrnrlat=latmax,llcrnrlon=lomin,
        urcrnrlon=lomax,resolution='h',epsg = 32142)
    #m.fillcontinents(color='grey',lake_color='black',zorder=0)
    m.arcgisimage(service='World_Shaded_Relief', xpixels = 1500, verbose= True,zorder=0)
    m.drawstates()
    m.drawcountries()
    #m.drawcoastlines()
    
    parallels = np.arange(35,47,1)
    # labels = [left,right,top,bottom]
    m.drawparallels(parallels,linewidth=0.1,labels=[True,False,False,False])
    meridians = np.arange(-115,-105,1)
    m.drawmeridians(meridians,linewidth = 0.1,labels=[False,False,False,True])
    for event in list(set(data.id)):
        idx = np.where(np.array(data['id']) == event)[0]
        #predictions = np.array(model.predict(np.array([scale_transform(x) for x in testcat['sx'][idx]])) )   
        lat=data.iloc[idx[0]].evla
        lon = data.iloc[idx[0]].evlo
        lo,la = m(lon,lat)
        slat,slo,stcol = [],[],[]
        for i in idx:
            if data.iloc[i]['lbl'] == 0:
                color = 'b'
        
            else:
                color= 'r'
            if data.iloc[i]['lbl'] == data.iloc[i]['preds']:
                   stcol ='white'
            else:
                stcol = 'black'
    
            x,y = m(data.iloc[i]['stlo'],data.iloc[i]['stla'])
            plt.plot([lo,x],[la,y],c=color,linewidth=.4)
            plt.scatter(x,y,facecolor=stcol,edgecolor='black',s=70,alpha =.2,zorder=500000)
        #plt.scatter(x,y,facecolor=color,edgecolor=color,alpha=.6,s= 30)
        plt.scatter(lo,la,facecolor=color,edgecolor=color,s=20, alpha = .6)
    
    slat,slo =[],[]
    for sta in list(set(data.id)):
        if sta > 0:
            idx = np.where(np.array(data['id']) == int(sta))[0]
            slat.append(data.iloc[idx[0]]['evla'])
            slo.append(data.iloc[idx[0]]['evlo'])
        else:
            idx = np.where(np.array(data['station']) == sta)[0]
            slat.append(data.iloc[idx[0]]['stla'])
            slo.append(data.iloc[idx[0]]['stlo'])
    x,y = m(slo,slat)
    #plt.scatter(x,y,facecolor='white',edgecolor='cornflowerblue',s=70, zorder = 50000)
    text= plot_txt(x,y,list(set(data.id)))
    adjustText.adjust_text(text, arrowprops=dict(arrowstyle="-", color='black', lw=0.1),
            autoalign='y')
        
    #plt.savefig('intro_map3.eps',bbox_inches='tight', transparent =True)
    return fig    

def make_stats(data,filename='stats_dict.html'):
    statsdict ={'id':[],'type':[],'stations':[],'datetime':[],'mag':[],
                'evla':[],'evlo':[],'depth':[],'class_prediction':[],'how_sure':[]}
    for ev in list(set(data.id)):
        idx = np.where(np.array(data.id) == ev)[0]
        statsdict['id'].append(ev)
        statsdict['type'].append(data.iloc[idx[0]]['lbl'])
        statsdict['stations'].append(np.array(data.iloc[idx]['station']))
        statsdict['evla'].append(data.iloc[idx[0]]['evla'])
        statsdict['evlo'].append(data.iloc[idx[0]]['evlo'])
        statsdict['datetime'].append(data.iloc[idx[0]]['datetime'])
        statsdict['mag'].append(data.iloc[idx[0]]['mag'])
        statsdict['depth'].append(data.iloc[idx[0]]['depth'])
        statsdict['class_prediction'].append(np.argmax(np.sum(data.iloc[idx]['predictions'])))
        statsdict['how_sure'].append(max(np.sum(data.iloc[idx]['predictions']))/len(data.iloc[idx]))
        pd.DataFrame(statsdict).to_html(filename)
    return pd.DataFrame(statsdict)
#%%
#fig= plt.figure()
#m = Basemap(llcrnrlat=35.5,urcrnrlat=45.1,llcrnrlon=-115,
#        urcrnrlon=-107.5,resolution='h',epsg = 32142)
##m.fillcontinents(color='grey',lake_color='black',zorder=0)
#m.arcgisimage(service='World_Shaded_Relief', xpixels = 1500, verbose= True,zorder=0)
#m.drawstates()
#m.drawcountries()
##m.drawcoastlines()
#
#parallels = np.arange(35,47,1)
## labels = [left,right,top,bottom]
#m.drawparallels(parallels,linewidth=0.1,labels=[True,False,False,False])
#meridians = np.arange(-115,-105,1)
#m.drawmeridians(meridians,linewidth = 0.1,labels=[False,False,False,True])
#for event in list(set(data.id)):
#    idx = np.where(np.array(data['id']) == event)[0]
#        #predictions = np.array(model.predict(np.array([scale_transform(x) for x in testcat['sx'][idx]])) )   
#    lat=data.iloc[idx[0]].evla
#    lon = data.iloc[idx[0]].evlo
#    lo,la = m(lon,lat)
#    slat,slo,stcol = [],[],[]
#    for i in idx:
#        if data.iloc[i]['lbl'] == 0:
#            color = 'b'
#        
#        else:
#            color= 'r'
#        if data.iloc[i]['lbl'] == data.iloc[i]['preds']:
#            stcol ='white'
#        else:
#            stcol = 'black'
#
#        x,y = m(data.iloc[i]['stlo'],data.iloc[i]['stla'])
#        plt.plot([lo,x],[la,y],c=color,linewidth=.4)
#        plt.scatter(x,y,facecolor=stcol,edgecolor=color,s=70,alpha =.2,zorder=500000)
#        #plt.scatter(x,y,facecolor=color,edgecolor=color,alpha=.6,s= 30)
#    plt.scatter(lo,la,facecolor=color,edgecolor=color,s=20, alpha = .6)
#
#slat,slo =[],[]
#for sta in np.concatenate((list(set(data.station)),list(set(data.id)))):
#    if sta[0] == '6':
#        idx = np.where(np.array(data['id']) == int(sta))[0]
#        slat.append(data.iloc[idx[0]]['evla'])
#        slo.append(data.iloc[idx[0]]['evlo'])
#    else:
#        idx = np.where(np.array(data['station']) == sta)[0]
#        slat.append(data.iloc[idx[0]]['stla'])
#        slo.append(data.iloc[idx[0]]['stlo'])
#x,y = m(slo,slat)
##plt.scatter(x,y,facecolor='white',edgecolor='cornflowerblue',s=70, zorder = 50000)
#text= plot_txt(x,y,np.concatenate((list(set(data.station)),list(set(data.id)))))
#adjustText.adjust_text(text, arrowprops=dict(arrowstyle="-", color='black', lw=0.1),
#            autoalign='y')
#        
##plt.savefig('intro_map3.eps',bbox_inches='tight', transparent =True)
#plt.show()
