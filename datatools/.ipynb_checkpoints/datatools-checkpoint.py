import numpy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import math
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from scipy import stats
from scipy.optimize import curve_fit
import numpy as np
import os
import sys

def mad_rolling_filter(df,window):
    '''Rolling M.A.D. filter 
    only intakes pandas dataframes
    returns data frame 
    '''
    mad = lambda x: np.median(np.fabs(x - x.mean()))
    MAD = df.rolling(window).apply(mad, raw=True)
    median = df.rolling(window).median()
    df_mad = df[(df<median+4*MAD)&(df>median-4*MAD)]
    return df_mad


def exponential_growth(t,a1,a2):
  '''inputs:
  t - vector of times
  a1 - initial population at time 0
  a2 - specific growth rate   
  returns: modeled population based on exponential growth equation
  '''
  model = a1*np.exp(t*a2)
  return model





def exponential_fit(t,y):
  '''
  This fuction fits your data to an expoential curve using the curve_fit function from scipy.optimize
  returns: optimal parameters a1 and a2, standard error
  '''
  
  logy = np.log(np.asarray(y))
  result = stats.linregress(t,logy)
  a1 = np.exp(result[1])
  a2 = result[0]
  function = a1*np.exp(t*a2)
  opt_para, std_err = curve_fit(exponential_growth, t, np.asarray(y), p0 = [a1,a2])
  return opt_para,std_err
  # insert code here

def princax(u,v=None):
    '''
Principal axes of a vector time series.
Usage:
theta,major,minor = princax(u,v) # if u and v are real-valued vector components
    or
theta,major,minor = princax(w)   # if w is a complex vector
Input:
u,v - 1-D arrays of vector components (e.g. u = eastward velocity, v = northward velocity)
    or
w - 1-D array of complex vectors (u + 1j*v)
Output:
theta - angle of major axis (math notation, e.g. east = 0, north = 90)
major - standard deviation along major axis
minor - standard deviation along minor axis
Reference: Emery and Thomson, 2001, Data Analysis Methods in Physical Oceanography, 2nd ed., pp. 325-328.
Matlab function: http://woodshole.er.usgs.gov/operations/sea-mat/RPSstuff-html/princax.html
    '''

    # if one input only, decompose complex vector
    if v is None:
        w = np.copy(u)
        u = np.real(w)
        v = np.imag(w)

    # only use finite values for covariance matrix
    ii = np.isfinite(u+v)
    uf = u[ii]
    vf = v[ii]

    # compute covariance matrix
    C = np.cov(uf,vf)

    # calculate principal axis angle (ET, Equation 4.3.23b)
    theta = 0.5*np.arctan2(2.*C[0,1],(C[0,0] - C[1,1])) * 180/np.pi

    # calculate variance along major and minor axes (Equation 4.3.24)
    term1 = C[0,0] + C[1,1]
    term2 = ((C[0,0] - C[1,1])**2 + 4*(C[0,1]**2))**0.5
    major = np.sqrt(0.5*(term1 + term2))
    minor = np.sqrt(0.5*(term1 - term2))

    return theta,major,minor

def map_capemend(ds_x,ds_y=None, ds_z = None, color=False,vec=False,extent= [-125.6,-124.07,37.75,42.13],scl=1, labelz=False, qlabel=False, colorstyle = 'viridis', zmin=None,zmax=None ):
    '''
Maps two xarray data sets in the Cape Mendocino region with vectors on each gridpoint to a LambertConformal projection.
The lat/lon extent for this mapping is lon=(-125.6,-124.07) lat=(37.75,42.13)
    
Color option is what the background grid data will be colored with.
For example: map_capemend(ds_x,ds_y,color='x') maps ds_x data to each gridpoint under the vectors mapped with ds_x and ds_y.
If there is an added 3rd data set,it will be assumed that the ds_x and ds_y are the vector components and ds_z is the coloring option.
Spatial varible must be in "lat" and "lon" for a single time (if time varible applies).
scl is scaling of vectors. Input float.

default colorstyle is 'RdBu_r', other recomended is 'viridis'
    '''
    proj = ccrs.LambertConformal()
    fig = plt.figure(figsize=(15, 8))
    ax = plt.axes(projection=proj)
    if ((ds_y is None)or(color=='x'))and(labelz is False):
        ds_x.plot(transform=ccrs.PlateCarree(), cmap = colorstyle, vmin = zmin,vmax = zmax )
    if ((ds_y is not None)and(color=='y'))and(labelz is False)and(ds_z is None):
        ds_y.plot(transform=ccrs.PlateCarree(), cmap = colorstyle, vmin = zmin,vmax = zmax)
    if ((ds_y is None)or(color=='x'))and(labelz is not False):
        ds_x.plot(transform=ccrs.PlateCarree(), cmap = colorstyle, cbar_kwargs={'label':labelz},vmin = zmin,vmax = zmax)
    if ((ds_y is not None)and(color=='y'))and(labelz is not False)and(ds_z is None):
        ds_y.plot(transform=ccrs.PlateCarree(), cmap = colorstyle, cbar_kwargs={'label':labelz},vmin = zmin,vmax = zmax)
    if ((ds_z is not None)or(color=='z'))and(labelz is False):
        ds_z.plot(transform=ccrs.PlateCarree(), cmap = colorstyle,vmin = zmin,vmax = zmax) 
    if ((ds_z is not None)or(color=='z'))and(labelz is not False):
        ds_z.plot(transform=ccrs.PlateCarree(), cmap = colorstyle, cbar_kwargs={'label':labelz},vmin = zmin,vmax = zmax)
    
    coast_10m = cfeature.NaturalEarthFeature("physical", "land", "10m", edgecolor="k", facecolor="0.8")

    ax.add_feature(coast_10m)


    if (ds_y is not None)and(ds_z is None)and(vec is True):
            ax.quiver(np.asarray(ds_x['lon']), np.asarray(ds_x['lat']), np.asarray(ds_x[0]), np.asarray(ds_y[0]),transform=ccrs.PlateCarree(),scale=scl, label = qlabel )
    if (ds_y is not None)and(ds_z is not None):
            ax.quiver(np.asarray(ds_x['lon']), np.asarray(ds_x['lat']), np.asarray(ds_x), np.asarray(ds_y),transform=ccrs.PlateCarree(),scale=scl, label = qlabel)
    gl = ax.gridlines(draw_labels=True)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    ax.set_extent(extent)
    
def get_data_paths_from_binary(path_to_data,variable,delim='.',file_end='1'):
    '''This function returns a list of paths to the files that you want given the path to the data and the varible directory.
        e.g.
        path_to_data = ./home/user/data/ 
        varible = var1
        get_data_path_from_binary(path_to_data,variable,delim='.',number_delim_until_end=4,file_end='1')
        where delim, number_of_delim_until_end,file_end selects for what file ending will be chosen
        returns
        ['./home/user/data/var1/d.a.t.a.1',''./home/user/data/var1/d.a.t.a.2'',''./home/user/data/var1/d.a.t.a.3'']
        '''
    all_paths = []
    filename_ = []
    for filename in os.listdir(path_to_data + variable):
        f = os.path.join(path_to_data,variable, filename)
    # checking if it is a file
        if filename.split(delim)[-1]==file_end:
            all_paths.append(filename)
            filename_.append(f)
    return all_paths, filename_
