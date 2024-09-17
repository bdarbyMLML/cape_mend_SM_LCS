import cartopy.crs as ccrs
import cartopy.feature as cfeature
import math
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from scipy.optimize import curve_fit
import numpy as np
import os
import sys

from scipy import stats, signal
def _filt(x,wts,return_weights=False):
    """
Private function to filter a time series and pad the ends of the filtered time series with NaN values. For N weights, N/2 values are padded at each end of the time series. The filter weights are normalized so that the sum of weights = 1.

Inputs:

x - the time series (may be 2d, will be filtered along columns)
wts - the filter weights
return_weights - if True, return the filter weights instead of a filtered time series (default: False)

Output:
- the filtered time series (default)
- filter weights (if `return_weights=True` is specified)
    """

    # convert to 2D array if necessary (general case)
    ndims = np.ndim(x)
    if ndims == 1:
        x = np.expand_dims(x,axis=1)

    # normalize weights
    wtsn = wts*sum(wts)**-1 # normalize weights so sum = 1

    if return_weights==False:
        # Convolve using 'direct' method. In older versions of scipy, this has to
        # be specified because the default 'auto' method could decide to use the
        # 'fft' method, which does not work for time series with NaNs. In newer
        # versions, there is no method option.
        try:
            xf = signal.convolve(x,wtsn[:,np.newaxis],mode='same',method='direct')
        except:
            xf = signal.convolve(x,wtsn[:,np.newaxis],mode='same')

        # note: np.convolve may be faster
        # http://scipy.github.io/old-wiki/pages/Cookbook/ApplyFIRFilter

        # pad ends of time series
        nwts = len(wts) # number of filter weights
        npad = int(np.ceil(0.5*nwts))
        xf[:npad,:] = np.nan
        xf[-npad:,:] = np.nan

        # return array with same number of dimensions as input
        if ndims == 1:
            xf = xf.flatten()
    elif return_weights==True:
        # return normalized weights instead of filtered time series
        xf = wtsn
    else:
        raise('return_weights must be a Boolean')

    return xf
def pl64(x=None,dt=1,T=33,return_weights=False):
    """
Filter a time series x with the PL64 filter. If x is 2D, the time series will be filtered along columns.

Inputs:
x - a numpy array to be filtered
dt - sample interval (hours), default = 1
T - half-amplitude period (hours), default = 33
return_weights - Boolean indicating whether to return the filter weights instead of a filtered time series, default = False

Output:
- numpy array of filtered time series, same size as input with ends NaN values at start and end (default)
- numpy array of filter weights (if `return_weights=True` is specified)

Reference: CODE-2: Moored Array and Large-Scale Data Report, WHOI 85-35
    """

    Tn=float(T)/dt # normalized cutoff period
    fqn=1./Tn # normalized cutoff frequency
    nw = int(np.round(64/dt)) # number of weights on one side

    # create filter weights
    j = np.arange(1,nw)
    tn = np.pi*j
    den=fqn*fqn*tn**3
    wts = (2*np.sin(2*fqn*tn)-np.sin(fqn*tn)-np.sin(3*fqn*tn))/den

    # make symmetric
    wts = np.hstack((wts[::-1],2*fqn,wts))

    xf = _filt(x,wts,return_weights)
    return xf



def lancz(x=None,dt=1,T=40,return_weights=False):
    """
Filter a time series x with cosine-Lanczos filter. If x is 2D, the time series will be filtered along columns.

The default half amplitude period of 40 hours corresponds to a frequency of 0.6 cpd. A half amplitude period of 34.29h corresponds to 0.7 cpd. The 40 hour half amplitude period is more effective at reducing diurnal-band variability but shifts periods of variability in low passed time series to >2 days.

Inputs:
x - a numpy array to be filtered
dt - sample interval (hours), default = 1
T - half-amplitude period (hours), default = 40
return_weights - Boolean indicating whether to return the filter weights instead of a filtered time series, default = False

Output:
- numpy array of filtered time series, same size as input with ends NaN values at start and end (default)
- numpy array of filter weights (if `return_weights=True` is specified)

Reference: Emery and Thomson, 2004, Data Analysis Methods in Physical Oceanography. 2nd Ed., pp. 539-540. Section 5.10.7.4 - The Hanning window.
    """

    cph = 1./dt   # samples per hour
    nwts = int(np.round(120*cph)) # number of weights

    # create filter weights
    wts = signal.firwin(nwts,
                        1./T,
                        window='hamming',
                        nyq=cph/2.)

    xf = _filt(x,wts,return_weights)
    return xf



def filt(da,dim='time',dt=None,T=33,filter_name='pl64'):
    '''
Low pass filter a DataArray along a temporal dimension

Inputs:
da  - the DataArray to be filtered
dim - name of the temporal dimension (default 'time')
dt  - time interval of DataArray, in units of hours (default, selected automatically), for data obtained at hourly intervals, dt=1
      note: this must be consistent throughout the dataset. Specifying a value may be useful if the interval is nearly consistent, but not exactly
filter_name - name of the low pass filter (functions in physoce.tseries), should be 'pl64', 'pl66' or 'lancz' (default 'pl64')

Returns:
- the DataArray, filtered along the temporal dimension
    '''

    # automatically find time step if not specified by user
    if dt == None:
        dt_array = da[dim].diff(dim=dim)/np.timedelta64(1,'h')
        dt_unique = np.unique(dt_array)

        if len(dt_unique) == 1:
            dt = float(dt_unique)
        else:
            raise ValueError('time step must be consistent, or specified manually as input')

    # get filter weights
    if filter_name == 'pl64':
        wts = pl64(dt=dt,T=T,return_weights=True)
    elif filter_name == 'pl66':
        wts = ts.pl66(dt=dt,T=T,return_weights=True)
    elif filter_name == 'lancz':
        wts = lancz(dt=dt,T=T,return_weights=True)
    else:
        raise NameError('filter_name not understood, must be pl64, pl66 or lancz')

    # filter along specified dimension
    # follows example at: http://xarray.pydata.org/en/stable/user-guide/computation.html#rolling-window-operations
    weight = xr.DataArray(wts,dims=['window'])
    daf = da.rolling({dim:len(weight)},center='True').construct({dim:'window'}).dot(weight)

    return daf











def resample_exf_field(exf_field, timestep):
    orig_def = pyresample.geometry.SwathDefinition(lons=Lon, lats=Lat)
    targ_def = pyresample.geometry.SwathDefinition(lons=XC, lats=YC)


    interpolated_field =  pyresample.kd_tree.resample_gauss(orig_def, exf_field[timestep,:,:], targ_def,
                                                            radius_of_influence=500000,
                                                            fill_value=np.nan,
                                                            sigmas=100000, neighbours=100)
    return(interpolated_field)




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

def map_capemend(ds_x,ds_y=None, ds_z = None, color=False,vec=False,extent= [-125.6,-124.07,37.75,42.13],scl=1, labelz=False, qlabel=False, colorstyle = 'viridis', zmin=None,zmax=None, ax=None ,proj = ccrs.LambertConformal(),figsize=(15, 8)):
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
    
    fig = plt.figure(figsize=figsize)
    ax_1 = plt.axes(projection=proj)
    if ((ds_y is None)or(color=='x'))and(labelz is False):
        ds_x.plot(transform=ccrs.PlateCarree(), cmap = colorstyle, vmin = zmin,vmax = zmax , ax=ax)
    if ((ds_y is not None)and(color=='y'))and(labelz is False)and(ds_z is None):
        ds_y.plot(transform=ccrs.PlateCarree(), cmap = colorstyle, vmin = zmin,vmax = zmax, ax=ax)
    if ((ds_y is None)or(color=='x'))and(labelz is not False):
        ds_x.plot(transform=ccrs.PlateCarree(), cmap = colorstyle, cbar_kwargs={'label':labelz},vmin = zmin,vmax = zmax, ax=ax)
    if ((ds_y is not None)and(color=='y'))and(labelz is not False)and(ds_z is None):
        ds_y.plot(transform=ccrs.PlateCarree(), cmap = colorstyle, cbar_kwargs={'label':labelz},vmin = zmin,vmax = zmax, ax=ax)
    if ((ds_z is not None)or(color=='z'))and(labelz is False):
        ds_z.plot(transform=ccrs.PlateCarree(), cmap = colorstyle,vmin = zmin,vmax = zmax) 
    if ((ds_z is not None)or(color=='z'))and(labelz is not False):
        ds_z.plot(transform=ccrs.PlateCarree(), cmap = colorstyle, cbar_kwargs={'label':labelz},vmin = zmin,vmax = zmax, ax=ax)
    
    coast_10m = cfeature.NaturalEarthFeature("physical", "land", "10m", edgecolor="k", facecolor="0.8")

    ax_1.add_feature(coast_10m)


    if (ds_y is not None)and(ds_z is None)and(vec is True):
            ax_1.quiver(np.asarray(ds_x['lon']), np.asarray(ds_x['lat']), np.asarray(ds_x[0]), np.asarray(ds_y[0]),transform=ccrs.PlateCarree(),scale=scl, label = qlabel )
    if (ds_y is not None)and(ds_z is not None):
            ax_1.quiver(np.asarray(ds_x['lon']), np.asarray(ds_x['lat']), np.asarray(ds_x), np.asarray(ds_y),transform=ccrs.PlateCarree(),scale=scl, label = qlabel)
    gl = ax_1.gridlines(draw_labels=True)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    ax_1.set_extent(extent)
    
def get_data_paths_from_binary(path_to_data,variable,delim='.',file_end='1'):
    '''This function returns a list of file names and paths with filenames to the files that you want given the path to the data and the varible directory. Returns sorted
        e.g.
        path_to_data = ./home/user/data/ 
        varible = var1
        get_data_path_from_binary(path_to_data,variable,delim='.',file_end='1')
        where delim,file_end selects for what file ending will be chosen
        returns
        ['d.a.t.a1.1','d.a.t.a2.1','d.a.t.a3.1'],['./home/user/data/var1/d.a.t.a1.1',''./home/user/data/var1/d.a.t.a2.1'',''./home/user/data/var1/d.a.t.a3.1'']
        '''
    all_paths = []
    filename_ = []
    for filename in os.listdir(path_to_data + variable):
        f = os.path.join(path_to_data,variable, filename)
    # checking if it is a file
        if filename.split(delim)[-1]==file_end:
            all_paths.append(filename)
            filename_.append(f)
    return sorted(all_paths), sorted(filename_)

def convert_itter_to_datetime(number,datetime_start,timestep,shift_itter=0):
    '''converts an itteration number to a datetime'''
    itter_dt = datetime.fromtimestamp(datetime_start.timestamp()+(int(number)+int(shift_itter))*timestep)
    return itter_dt

def convert_binary_to_nc(file_name,file_path, shape, dims_list, coords_list, name,time_in_name_location=None, output_filepath='./my_netcdf.nc',to_nc_mode='w'):
    '''converts binary files to netcdf through xarray framework
    '''
    file = np.fromfile(file_path,'>f4')
    file = np.reshape(file, shape)
    if time_in_name_location!=None:
        time = [int(filename[time_in_name_location[0]:time_in_name_location[1]])]
    field = xr.DataArray(file,coords=coords_list,dims=dims_list).rename(name)
    field.to_netcdf(output_filepath,mode=to_nc_mode)
    field.close()

def f_grid(grid,latitude_name):
    rot_ear = 7.292e-5  # Earth's rotation rate in radians/s
    Rearth = 6371e3 # Earth's radius in m

    f = 2*rot_ear* np.sin(np.deg2rad(grid[latitude_name])) # convert lat to radians
    #rename for future use
    f = f.rename('Coriolis Frequancy') 
    f.attrs['long_name'] = 'Coriolis parameter'
    f.attrs['units'] = 's-1'
    return f
    
    
    
def calculate_divergence(uvel,vvel,dxC,dyC,rAz,f):
    field = np.zeros((np.shape(uvel)[0]-1,np.shape(uvel)[1]-1))
    numerator = np.diff(np.asarray(uvel[:,:]) * np.asarray(dxC), axis=0)[:, :-1] + np.diff(np.asarray(vvel[:,:]) * np.asarray(dyC), axis=1)[:-1, :]
    denominator = np.asarray(rAz[:-1, :-1])
    zeta = np.zeros_like(numerator)
    zeta[denominator != 0] = numerator[denominator != 0] / denominator[denominator != 0]
    field[:,:] = zeta / np.asarray(f[:-1, :-1])
    return field
def calculate_vorticity(uvel,vvel,dxC,dyC,rAz,f):
    field = np.zeros((np.shape(uvel)[0]-1,np.shape(uvel)[1]-1))
    numerator = -1*np.diff(np.asarray(uvel[:,:]) * np.asarray(dyC), axis=0)[:, :-1] + np.diff(np.asarray(vvel[:,:]) * np.asarray(dxC), axis=1)[:-1, :]
    denominator = np.asarray(rAz[:-1, :-1])
    zeta = np.zeros_like(numerator)
    zeta[denominator != 0] = numerator[denominator != 0] / denominator[denominator != 0]
    field[:,:] = zeta / np.asarray(f[:-1, :-1])
    return field

def calculate_strain_rate(uvel,vvel,dxC,dyC,rAz,f):
    field_1 = np.zeros((np.shape(uvel)[0]-1,np.shape(uvel)[1]-1))
    numerator = np.diff(np.asarray(uvel[:,:]) * np.asarray(dxC), axis=0)[:, :-1] - np.diff(np.asarray(vvel[:,:]) * np.asarray(dyC), axis=1)[:-1, :]
    denominator = np.asarray(rAz[:-1, :-1])
    zeta = np.zeros_like(numerator)
    zeta[denominator != 0] = numerator[denominator != 0] / denominator[denominator != 0]
    field_1[:,:] = (zeta / np.asarray(f[:-1, :-1]))**2
    field_2 = np.zeros((np.shape(uvel)[0]-1,np.shape(uvel)[1]-1))
    numerator = np.diff(np.asarray(uvel[:,:]) * np.asarray(dyC), axis=0)[:, :-1] + np.diff(np.asarray(vvel[:,:]) * np.asarray(dxC), axis=1)[:-1, :]
    denominator = np.asarray(rAz[:-1, :-1])
    zeta = np.zeros_like(numerator)
    zeta[denominator != 0] = numerator[denominator != 0] / denominator[denominator != 0]
    field_2[:,:] = (zeta / np.asarray(f[:-1, :-1]))**2
    field_str = (field_1+field_2)**(0.5)
    return field_str

def calculate_frontogenesis_function(uvel,vvel,density,dxC,dyC,rAz):
    #dudx
    field_1 = np.zeros((np.shape(uvel)[0]-1,np.shape(uvel)[1]-1))
    numerator = np.diff(np.asarray(uvel[:,:]) * np.asarray(dxC), axis=0)[:, :-1]
    denominator = np.asarray(rAz[:-1, :-1])
    zeta_dudx = np.zeros_like(numerator)
    zeta_dudx[denominator != 0] = numerator[denominator != 0] / denominator[denominator != 0]
    dudx = zeta_dudx
    #d_rhodx
    field_1 = np.zeros((np.shape(density)[0]-1,np.shape(density)[1]-1))
    numerator = np.diff(np.asarray(density[:,:]) * np.asarray(dxC), axis=0)[:, :-1]
    denominator = np.asarray(rAz[:-1, :-1])
    zeta_d_rhodx = np.zeros_like(numerator)
    zeta_d_rhodx[denominator != 0] = numerator[denominator != 0] / denominator[denominator != 0]
    d_rhodx = zeta_d_rhodx
    
    term1 = dudx*(d_rhodx)**2
    #dvdx
    field_1 = np.zeros((np.shape(vvel)[0]-1,np.shape(vvel)[1]-1))
    numerator = np.diff(np.asarray(vvel[:,:]) * np.asarray(dxC), axis=0)[:, :-1]
    denominator = np.asarray(rAz[:-1, :-1])
    zeta_dvdx = np.zeros_like(numerator)
    zeta_dvdx[denominator != 0] = numerator[denominator != 0] / denominator[denominator != 0]
    dvdx=zeta_dvdx
    #d_rhody
    field_1 = np.zeros((np.shape(density)[0]-1,np.shape(density)[1]-1))
    numerator = np.diff(np.asarray(density[:,:]) * np.asarray(dyC), axis=1)[:-1, :]
    denominator = np.asarray(rAz[:-1, :-1])
    zeta_d_rhody = np.zeros_like(numerator)
    zeta_d_rhody[denominator != 0] = numerator[denominator != 0] / denominator[denominator != 0]
    d_rhody=zeta_d_rhody
    term2 = dvdx*d_rhody*d_rhodx
    #dudy
    field_1 = np.zeros((np.shape(uvel)[0]-1,np.shape(uvel)[1]-1))
    numerator = np.diff(np.asarray(uvel[:,:]) * np.asarray(dyC), axis=1)[:-1, :]
    denominator = np.asarray(rAz[:-1, :-1])
    zeta_dudy = np.zeros_like(numerator)
    zeta_dudy[denominator != 0] = numerator[denominator != 0] / denominator[denominator != 0]
    dudy=zeta_dudy
    #dvdy
    field_1 = np.zeros((np.shape(vvel)[0]-1,np.shape(vvel)[1]-1))
    numerator = np.diff(np.asarray(vvel[:,:]) * np.asarray(dyC), axis=1)[:-1, :]
    denominator = np.asarray(rAz[:-1, :-1])
    zeta_dvdy = np.zeros_like(numerator)
    zeta_dvdy[denominator != 0] = numerator[denominator != 0] / denominator[denominator != 0]
    dvdy=zeta_dvdy
    term3 = dudy*d_rhody*d_rhodx
    term4 = dvdy*(d_rhody)**2
    frontogenesis = -(term1+term2+term3+term4)
    return frontogenesis

def YMD_to_DecYr(year,month,day):
    start = datetime.datetime(int(year),1,1).timestamp()
    end = datetime.datetime(int(year)+1,1,1).timestamp()
    current =  datetime.datetime(int(year),int(month),int(day)).timestamp()
    current_diff = current-start
    percent = current_diff/(start-end)
    decY = int(year)-percent
    # define a date object using the datetime module
    return decY

def convert_mitgcm_grid_to_nc(path_to_grid_files,dim_of_grid,output_path='./grid.nc'):
    grid_prefix_1d = ['DRF','RC']
    grid_prefix_2d = ['Depth','DXV','DYU','RAZ','YC','DXC','DYC','XC','YG','DXG','DYG','RAC','XG']
    grid_prefix_3d = ['hFacC']
    if type(dim_of_grid)==tuple:
        dimx, dimy = dim_of_grid[0], dim_of_grid[1]
        try:
            dimz = dim_of_grid[2]
        except:
            print('no 3d shape')
    elif type(dim_of_grid)==str:
        dimx, dimy = dim_of_grid.split('_')[1].split('x')
        try:
            dimz = dim_of_grid[2]
        except:
    
            print('no 3d shape')

    #2D files
    for i in range(0,len(grid_prefix_2d)):
        file1 = path_to_grid_files+'/'+grid_prefix_2d[i]+'_'+str(dimx)+'x'+str(dimy)
        file_name1 = grid_prefix_2d[i]+'_'+str(dimx)+'x'+str(dimy)
        shape = (dimy,dimx) 
        i0 = np.arange(dimx)
        j0 = np.arange(dimy)
        dim = ['i','j']
        coord = [j0,i0]
        convert_binary_to_nc(file_name1,file1,shape,dim,coord,grid_prefix_2d[i],output_filepath= './.'+file_name1+'.nc')
    
    #3D files
    try:
        k0 = np.arange(dimz)
        #1D files
        for m in range(0,len(grid_prefix_1d)):
            file1 = path_to_grid_files+'/'+grid_prefix_1d[m]+'_'+str(dimz)
            file_name1 = grid_prefix_1d[i]+'_'+str(dimz)
            shape = (dimz) 
        
        
            dim = ['k']
            coord = [k0]
            convert_binary_to_nc(file_name1,file1,shape,dim,coord,grid_prefix_2d[i],output_filepath= './.'+file_name1+'.nc')
        
        dim = ['i','j','k']
        coords = [j0,i0,k0]
        
        for j in range(0,len(grid_prefix_3d)):
            file2 = path_to_grid_files+'/'+grid_prefix_3d[j]+'_'+str(dimx)+'x'+str(dimy)+'x'+str(dimz)
            file_name2 = grid_prefix_3d[j]+'_'+str(dimx)+'x'+str(dimy)+'x'+str(dimz)
            shape = (dimy,dimx,dimz)
            convert_binary_to_nc(file_name2,file2,shape,dim,coords,grid_prefix_3d[j],output_filepath= './.'+file_name2+'.nc')
    except:
        print('no 3d shape')
    one_d_files, one_d_filepaths= get_data_paths_from_binary('./','./',delim='_',file_end=str(dim_of_grid[2]))
    two_d_files, two_d_filepaths= get_data_paths_from_binary('./','./',delim='_',file_end=str(dim_of_grid[0])+'x'+str(dim_of_grid[1])+'.nc')
    three_d_files, three_d_filepaths = get_data_paths_from_binary('./','./',delim='_',file_end=str(dim_of_grid[0])+'x'+str(dim_of_grid[1])+'x'+str(dim_of_grid[2])+'.nc')
    total_file_paths = one_d_filepaths+two_d_filepaths+three_d_filepaths
    grid_full = xr.open_mfdataset(total_file_paths)
    grid_full.to_netcdf(output_path)
    grid_full.close()
    for files in total_file_paths:
                   os.system('rm '+ files)

def rot(u,v,theta):
    """
Rotate a vector counter-clockwise OR rotate the coordinate system clockwise.

Usage:
ur,vr = rot(u,v,theta)

Input:
u,v - vector components (e.g. u = eastward velocity, v = northward velocity)
theta - rotation angle (degrees)

Output:
ur,vr - rotated vector components

Example:
rot(1,0,90) returns (0,1)
    """

    # Make sure inputs are numpy arrays
    if type(u) is list:
        u = np.array(u)
        v = np.array(v)

    w = u + 1j*v            # complex vector
    ang = theta*np.pi/180   # convert angle to radians
    wr = w*np.exp(1j*ang)  # complex vector rotation
    ur = np.real(wr)        # return u and v components
    vr = np.imag(wr)
    return ur,vr
    
    