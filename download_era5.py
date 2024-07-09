"""
Download data from ERA5 reanalysis dataset

API guide: https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels-monthly-means?tab=form

Notes:
- I usually run this in Colab to save files to Google Drive easily
- You can use a website like http://bboxfinder.com to get the coordinates of the desired area,
    but you need to pay attention to the ERA 5 API correct order: [North, West, South, East ]

# !pip install cdsapi --quiet

# !pip install cdstoolbox-remote

- to be able to read grib files:
# !pip install cfgrib --quiet
# !pip install eccodes --quiet
# !sudo apt-get install libeccodes0  --quiet
!pip install netCDF4 scipy
!pip install h5netcdf

"""
#%%
import os
import cdsapi
import cfgrib
import xarray as xr
import numpy as np
from time import sleep

API_KEY =  "202370:f787b403-6f87-40ee-b128-22625fee3ec7"
cds = cdsapi.Client(url="https://cds.climate.copernicus.eu/api/v2",key=API_KEY)

def download(area_download,years_to_download,times_to_get,variables):

    #download blocks
    block_size = 6
    block_start, block_end = 0, block_size

    datasets = []
    while block_start < len(years_to_download):

        cur_years = years_to_download[block_start:block_end]
        start_y, end_y = cur_years[0],cur_years[-1]

        print('Downloading',cur_years)

        dataset_name =f"era_{start_y}_{end_y}"
        dataset_name =f'data/raw/{dataset_name}.grib'

        datasets.append(dataset_name)
        
        data = cds.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'variable': variables,

                'year': cur_years,

                'month': [
                    '01', '02', '03','04', '05', '06',
                    '07', '08', '09','10', '11', '12'],

                'day': [
                    '01', '02', '03','04', '05', '06',
                    '07', '08', '09','10', '11', '12',
                    '13', '14', '15','16', '17', '18',
                    '19', '20', '21','22', '23', '24',
                    '25', '26', '27','28', '29', '30','31'],

                'time': times_to_get,

                'area': [2.46, -71.97, -33.26,-36.64], #Brazil box http://bboxfinder.com/#-33.268014,-71.975849,2.469466,-36.648545

                'format': 'grib',

            },dataset_name)

        block_start = block_end
        block_end += block_size

        print('-- finished --')

        sleep(10)


def concat_files():
    
    #tp timesteps are diff than t2m and msl

    datasets = [i for i in os.listdir('data/') if i[-5:] == '.grib']

    ds_list = []

    for file in sorted(datasets):

        if file:
        
            ds = cfgrib.open_dataset( 'data/raw/' + file)
            ds_list.append(ds)

            ds.close()  

    ds = xr.concat(ds_list, dim='time')
    output_file = 'data/msl_t2m_1983_2023.nc'
    ds.to_netcdf(output_file)


def save_values(file):
    #just to save process

    data = cfgrib.open_dataset('data/raw/' + file)
    df = data.t2m.values

    q99 = np.quantile(df,0.99)
    q10 = np.quantile(df,0.1)

    norm_imgs = np.clip(df, q10, q99)
    norm_imgs = (norm_imgs - q10)/(q99 -q10)

    time, lat, long = norm_imgs.shape

    wind = 8 #find best after

    norm_imgs = norm_imgs.reshape(int(time/wind), wind, lat, long)
    norm_imgs = np.transpose(norm_imgs, (0, 2, 3, 1))

    np.save('data/test',norm_imgs,allow_pickle=False)


###################

area_download = [2.46, -71.97, -33.26,-36.64] #brazil area

variables = [
    '2m_temperature',
    'total_precipitation',
    'mean_sea_level_pressure'
]

years_to_download = [str(i) for i in range(1995,2000)]

times_available = [ '00:00', '01:00', '02:00',
                  '03:00', '04:00', '05:00',
                  '06:00', '07:00', '08:00',
                  '09:00', '10:00', '11:00',
                  '12:00', '13:00', '14:00',
                  '15:00', '16:00', '17:00',
                  '18:00', '19:00', '20:00',
                  '21:00', '22:00', '23:00',]

times_to_get = times_available[::3]

# download(area_download,years_to_download,times_to_get,variables)
