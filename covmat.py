import numpy as np
import xarray as xr

all_points = xr.load_dataarray(r'/home/labs/leeat/vovam/Vova/data/all_data.nc')
needed_channels = ['CD20', 'CD4', 'CD8', 'HLA-DR', 'dsDNA', 'Pan-Keratin', 'CD45', 'CD3']

channels_data = all_points.sel(channels=needed_channels)
channels_mat = np.transpose(channels_data.values, [1, 0, 2, 3])
# channels_mat.shape == (8, 38, 2048, 2048)
print(channels_mat.shape)

channels_mat = channels_mat.reshape((8, -1))
# channels_mat.shape == (8, 38*2048*2048)
print(channels_mat.shape)

channels_mat = channels_mat.clip(min=0)
covariance_mat = np.cov(np.rint(channels_mat))
print(covariance_mat.shape)

np.save(r'/home/labs/leeat/vovam/cov.npy', covariance_mat)
