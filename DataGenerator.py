#http://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
import numpy as np
import tensorflow as tf
import os
from astropy.io import fits
import random
import scipy.ndimage
import math
from astropy.visualization import ZScaleInterval

class DataGenerator(tf.keras.utils.Sequence):
#    'Generates data for Keras'
	def __init__(self, dirname, sourcelist=[],batch_size=8, dim=(1024,1024), shuffle=True, 
			prefix="frame_", validation_split=None, subset="training", 
			seed=42, normalize=True, suffixlist=None, shift=None):

		self.dirname = dirname
		self.dim = dim
		self.batch_size = batch_size
		self.n_channels = len(suffixlist)
		self.shuffle = shuffle
		self.shift = shift #shift should be [-1,+1], that is a fraction of the image width & height
		np.random.seed(seed)
		self.normalize = normalize
		self.suffixlist = suffixlist
		self.sourcelist = sourcelist
		self.prefix = prefix
#		tempfilelist = sorted([f for f in os.listdir(self.dirname) if (f.startswith(prefix) and f.endswith(suffixlist[0]+".fits"))])
		tempfilelist=[]
		for sourcename in sorted(sourcelist):
			tempfilelist.append(dirname+prefix+sourcename+suffixlist[0]+".fits")
		self.basefilelist = [f.split(suffixlist[0])[0] for f in tempfilelist]
		print("Found "+str(len(self.basefilelist))+" files.")
		if validation_split:
			numfiles =len(self.basefilelist)
			valnum=np.floor(validation_split*numfiles).astype(int)
			trainnum = numfiles - valnum
			self.trainfilelist = self.basefilelist[0:trainnum]
			self.validationfilelist = self.basefilelist[trainnum:]
			if subset == "training": self.filelist = self.trainfilelist
			if subset == "validation": self.filelist = self.validationfilelist
			print("Found "+str(len(self.trainfilelist))+" training files.")
			print("Found "+str(len(self.validationfilelist))+" validation files.")
		else:
			self.filelist = self.basefilelist
		self.on_epoch_end()

	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.floor(len(self.filelist) / self.batch_size))

	def __getitem__(self, index):
		'Generate one batch of data'
        # Generate indexes of the batch
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of filenames
		filelist_temp = [self.filelist[k] for k in indexes]
        # Generate data
		X, y = self.__data_generation(filelist_temp)

		return X, y

	def on_epoch_end(self):
		'Updates indexes after each epoch'
		self.indexes = np.arange(len(self.filelist))
		if self.shuffle == True:
			np.random.shuffle(self.indexes)

	def __data_generation(self, filelist_temp):
		'Generates data containing batch_size samples' # X : (n_samples, dim, n_channels)
        # Initialization
		X = np.empty((self.batch_size, *self.dim, self.n_channels))
		y = np.empty((self.batch_size, *self.dim, 1), dtype=np.float32)

        # Generate data
		zscale = ZScaleInterval(contrast=0.05)
		for i, filename in enumerate(filelist_temp):
# Determine shift, if necessary, using random number. Will use the same shift on all channels and mask, of course
			for chan, suffix in enumerate(self.suffixlist):
				hdu=fits.open(filename+suffix+".fits")
				if self.normalize:
#					data = hdu[0].data/np.nanmax(hdu[0].data)
					data = hdu[0].data
					(zmin, zmax) = zscale.get_limits(data)
					np.clip(data, zmin, zmax, out=data)
					np.nan_to_num(data, copy=False, nan=zmax)
					data[data == zmax] = -1
					data = (data-zmin)/(zmax-zmin)
				else:
					data = hdu[0].data
				X[i,:,:,chan] = data
				hdu.close()

            # Store mask
			sourcename=filename.split("_")[1]
			maskname="mask_"+sourcename+"B4.fits"
			hdu=fits.open(self.dirname+maskname)
			y[i,] = hdu[0].data.reshape(*self.dim,1)
			hdu.close()

			if self.shift != None:
				pixelshift = math.floor(self.shift*1024)
				randompixelshift = random.randint(-pixelshift, pixelshift)
				X = scipy.ndimage.shift(X, (0, randompixelshift, randompixelshift, 0), order=0)
				y = scipy.ndimage.shift(y, (0, randompixelshift, randompixelshift, 0), order=0)

		return X, y
