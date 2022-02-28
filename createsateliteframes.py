import numpy as np
import os
import math
from astropy.io import fits
import shutil
from astropy.wcs import WCS
import astropy
from downloadfits import *
import reproject

pixsize=1024
band=8
dirlist =['irac1/', 'irac2/', 'irac3/', 'irac4/', '/home/cwatson/wise/', '/home/cwatson/mips/', '/home/cwatson/pacs/','thor/']
satelitedir = dirlist[band-1]

writingdir = "1024/data/"
satelitelist=[]
print(satelitedir)
for filename in os.listdir(satelitedir):
	hdu = fits.open(satelitedir+filename)
	firsthdu = hdu[0]
	wcs=WCS(firsthdu.header)
	centerxpix=math.floor(wcs.array_shape[1]/2)
	centerypix=math.floor(wcs.array_shape[0]/2)
	centerposition = astropy.wcs.utils.pixel_to_skycoord(centerxpix, centerypix, wcs=wcs)
	long=float(centerposition.galactic.to_string().split()[0])
	lat=float(centerposition.galactic.to_string().split()[1])
	satelitelist.append((filename,long,lat))
	hdu.close()

data = readbubblesdat()

for i in range(800,1200):
	basename=format(i, '05d')+"B"+format(band,'1d')+".fits"
	satelitecutoutfilename = "frame_"+basename
	b4fitsname = "frame_"+format(i, '05d')+"B"+format(4,'1d')+".fits"
	print(i)
	longit=data[i][0]
	latit=data[i][1]
	distancelist=[]
	for row in satelitelist:
		distance = np.sqrt((longit-row[1])**2+(latit-row[2])**2)
		distancelist.append(distance)

	minindex = np.argmin(distancelist)
	satelitefilename = satelitelist[minindex][0]
	hdu1 = fits.open(satelitedir+satelitefilename)
	satelitehdu = hdu1[0]
	satelitewcs=WCS(firsthdu.header)
	if os.path.exists(writingdir+b4fitsname):
		hdu2 = fits.open(writingdir+b4fitsname)
	else:
		continue
	readinghdu = hdu2[0]

	array, footprint = reproject.reproject_interp(satelitehdu, readinghdu.header)
	fits.writeto(writingdir+satelitecutoutfilename, array, readinghdu.header, overwrite=True)
	hdu1.close()
	hdu2.close()

#	position = astropy.coordinates.SkyCoord(longit,latit,unit="deg",frame='galactic')
#	cutout = astropy.nddata.Cutout2D(firsthdu.data, position, pixsize, wcs=wcs, mode='strict', copy=True)
#	firsthdu.data = cutout.data
#	firsthdu.header.update(cutout.wcs.to_header())
#	firsthdu.writeto(writingdir+wisefitsfilename, overwrite=True)
#	hdu.close()

