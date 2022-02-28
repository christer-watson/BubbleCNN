from astropy.io import fits
from astropy.utils.data import download_file
from astropy.wcs import WCS
import astropy
from astropy import coordinates as coord
from astropy import units as u
from downloadfits import *
from math import cos, sin, radians
import time
import os
from PIL import Image

#returned data has following format (copy & pasted from readme file):
#   22- 29  F8.4  deg     GLON      Galactic Longitude
#   31- 38  F8.4  deg     GLAT      Galactic Latitude
#   40- 44  F5.2  arcmin  Disp      [0.01/4.64] Dispersion of the central coordinates
#   46- 50  F5.2  arcmin  MajAxis   [0.11/19.25] Semi-major axis
#   52- 56  F5.2  arcmin  MinAxis   [0.08/14.29] Semi-minor axis
#   58- 62  F5.2  arcmin  Reff      [0.11/16.96] Effective Radius
#   64- 68  F5.2  arcmin  dReff     [0.01/3.14] Error in the Effective Radius
#   70- 72  I3    deg     theta     [0/180] Orientation Angle (from Galactic North towards increasing l)
data = readbubblesdat()
pixsize=1024
if pixsize == 1024:
	dir="1024/data/"
	arcsecsize=613.5
elif pixsize == 512:
	dir="512/data/"
	arcsecsize=306.

for i in range(1000,1200):
	print(i)
	longit=data[i][0]
	latit=data[i][1]
	a=data[i][3]/60.
	b=data[i][4]/60.
	angle=data[i][7]
	band=4
#fitsname="G"+str(longit)+"_IRAC"+str(band)+".fits"
	basename=format(i, '05d')+"B"+format(band,'1d')+".fits"
	fitsname="frame_"+basename
#name files by index/row in bubbles catalog
	if not(os.path.exists(dir+fitsname)):
		downloadfits(longit,latit,band=band,filename=fitsname,size=arcsecsize)
		os.rename(fitsname,dir+fitsname)

	hdu=fits.open(dir+fitsname)
	mask=hdu[0]
	if not(mask.data.shape == (pixsize,pixsize)):
		print(mask.data.shape)
		hdu.close()
		os.remove(dir+fitsname)
		print("not correct size, "+str(pixsize)+ ", removing file")
		continue
#if file is not correct size (pixsize x pixsize), then delete it and skip everything else


