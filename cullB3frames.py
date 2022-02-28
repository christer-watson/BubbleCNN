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
data = readbubblesdat()
dir="1024/data/"

for i in range(900,1200):
#	print(i)
	longit=data[i][0]
	latit=data[i][1]
	a=data[i][3]/60.
	b=data[i][4]/60.
	angle=data[i][7]
#fitsname="G"+str(longit)+"_IRAC"+str(band)+".fits"
	baseb3name=format(i, '05d')+"B"+format(3,'1d')+".fits"
	baseb4name=format(i, '05d')+"B"+format(4,'1d')+".fits"
	baseb5name=format(i, '05d')+"B"+format(5,'1d')+".fits"
	baseb6name=format(i, '05d')+"B"+format(6,'1d')+".fits"
	baseb7name=format(i, '05d')+"B"+format(7,'1d')+".fits"
	if not(os.path.exists(dir+"frame_"+baseb3name)):
		print("Culling bubble "+baseb3name)
		if os.path.exists(dir+"frame_"+baseb4name): os.remove(dir+"frame_"+baseb4name)
		if os.path.exists(dir+"frame_"+baseb5name): os.remove(dir+"frame_"+baseb5name)
		if os.path.exists(dir+"frame_"+baseb6name): os.remove(dir+"frame_"+baseb6name)
		if os.path.exists(dir+"frame_"+baseb7name): os.remove(dir+"frame_"+baseb7name)


