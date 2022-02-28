from astropy.io import fits
#from astropy.nddata import Cutout2D
from astropy.utils.data import download_file
from astropy.wcs import WCS
import astropy
#from astroquery import sha
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

for i in range(900,1200):
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
		continue
#		downloadfits(longit,latit,band=band,filename=fitsname,size=arcsecsize)
#		os.rename(fitsname,dir+fitsname)
#if file is not present, then skip everything below. The commented code would download it and move it appropriately. I have made sure to have only the 1024x1024 files in another project and I don't want to pollute my directory with oddly shaped fits files.

	hdu=fits.open(dir+fitsname)
	mask=hdu[0]
	if not(mask.data.shape == (pixsize,pixsize)):
		hdu.close()
		os.remove(dir+fitsname)
		continue
#if file is not 512x512, then delete it and skip everything else

#equation for ellipse rotated theta ccw from positive x-axis:
#((x - x0) cos theta + (y - y0) sin theta)^2 / a^2 + ((x - x0) sin theta - (y - y0) cos theta )^2 / b^2 = 1
#center: x0, y0
#semiaxis a & b (a is the "unrotated" x-direction semi-axis, b is for y)

	position = astropy.coordinates.SkyCoord(longit,latit,unit="deg",frame='galactic')
	wcs=WCS(mask.header)
	numx=wcs.array_shape[1]
	numy=wcs.array_shape[0]
	xpixcoord=astropy.wcs.utils.pixel_to_skycoord(range(numx),0,wcs)
	ypixcoord=astropy.wcs.utils.pixel_to_skycoord(0,range(numy),wcs)
	xpixcoordlist=[]
	ypixcoordlist=[]
	for skycoordx in xpixcoord:
		xpixcoordlist.append(float(str(skycoordx.to_string()).split()[0]))

	for skycoordy in ypixcoord:
		ypixcoordlist.append(float(str(skycoordy.to_string()).split()[1]))
	mask.data.fill(False)
	closebubbles=[]
	for row in data:
		if abs(longit-row[0]) < 0.07:
			if abs(latit - row[1]) < 0.07:
				closebubbles.append(row)

	for px in range(mask.data.shape[1]):
		for py in range(mask.data.shape[0]):
#		pixcoord = astropy.coordinates.SkyCoord.from_pixel(px,py,wcs)
#the above is dead slow. instead, I do something more efficient by creating an array of coordinates that index match.
			pixlongit = xpixcoordlist[px]
			pixlatit = ypixcoordlist[py]
#		maskvalue = ((pixlongit - longit)*cos(radians(angle))+(pixlatit-latit)*sin(radians(angle)))**2./a**2.+((pixlongit - longit)*sin(radians(angle))-(pixlatit-latit)*cos(radians(angle)))**2./b**2. < 1.
			for row in closebubbles:
#			rowposition=astropy.coordinates.SkyCoord(row[0],row[1],unit="deg",frame='galactic')
#			if wcs.footprint_contains(rowposition):
#weirdly, SkyCoord and footprint_contains require more time than the calc below. Ugh. 
				rowl=row[0]
				rowb=row[1]
				rowang=row[7]
				rowxrad=row[3]/60.
				rowyrad=row[4]/60.
				partA = ((pixlongit - rowl)*cos(radians(rowang)) + (pixlatit - rowb)*sin(radians(rowang)))**2./rowxrad**2.
				partB = ((pixlatit  - rowb) *cos(radians(rowang)) - (pixlongit-rowl)*sin(radians(rowang)))**2./rowyrad**2.
				maskvalue = partA+partB < 1.
#		maskvalue = ((pixlongit - longit)*cos(radians(angle))+(pixlatit-latit)*sin(radians(angle))**2./a**2.+((pixlatit-latit)*cos(radians(angle))-(pixlongit-longit)*sin(radians(angle)))**2./b**2. < 1.
				mask.data[py,px] = mask.data[py,px] or maskvalue 
#since each pixel is tested several times, we don't want to overwrite a True with a False
# Notice that the array is in y,x order... because. I obviously didn't know or even think about this until I noticed something weird.

	maskfilename="mask_"+basename
	mask.writeto(maskfilename,overwrite=True)
	hdu.close()
	os.rename(maskfilename,dir+maskfilename)

