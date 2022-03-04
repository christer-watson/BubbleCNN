from astroquery.cadc import Cadc
import numpy as np
import os
import math
from astropy.io import fits
import shutil
from astropy.wcs import WCS
import astropy
from downloadfits import *
import reproject


cadc = Cadc()
coords = '01h45m07.5s +23d18m00s'
radius = '0.1 degree'
results = cadc.query_region(coords, radius, collection='VLASS')
image_list = cadc.get_image_list(results, coords, '0.1 degree')
pixsize=1024
band=9
writingdir = "1024/data/"
data = readbubblesdat()
