import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
#import tensorflow_addons as tfa
import os
from cnnmodel import *
import math
import pandas as pd
import seaborn as sns
from astropy.io import fits
from DiceLoss import *
import shutil
import tarfile
import matplotlib.pyplot as plt
import fnmatch
from matplotlib.backends.backend_pdf import PdfPages

model=tf.keras.models.load_model('022822model')

pixscale=1024
dirname=str(pixscale)+"/data/"
suffixlist = ["B3", "B4", "B5", "B6", "B7", "B8"]
prefix="frame_"
#sourcelist = ["00801","00812", "00814","00816", "00820","00916"]
#sourcelist = ["00801","00812", "00814","00816", "00820","00821",  "00822","00823", "00824", "00828",  "00866", "00888",]
#sourcelist = ["00801", "00802", "00805","00812","00813","00814","00816","00818","00819","00820", "00821", "00822", "00823","00824","00826", "00827", "00828", "00833", "00834", "00866", "00888"]
sourcelist = ['00800', '00801', '00802', '00805', '00812', '00813', '00814', '00816', '00818', '00819', '00820', '00821', '00822', '00823', '00824', '00826', '00827', '00828', '00833', '00834', '00836', '00838', '00839', '00840', '00841', '00842', '00843', '00844', '00845', '00846', '00847', '00848', '00850', '00851', '00852', '00853', '00855', '00856', '00857', '00860', '00862', '00863', '00864', '00865', '00866', '00868', '00869', '00870', '00871', '00872', '00873', '00874', '00878', '00879', '00880', '00881', '00882', '00883', '00884', '00885', '00886', '00887', '00888', '00891', '00894', '00896', '00898', '00899', '00900', '00901', '00902', '00903', '00904', '00905', '00907', '00916', '00917', '00918', '00919', '00920', '00921', '00922', '00923', '00924', '00925', '00926', '00927', '00936', '00937', '00938', '00941', '00943', '00944', '00945', '00946', '00947', '00952', '00953', '00954', '00955', '00956', '00957', '00958', '00962', '00963', '00964', '00965', '00966', '00969', '00970', '00971', '00972', '00973', '00982', '00983', '00985', '00986', '00987', '00988', '00989', '00995', '00996', '00997', '00998', '01000', '01001', '01002', '01003', '01004', '01005', '01006', '01007', '01008', '01009', '01010', '01013', '01014', '01017', '01018', '01019', '01020', '01022', '01023', '01027', '01028', '01029', '01030', '01031', '01033', '01038', '01039', '01040', '01041', '01042', '01043', '01044', '01045', '01046', '01047', '01058', '01059', '01060', '01061', '01062', '01063', '01069', '01073', '01074', '01076', '01077', '01078', '01079', '01080', '01081', '01082', '01083', '01084', '01086', '01089', '01090', '01098', '01101', '01102', '01104', '01106', '01108', '01110', '01111', '01115', '01117', '01118', '01120', '01122', '01130', '01131', '01134', '01135', '01137', '01138', '01140', '01145', '01146', '01147', '01148', '01149', '01153', '01154', '01159', '01160', '01161', '01162', '01163', '01164', '01165', '01166', '01167', '01170', '01174', '01175', '01176', '01177', '01178', '01179', '01180', '01181', '01182', '01183', '01184', '01185', '01186', '01187', '01188', '01189', '01190', '01191', '01192', '01193', '01194', '01195']

X = np.empty((1,pixscale,pixscale,len(suffixlist)), dtype=np.float32)

#filelist = fnmatch.filter(os.listdir('1024/data/'),"fram*B4.fits")
#sourcelist = []
#for filename in filelist:
#	sourcelist.append(filename[6:11])
pdf_pages = PdfPages('022822allbubbles.pdf')


for sourcename in sourcelist:
	print(sourcename)
	for chan, suffix in enumerate(suffixlist):
		hdu = fits.open(dirname+prefix+sourcename+suffix+".fits")
		X[0,:,:,chan] = normalize(hdu[0].data)
		hdu.close()

	hdu=fits.open(dirname+"mask_"+sourcename+"B4.fits")
	maskdata = hdu[0].data
	hdu.close()


	predictfilename = dirname+"predict_"+sourcename+".fits"
	predictbinaryfilename = dirname+"predictbinary_"+sourcename+".fits"

	ypredict = model.predict(X)
	hdu = fits.open(dirname+prefix+sourcename+suffix+".fits")
	hdu[0].data = ypredict.reshape(pixscale,pixscale)
	hdu.writeto(predictfilename, overwrite = True)

	ypredictbinary = (ypredict > 0.6).astype(float)
	hdu[0].data = ypredictbinary.reshape(pixscale,pixscale)
	hdu.writeto(predictbinaryfilename, overwrite = True)
	hdu.close()
	zscale = ZScaleInterval(contrast=0.05)

	fig, axs = plt.subplots(3,3)
	axs[0,0].imshow(X[0,:,:,0])
	axs[0,0].set_title(sourcename+": 5.8 um")
	axs[0,1].imshow(X[0,:,:,1])
	axs[0,1].set_title("8.0 um")
	axs[0,2].imshow(X[0,:,:,2])
	axs[0,2].set_title("12 um")
	axs[1,0].imshow(X[0,:,:,3])
	axs[1,0].set_title("24 um")
	axs[1,1].imshow(X[0,:,:,4])
	axs[1,1].set_title("70 um")
	axs[1,2].imshow(X[0,:,:,5])
	axs[1,2].set_title("1060 MHz")
	axs[2,0].imshow(ypredict.squeeze())
	axs[2,0].set_title("Prob prediction")
	axs[2,1].imshow(ypredictbinary.squeeze())
	axs[2,1].set_title("Binary prediction")
	axs[2,2].imshow(maskdata)
	axs[2,2].set_title("Bubble Catalog")
#	fig.savefig(sourcename+".png")
	plt.tight_layout()
	for ax in fig.get_axes():
		ax.axis('off')
	
#	fig.savefig(sourcename+".png")
#	plt.close(fig)
	pdf_pages.savefig(fig)
	plt.close(fig)
	
pdf_pages.close()

#tarfilename = "trainpredict.tgz"
#if os.path.exists(tarfilename):
#	os.remove(tarfilename)

#tar = tarfile.open(tarfilename,"w:gz")
#for name in [f for f in os.listdir(dirname) if f.endswith(".fits")]:
#	tar.add(dirname+name)

#tar.close()
