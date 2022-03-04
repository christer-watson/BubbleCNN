import numpy as np
from astropy.io import fits
from astropy.visualization import ZScaleInterval
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sourcelist = ['00800', '00801', '00802', '00805', '00812', '00813', '00814', '00816', '00818', '00819', '00820', '00821', '00822', '00823', '00824', '00826', '00827', '00828', '00833', '00834', '00836', '00838', '00839', '00840', '00841', '00842', '00843', '00844', '00845', '00846', '00847', '00848', '00850', '00851', '00852', '00853', '00855', '00856', '00857', '00860', '00862', '00863', '00864', '00865', '00866', '00868', '00869', '00870', '00871', '00872', '00873', '00874', '00878', '00879', '00880', '00881', '00882', '00883', '00884', '00885', '00886', '00887', '00888', '00891', '00894', '00896', '00898', '00899', '00900', '00901', '00902', '00903', '00904', '00905', '00907', '00916', '00917', '00918', '00919', '00920', '00921', '00922', '00923', '00924', '00925', '00926', '00927', '00936', '00937', '00938', '00941', '00943', '00944', '00945', '00946', '00947', '00952', '00953', '00954', '00955', '00956', '00957', '00958', '00962', '00963', '00964', '00965', '00966', '00969', '00970', '00971', '00972', '00973', '00982', '00983', '00985', '00986', '00987', '00988', '00989', '00995', '00996', '00997', '00998', '01000', '01001', '01002', '01003', '01004', '01005', '01006', '01007', '01008', '01009', '01010', '01013', '01014', '01017', '01018', '01019', '01020', '01022', '01023', '01027', '01028', '01029', '01030', '01031', '01033', '01038', '01039', '01040', '01041', '01042', '01043', '01044', '01045', '01046', '01047', '01058', '01059', '01060', '01061', '01062', '01063', '01069', '01073', '01074', '01076', '01077', '01078', '01079', '01080', '01081', '01082', '01083', '01084', '01086', '01089', '01090', '01098', '01101', '01102', '01104', '01106', '01108', '01110', '01111', '01115', '01117', '01118', '01120', '01122', '01130', '01131', '01134', '01135', '01137', '01138', '01140', '01145', '01146', '01147', '01148', '01149', '01153', '01154', '01159', '01160', '01161', '01162', '01163', '01164', '01165', '01166', '01167', '01170', '01174', '01175', '01176', '01177', '01178', '01179', '01180', '01181', '01182', '01183', '01184', '01185', '01186', '01187', '01188', '01189', '01190', '01191', '01192', '01193', '01194', '01195']

#sourcelist = ["00801","00812", "00814","00816", "00820","00821",  "00822","00823", "00824", "00828",  "00866", "00888",]

filenamelist=[]
suffixlist=['B3','B4','B5','B6','B7','B8']
dirname="../1024/data/"
zdata=[]
for sourcename in sourcelist:
	filenamelist.append(dirname+"frame_"+sourcename)

zscale = ZScaleInterval(contrast=0.05)
loadbool = False
if loadbool:
	for i, filename in enumerate(filenamelist):
		print(filename)
# Determine shift, if necessary, using random number. Will use the same shift on all channels and mask, of course
		for chan, suffix in enumerate(suffixlist):
			hdu=fits.open(filename+suffix+".fits")
			data = hdu[0].data
			(zmin, zmax) = zscale.get_limits(data)
			hdu.close()
			zdata.append((filename,suffix,'zmin',zmin))
			zdata.append((filename,suffix,'zmax',zmax))
	df = pd.DataFrame(data=zdata,columns=['source','band','zbound','zvalue'])
	df.to_csv("zbounds.csv")
else:
	df=pd.read_csv("zbounds.csv")

pdf_pages = PdfPages('zbounds.pdf')
for band in suffixlist:
	fig, axs = plt.subplots(1,2)
	sns.histplot(data=df[df['band']==band],x='zvalue',hue='zbound',ax=axs[0]).set(title=band)
	sns.histplot(data=df[df['band']==band],x='zvalue',hue='zbound',ax=axs[1],log_scale=True).set(title=band)
	plt.tight_layout()
	pdf_pages.savefig()
	plt.close()
	
pdf_pages.close()
