import numpy as np
import tensorflow as tf
from cnnmodel import *
from astropy.io import fits
from DiceLoss import *
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval
import os
import DataGenerator as dg
import fnmatch

class LearningRateReducerCb(tf.keras.callbacks.Callback):

  def on_epoch_end(self, epoch, logs={}):
    old_lr = self.model.optimizer.lr.read_value()
    new_lr = old_lr * .99
    print("\nEpoch: {}. Reducing Learning Rate from {} to {}".format(epoch, old_lr, new_lr))
    self.model.optimizer.lr.assign(new_lr)

#sourcelist = ["00801","00812", "00814","00816", "00820","00821",  "00822","00823", "00824", "00828",  "00866", "00888",]
#sourcelist = ["00801", "00821", "00822", "00823","00824", "00827", "00828", "00833", "00834", "00866", "00888"]
#sourcelist = ["00801", "00802", "00805","00812","00813","00814","00816","00818","00819","00820", "00821", "00822", "00823","00824","00826", "00827", "00828", "00833", "00834", "00866", "00888"]
sourcelist = ['00800', '00801', '00802', '00805', '00812', '00813', '00814', '00816', '00818', '00819', '00820', '00821', '00822', '00823', '00824', '00826', '00827', '00828', '00833', '00834', '00836', '00838', '00839', '00840', '00841', '00842', '00843', '00844', '00845', '00846', '00847', '00848', '00850', '00851', '00852', '00853', '00855', '00856', '00857', '00860', '00862', '00863', '00864', '00865', '00866', '00868', '00869', '00870', '00871', '00872', '00873', '00874', '00878', '00879', '00880', '00881', '00882', '00883', '00884', '00885', '00886', '00887', '00888', '00891', '00894', '00896', '00898', '00899']
#sourcelist = ['00800', '00801', '00802', '00805', '00812', '00813', '00814', '00816', '00818', '00819', '00820', '00821', '00822', '00823', '00824', '00826', '00827', '00828', '00833', '00834', '00836', '00838', '00839', '00840', '00841', '00842', '00843', '00844', '00845', '00846', '00847', '00848', '00850', '00851', '00852', '00853', '00855', '00856', '00857', '00860', '00862', '00863', '00864', '00865', '00866', '00868', '00869', '00870', '00871', '00872', '00873', '00874', '00878', '00879', '00880', '00881', '00882', '00883', '00884', '00885', '00886', '00887', '00888', '00891', '00894', '00896', '00898', '00899', '00900', '00901', '00902', '00903', '00904', '00905', '00907', '00916', '00917', '00918', '00919', '00920', '00921', '00922', '00923', '00924', '00925', '00926', '00927', '00936', '00937', '00938', '00941', '00943', '00944', '00945', '00946', '00947', '00952', '00953', '00954', '00955', '00956', '00957', '00958', '00962', '00963', '00964', '00965', '00966', '00969', '00970', '00971', '00972', '00973', '00982', '00983', '00985', '00986', '00987', '00988', '00989', '00995', '00996', '00997', '00998', '01000', '01001', '01002', '01003', '01004', '01005', '01006', '01007', '01008', '01009', '01010', '01013', '01014', '01017', '01018', '01019', '01020', '01022', '01023', '01027', '01028', '01029', '01030', '01031', '01033', '01038', '01039', '01040', '01041', '01042', '01043', '01044', '01045', '01046', '01047', '01058', '01059', '01060', '01061', '01062', '01063', '01069', '01073', '01074', '01076', '01077', '01078', '01079', '01080', '01081', '01082', '01083', '01084', '01086', '01089', '01090', '01098', '01101', '01102', '01104', '01106', '01108', '01110', '01111', '01115', '01117', '01118', '01120', '01122', '01130', '01131', '01134', '01135', '01137', '01138', '01140', '01145', '01146', '01147', '01148', '01149', '01153', '01154', '01159', '01160', '01161', '01162', '01163', '01164', '01165', '01166', '01167', '01170', '01174', '01175', '01176', '01177', '01178', '01179', '01180', '01181', '01182', '01183', '01184', '01185', '01186', '01187', '01188', '01189', '01190', '01191', '01192', '01193', '01194', '01195']

#filelist = fnmatch.filter(os.listdir('1024/data/'),"fram*B4.fits")
#sourcelist = []
#for filename in filelist:
#	sourcelist.append(filename[6:11])

#for sourcename in sourcelist:


print(sourcelist)
suffixlist=['B3','B4','B5','B6','B7','B8']
dirname="../1024/data/"
batch_size=5
sourcegenerator = dg.DataGenerator(dirname, sourcelist = sourcelist, batch_size=batch_size, prefix="frame_", seed=42, suffixlist = suffixlist, shift=0.1)

model = unetmodel()
adam = tf.keras.optimizers.Adam(learning_rate=0.001)
loss = soft_dice_loss
metrics = [tf.keras.metrics.BinaryAccuracy(), intersection, union, iou]
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='iou', mode='max', restore_best_weights = True, patience=100)
callbacks = [LearningRateReducerCb(), earlystopping]
epochs=500
model.compile(optimizer=adam,loss=loss, metrics = metrics)

#results = model.fit(sourcegenerator, epochs=epochs, callbacks=callbacks,steps_per_epoch=2)
results = model.fit(sourcegenerator, epochs=epochs, callbacks=callbacks)

#tf.keras.utils.plot_model(model,to_file='foo.png')
model.optimizer = None
model.compiled_loss = None
model.compiled_metrics = None
model.save('cnnmodeltemp')
plt.plot(results.history['iou'])
plt.savefig('ioutemp.png')

