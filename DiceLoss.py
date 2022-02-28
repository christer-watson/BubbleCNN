from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
from astropy.visualization import ZScaleInterval

def DiceLoss(targets, inputs, smooth=1e-6):
    
	intersection = K.sum(targets*inputs)
	dice = (2*intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
	return 1 - dice
 
def dice_metric(inputs, target):
	intersection = 2.0 * K.sum(target * inputs)
	union = K.sum(target) + K.sum(inputs)
	if target.sum() == 0 and inputs.sum() == 0:
		return 1.0

	return intersection / union

def soft_dice_loss(y_true, y_pred, epsilon=1e-6): 
    ''' 
    Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
    Assumes the `channels_last` format.
  
    # Arguments
        y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
        y_pred: b x X x Y( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax) 
        epsilon: Used for numerical stability to avoid divide by zero errors
    
    # References
        V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation 
        https://arxiv.org/abs/1606.04797
        More details on Dice loss formulation 
        https://mediatum.ub.tum.de/doc/1395260/1395260.pdf (page 72)
        
        Adapted from https://github.com/Lasagne/Recipes/issues/99#issuecomment-347775022
    '''
    
    # skip the batch and class axis for calculating Dice score
    axes = tuple(range(1, len(y_pred.shape)-1)) 
    numerator = 2. * K.sum(y_pred * y_true, axes)
    denominator = K.sum(K.square(y_pred) + K.square(y_true), axes)
    
    return 1 - K.mean((numerator + epsilon) / (denominator + epsilon)) # average over classes and batch
    # thanks @mfernezir for catching a bug in an earlier version of this implementation!
#def myMeanIoU(inputs, target):
#	inputsbinary = (inputs>0.5)
#	targetbinary = K.cast(target,tf.bool)
#	intersection = K.sum(K.cast(K.all(K.stack([inputsbinary, targetbinary],axis=0),axis=0),tf.float32))
#	union = K.sum(K.cast(K.any(K.stack([inputsbinary, targetbinary], axis=0),axis=0),tf.float32))
#	return(intersection/union)

#class myMetricMeanIoU(tf.keras.metrics.Metric):

#	def __init__(self, name="mymetricmeaniou",**kwargs):
#		super(myMetricMeanIoU, self).__init__(name=name,**kwargs)
#		self.intersection = self.add_weight(name="intersection", initializer = 'zeros')
#		self.union = self.add_weight(name="union", initializer = 'zeros')
#		self.IoU = self.add_weight(name="IoU", initializer = 'zeros')
#		self.IoU = 0.

#	def update_state(self, inputs, target, sample_weight = None):
#		inputs10 = K.cast(inputs>0.8,tf.float32)
#		self.intersection.assign(K.sum(inputs10*target))
#		self.union.assign(K.sum(inputs10+target)-self.intersection)
#		self.IoU.assign(self.intersection/self.union)

#	def result(self):
#		return self.IoU

#	def reset_state(self):
#		self.IoU.assign(0.)
#		self.IoU = 0.

class MyMeanIOU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight)


#class UpdatedMeanIoU(tf.keras.metrics.MeanIoU):
#    @tf.function
#    def __call__(self, y_true, y_pred, sample_weight=None):
#        y_pred = tf.argmax(y_pred, axis=-1) # this is the fix
#        return super().__call__(y_true, y_pred, sample_weight=sample_weight)

#class UpdatedMeanIoU(tf.keras.metrics.MeanIoU):
#	def __init__(self,
#               y_true=None,
#               y_pred=None,
#               num_classes=None,
#               name=None,
#               dtype=None):
#		super(UpdatedMeanIoU, self).__init__(num_classes = num_classes,name=name, dtype=dtype)

#	def update_state(self, y_true, y_pred, sample_weight=None):
#		y_pred = tf.math.argmax(y_pred, axis=-1)
#		return super().update_state(y_true, y_pred, sample_weight)

def iou(y_true, y_pred):
	def f(y_true, y_pred):
		axes = tuple(range(1, len(y_pred.shape)))
#		axes = tuple(range(1, len(y_pred.shape)-1))
		y_pred_binary = (y_pred>0.6).astype(np.float32)
		intersection = (y_true * y_pred_binary).sum(axis=axes).astype(np.float32)
		union = y_true.sum(axis=axes) + y_pred_binary.sum(axis=axes) - intersection
		x = np.mean((intersection + 1e-15) / (union + 1e-15))
#		x = intersection / union
		x = x.astype(np.float32)
		return x
#        return (y_pred>0.6).astype(np.float32).sum()
	return tf.numpy_function(f, [y_true, y_pred], tf.float32)

def intersection(y_true, y_pred):
	def f1(y_true, y_pred):
		axes = tuple(range(1, len(y_pred.shape)))
#		axes = tuple(range(1, len(y_pred.shape)-1))
		y_pred_binary = (y_pred>0.6).astype(np.float32)
		intersection = (y_true * y_pred_binary).sum().astype(np.float32)
		return intersection
	return tf.numpy_function(f1, [y_true, y_pred], tf.float32)

def union(y_true, y_pred):
	def f2(y_true, y_pred):
		y_pred_binary = (y_pred>0.6).astype(np.float32)
		intersection = (y_true * y_pred_binary).sum()
		union = (y_true.sum() + y_pred_binary.sum() - intersection).astype(np.float32)
		return union
	return tf.numpy_function(f2, [y_true, y_pred], tf.float32)

def normalize(data,mode='zscale'):
#assume data is 2D (e.g. 512x512)
	if mode == 'zscale':
		zscale = ZScaleInterval(contrast=0.05)
		(zmin, zmax) = zscale.get_limits(data)
		np.clip(data, zmin, zmax, out=data)
		np.nan_to_num(data, copy=False, nan=zmax)
		data = (data-zmin)/(zmax-zmin)
	return(data)
