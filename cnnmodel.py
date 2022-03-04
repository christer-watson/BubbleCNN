import tensorflow as tf

#does a basic pool, conv, conv for the unet model

def convmodel(inlayer, commonlayer1, commonlayer3, commonlayer7, pool_size=1):
	layerX1 = tf.keras.layers.AveragePooling2D(pool_size = pool_size)(inlayer)
	layerX2 = commonlayer1(layerX1)
	layerX3 = tf.keras.layers.MaxPooling2D(pool_size=(4, 4))(layerX2)
	layerX4 = commonlayer3(layerX3)
	layerX5 = tf.keras.layers.MaxPooling2D(pool_size=(4, 4))(layerX4)
	layerX6 = tf.keras.layers.UpSampling2D(size = (4,4))(layerX5)
	layerX7 = tf.keras.layers.concatenate([layerX6, layerX4])
	layerX8 = commonlayer7(layerX7)
	layerX9 = tf.keras.layers.UpSampling2D(size=(4,4))(layerX8)
	layerX10 = tf.keras.layers.concatenate([layerX9, layerX2])
	layerX11 = tf.keras.layers.UpSampling2D(size=pool_size)(layerX10)
	return(layerX11)

def unetmodel(pretrained_weights = None,input_size = (1024,1024,6), initfilters=16):
	inputs = tf.keras.layers.Input(shape=input_size)

	commonlayer1 = tf.keras.layers.Conv2D(initfilters, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='commonlayer1')
	commonlayer3 = tf.keras.layers.Conv2D(initfilters, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='commonlayer3')
	commonlayer7 = tf.keras.layers.Conv2D(initfilters/2, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='commonlayer7')
	
	layerA = convmodel(inputs, commonlayer1, commonlayer3, commonlayer7, pool_size=1)
	layerB = convmodel(inputs, commonlayer1, commonlayer3, commonlayer7, pool_size=2)
	layerC = convmodel(inputs, commonlayer1, commonlayer3, commonlayer7, pool_size=4)
	layerD = convmodel(inputs, commonlayer1, commonlayer3, commonlayer7, pool_size=8)
	layerE = convmodel(inputs, commonlayer1, commonlayer3, commonlayer7, pool_size=16)
	layerF = convmodel(inputs, commonlayer1, commonlayer3, commonlayer7, pool_size=32)
	layerG = convmodel(inputs, commonlayer1, commonlayer3, commonlayer7, pool_size=64)
	
	
	layer10 = tf.keras.layers.concatenate([layerA, layerB, layerC, layerD, layerE, layerF, layerG])
	layer11 = tf.keras.layers.Conv2D(1,1,activation = 'sigmoid')(layer10)
	model = tf.keras.models.Model(inputs = inputs, outputs = layer11)
    
#	model.summary()

	if(pretrained_weights):
		model.load_weights(pretrained_weights)

	return model

