# BubbleCNN
BubbleCNN is a tool to use a convolutional neural network to recognize wind-blown bubbles in astronomical images. In broad strokes, a convolutional neural network is a tool for recognizing specific features, from simple features like horizontal lines to complex features like human faces. This project uses tensorflow's standard convolutional neural network tools to build a model that is tailored toward astronomical data. The modifications made, mostly to accomodate properties common in astronomical data but uncommon in other data:
- Data are high signal to noise, often S/N > 3000, which is higher than supported by stanard 8-bit PNG data format
- 6 data channels ('colors' to non-astronomers) are supported, allowing features from IR to radio light to be used for recognizing wind-blown-bubbles
- A catalog of 1000s of wind-blown bubbles, each represented by seven 1024x1024 image requires loading data in batches. A data generator customized was required to handle the high S/N data files and 6 data channels
- Typical wind-blown bubbles occupy a small area of the data (< 2%), so an intersection/union-style metric, Dice, was used.
- Becauses wind-blown bubbles can be a wide range of distances away (200 pc to 20,000 pc), their appearance can also range from small to large. The convolutional neural network reuses the same matrix on several scales, effectively applying the same feature-seeking tools to a wide range of size-scales.
- Astronomical data is often dominated by small, bright sources (stars) that are much brighter than emission spread out across images (clouds). Linear scaling of images results is squeezing interesting differences over a relatively small range, making training difficult. For pre-processing the images, we adapt a common non-linear scaling algorithm in astronomical image analysis, called z-scaling.

The convolutional neural network has a U-Net-inspired structure. A series of convolutional layers and max pooling layers are applied. Each series looks for ever more complicated structures. For example, the first layer may activate on horizontal or vertical lines. The second layer might activate on combinations of these shapes from the earlier layer. The third and final layer recognizes shapes associated with wind-blown bubbles.

This series of layers is first applied to the seven 1024x1024 images. The data dimensions are reduced by a factor of 2 (to 512x512) and the same series of layers are applied again. This process is repeated for reduction factors of 4, 8 and 16. The results of each application are combined for a final prediction. The prediction is a probability (between 0 & 1) that each pixel is inside or outside a wind-blown bubble. The model is defined in the file: cnnmodel.py

Observational data are used from several recent astronomical surveys. These include:
- Spitzer/GLIMPSE survey of the Galactic plane at 5.8 & 8.0 um
- WISE survey at 12 um
- Spitzer/MIPGAL survey at 24 um
- Herschel/PACS survey at 70 um
- THOR survey using the VLA at 1.1 cm

For training we use the catalog of wind-blown bubbles constructed by Povich et al. (20XX). This catalog was built by citizen-scientists identifying bubble shapes using the GLIMPSE and MIPSGAL surveys. The catalog contains about 3600 bubbles across Galactic positions |l| < 65 deg and |b| < 1. Teh catalog is available in csv format in file: mwpdr2bubbles.dat

Several scripts are used to download the data and convert them into a standard image size and name for training. The scripts are:
- createsatelliteframes.py
- cullB3frames.py
- downloadfits.py
- downloadirac3.py
- vlassdownload.py

The training is performed in the file cnnfitting.py and saved in a directory-formatted structure. Several draft models are provided, named cnnmodel(date)A where (date) is a MMDDYYYY format and A may be replaced with other letters.

To apply a model to new data, the file makepredictfitsandimages.py is provided. The script requires a user to refer to a model, such as are provided, and the input data. The script uses the input data and model to predict which pixels are inside wind-blown bubbles, then outputs the results as fits-formatted astronomy files and png-formatted image files. 
