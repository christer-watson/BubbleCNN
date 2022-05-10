# BubbleCNN
BubbleCNN is a tool to use a convolutional neural network to recognize wind-blown bubbles in astronomical images. In broad strokes, a convolutional neural network is a tool for recognizing specific features, from simple features like horizontal lines to complex features like human faces. This project uses tensorflow's standard convolutional neural network tools to build a model that is tailored toward astronomical data. The modifications made, mostly to accomodate properties common in astronomical data but uncommon in other data:
- Data are high signal to noise, often S/N > 3000, which is higher than supported by stanard 8-bit PNG data format
- 7 data channels ('colors' to non-astronomers) are supported, allowing features from IR to radio light to be used for recognizing wind-blown-bubbles
- A catalog of 1000s of wind-blown bubbles, each represented by seven 1024x1024 image requires loading data in batches. A data generator customized was required to handle the high S/N data files and 7 data channels
- Typical wind-blown bubbles occupy a small area of the data (< 2%), so an intersection/union-style metric, Dice, was used.

