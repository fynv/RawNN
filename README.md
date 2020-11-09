# RawNN

In this repository, I'm just practising to use cudnn to implement neural networks directly.
The 2 examples are the cudnn code port from PyTorch and Keras respectively.

The code in the BlazeFace-PyTorch folder is a direct port of [BlazeFace-PyTorch](https://github.com/hollance/BlazeFace-PyTorch), and it is expected to generate exactly the same result.

The code in the ImageClassifier folder is a direct port of [Image classification from scratch](https://keras.io/examples/vision/image_classification_from_scratch/), which is originally based on Keras. For various reason, the code doesn't reproduce the exact result of the original code. There are numerical errors, but the classification is still valid.

Note that these is inference only. Pretrained weights are statically included into the executable.

