# ARTNet
This is a simple convolutional neural network for recognizing the artist of several old style paintings by just by feeding the network only with the images itself.

### Table of contents
* [Introduction](#Introduction)
* [Neural Net Architecture](#Neural Net Architecture)
* [Training of the Network](#Training of the Network)
* [Examples](#Examples)
* [Built With](#built-with)
* [Authors](#authors)

## Introduction

The purpose of this project was to learn something about artificial neura networks. Therefore, I decided to program one for my self using the Matlab inbuild neural network tools. Furthermore, I decided to do some image classification and found a decend dataset containing paintings from thousands of artists including images of paintings, the artist names and painting titles. This sparked a idea inside me to create a CNN which classifies all the art pieces to their corresponding artists.

## Neural Net Architecture

For this Image classification task I used a convolutional neural network (CNN). I will not explain the working principle of a CNN, since there are alreday man good explanations on CNN's out there. Important is probably the the used architechture, which can be found seen in the list below.


Layers:

  * imageInputLayer(inputSize)
  * convolution2dLayer(10, 30)
  * batchNormalizationLayer
  * reluLayer
  * convolution2dLayer(3, 30)
  * batchNormalizationLayer
  * reluLayer
  * convolution2dLayer(3, 30)
  * batchNormalizationLayer
  * reluLayer
  * convolution2dLayer(2, 30)
  * batchNormalizationLayer
  * reluLayer
  * fullyConnectedLayer(height(labelCount))
  * reluLayer
  * fullyConnectedLayer(height(labelCount))
  * softmaxLayer
  * classificationLayer

inputSize = [100, 100, 3];
labelCount --> Number of different Labels

As you can see, the architecture is quite straight forward. As i tried out several Architectures, I came to the conlcusion that most importantly the CNN training time shouldnt take years. Therefore the CNN architecture was kept simple so that multiple epochs of training could be performed over night.
    
    
## Training of the Network

.....

## Examples

The CNN in action can be seen here, where random paintings on canvas from the artwork dataset are used and the corresponding artists are identified. The CNN is only fed with the image data itself.

![alt text]([http://url/to/img.png](https://github.com/HanSur94/ARTNet/blob/main/gif_1.gif))

## Built With

* [MATLAB](https://www.mathworks.com/products/matlab.html) - Version R2022a

## Authors

* **HanSur94** - [HanSur94](https://github.com/HanSur94)

