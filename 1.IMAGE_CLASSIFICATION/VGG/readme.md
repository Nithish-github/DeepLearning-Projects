# VGG Net Classification Model 🌐🖼️

Welcome to the VGG Net Classification Model repository! 🎉 This project implements the VGG (Visual Geometry Group) network, a popular deep learning architecture used for image classification tasks. Developed by researchers at the University of Oxford, VGG Net has been widely recognized for its simplicity and effectiveness in image recognition challenges. 📚🔍
# What is VGG Net? 🤔

VGG Net is a convolutional neural network (CNN) architecture designed to classify images into various categories. The hallmark of VGG Net is its use of very small (3x3) convolution filters, which enables the network to learn deep representations of visual data. The most commonly used versions of VGG Net are VGG16 and VGG19, which refer to the number of layers in the network.

# Key Features ✨

Simplicity: VGG Net employs a straightforward and uniform architecture, making it easy to understand and implement.
Depth: With 16 or 19 layers, VGG Net can learn complex patterns and features from images.
Performance: Despite its simplicity, VGG Net has achieved top performance in various image classification benchmarks. 🏅

# Loss Function Used 

The VGG network model uses the cross-entropy loss function, which works on a logarithmic approach. After one-hot encoding the corresponding class, the log value is calculated. By default, the softmax activation function is used with the cross-entropy function. The softmax activation function gives a probability output: it takes the exponential value of each class score and divides it by the sum of all exponential values. Through this process, the loss is calculated.

# Applications 📸

VGG Net is widely used in various computer vision applications, including:

Object recognition 🚗 <br> 
Facial recognition 👤 <br> 
Scene understanding 🏞️ <br> 
Image retrieval 🔍