object detection - 
1. locating the object and 
2. identifying it.

Object Detection has found its application in a wide variety of domains such as 
- video surveillance, 
- image retrieval systems, 
- autonomous driving vehicles and many more.

YOLO is one of the fastest real-time object detection algorithm as compared to R-CNN family (R-CNN, Fast R-CNN, Faster R-CNN, etc.)

The R-CNN family of algorithms uses regions to localise the objects in images which means the model is applied to multiple regions and high scoring regions of the image are considered as object detected. But YOLO follows a completely different approach. Instead of selecting some regions, it applies a neural network to the entire image to predict bounding boxes and their probabilities.

We have two options to get started with object detection:

1. Using the pre-trained model
2. Training custom object detector from scratch

#### 1. Using the pre-trained model

Steps:
1. load the YoloV3 weights and configuration file with the help of `dnn` module of OpenCV.

```py
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
```

2. `coco.names` file contains the names of the different objects that our model has been trained to identify. We store them in a list called classes.

```py
classes = []
with open("coco.names", "rb") as f:
    classes = f.read().splitlines()
    
#classes = [line.strip() for line in f.readlines()]
```

3. Now to run a forward pass using the `cv2.dnn` module, we need to pass in the names of layers for which the output is to be computed. `net.getUnconnectedOutLayers()` returns the indices of the output layers of the network.

```py
output_layers_names = net.getUnconnectedOutLayersNames()

#print(output_layers_names) -> ['yolo_82', 'yolo_94', 'yolo_106']
```

4. read the image, resize it

5. To correctly predict the objects with deep neural networks, we need to preprocess our data and `cv2.dnn` module provides us with two functions for this purpose: 
    1. blobFromImage and 
    2. blobFromImages. 
    
    These functions perform scaling, mean subtraction and channel swap which is optional. We will use `blobFromImage` in a function called detect_objects() that accepts image/frame from video or webcam stream, model and output layers as parameters.
    
```py
blob = cv2.dnn.blobFromImage(img, scalefactor=1/255, size=(416, 416), mean=(0, 0, 0), swapRB=True, crop=False)
net.setInput(blob)
layerOutputs = net.forward(output_layers_names)
```

Hence, we are scaling the image pixels to the range of 0 to 1. There is no need for mean subtraction and that’s why we set it to (0, 0, 0) value.

The `forward()` function of `cv2.dnn` module returns a nested list containing information about all the detected objects which includes 
- the x and y coordinates of the centre of the object detected, 
- height and width of the bounding box,
- confidence and scores for all the classes of objects listed in coco.names. 

The class with the highest score is considered to be the predicted class.