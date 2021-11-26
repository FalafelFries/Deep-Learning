#Real Time Object Detection using SSD with MobileNet as Base Network

#importing required libraries
import numpy as np, imutils, cv2

#path to prototype machine learning model created for use with Caffe
prototxt = <<enter path>>
#path to pre-trained model - Caffe, a deep learning framework
model = <<enter path>>
confThresh = 0.2 #initialising minimum threshold to filter weak detections

# initialising the list of class labels MobileNet SSD was trained to
CLASSES = ["bicycle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

#generating a set of bounding box colors for each class
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

#loading model
print("Loading the model ...")
net = cv2.dnn.readNetFromCaffe(prototxt, model)
print("Model loaded.")
print("Starting camera ...")
cam = cv2.VideoCapture(0)

#looping over every frame in the video
while True :

    #preprocessing
    _, frame = cam.read() #reading frame from camera
    frame = imutils.resize(frame, width = 500) #resizing output frame
    (h, w) = frame.shape[:2] #getting frame dimensions
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5) #converting to blob

    net.setInput(blob) #setting blob as the input to our neural network
    detections = net.forward() #feeding the input through the network to obtain detections

    #drawing bounding box around detected objects
    # loop over all the detected objects in the frame
    for i in np.arange(0, detections.shape[2]) :

        confidence = detections[0, 0, i, 2] #extracting confidence of detected object
        #if confidence of detected object is above our minimum threshold then:
        if confidence > confThresh :

            idx = int(detections[0, 0, i, 1]) #extract the index of the class label from the detections
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h]) #extracting the (x, y) coordinates of the box
            startX, startY, endX, endY = box.astype("int")

            #displaying label which contains class id of the object and its associated confidence
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            # drawing the bounding box on the frame
            cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
            cv2.putText(frame, label, (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
    
    cv2.imshow("Frame", frame) #displaying the outputframe
    key = cv2.waitKey(1)
    if key == ord("q") : break #if the 'q' key ispressed, break from the loop

#cleanup work
cam.release()
cv2.destroyAllWindows()