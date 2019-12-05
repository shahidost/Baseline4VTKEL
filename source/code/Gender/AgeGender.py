

    """
    python code for pre-train gender classification from github repo of:
    Ref: https://www.cv-foundation.org/openaccess/content_cvpr_workshops_2015/W08/html/Levi_Age_and_Gender_2015_CVPR_paper.html
    """        

# Import required modules
import cv2 as cv

def getFaceBox(net, frame, conf_threshold=0.7):
    """
    Processing pre-trained gender classifier model for face detection with respect to male/female features.
    """        

    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 4)
            #(x1=x,y1=y),(x2=x+w,y2=y+h)
    return frameOpencvDnn, bboxes

#Gender classification helping files path
faceProto = "/root-directory/Gender/opencv_face_detector.pbtxt"
faceModel = "/root-directory/Gender/opencv_face_detector_uint8.pb"

ageProto = "/root-directory/Gender/age_deploy.prototxt"
ageModel = "/root-directory/Gender/age_net.caffemodel"

genderProto = "/root-directory/Gender/gender_deploy.prototxt"
genderModel = "/root-directory/Gender/gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# We linked two gender classify by algorithm into Person->Male->Man110287213 and Person->Female->Female109619168 

def Gender_main(image):

    """
    This function takes input image and process for gender classification into Male or Female classes. We used pre-trained model developed by Gil Levi et. all..
     
    input:
        image - Flickr30k image file path
 
     Output:
         gender_data - gender labels (male/female)
         bboxes - bounding boxes information
    """        
    
    gender_data=[]
    args = image
    ageNet = cv.dnn.readNet(ageModel, ageProto)
    genderNet = cv.dnn.readNet(genderModel, genderProto)
    faceNet = cv.dnn.readNet(faceModel, faceProto)
    
    cap = cv.VideoCapture(args if args else 0)
    padding = 20
    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            cv.waitKey()
            break
    
        frameFace, bboxes = getFaceBox(faceNet, frame)
        if not bboxes:
            print("No face Detected, Checking next frame")
            continue
    
        for bbox in bboxes:
            face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
    
            blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]
            gender_data.append(gender)
    
            ageNet.setInput(blob)
#            agePreds = ageNet.forward()

    return gender_data, bboxes
