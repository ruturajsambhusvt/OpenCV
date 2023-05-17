import os
import cv2
import sys
from zipfile import ZipFile
from urllib.request import urlretrieve


# ========================-Downloading Assets-========================
def download_and_unzip(url, save_path):
    print(f"Downloading and extracting assests....", end="")

    # Downloading zip file using urllib package.
    urlretrieve(url, save_path)

    try:
        # Extracting zip file using the zipfile package.
        with ZipFile(save_path) as z:
            # Extract ZIP file contents in the same directory.
            z.extractall(os.path.split(save_path)[0])

        print("Done")

    except Exception as e:
        print("\nInvalid file.", e)


URL = r"https://www.dropbox.com/s/efitgt363ada95a/opencv_bootcamp_assets_12.zip?dl=1"

asset_zip_path = os.path.join(os.getcwd(), f"opencv_bootcamp_assets_12.zip")

# Download if assest ZIP does not exists.
if not os.path.exists(asset_zip_path):
    download_and_unzip(URL, asset_zip_path)
# ====================================================================

device = 1 #Since we use USB camera, the device is 1

if len(sys.argv)>1:
    device = sys.argv[1]
    
cap = cv2.VideoCapture(device)

win_name = "Camera_Preview"

cv2.namedWindow(win_name,cv2.WINDOW_NORMAL)

net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")
#prototext file contains the model architecture
#caffemodel file contains the model parameters
#Go to the github page of opencv and download the files, see the model.yaml file for more info about weights and model parameters

#Model parameters
in_width = 300
in_height = 300
mean = [104, 117, 123] #BGR
conf_threshold = 0.7

while cv2.waitKey(1)!=27: #esc
    has_frame,frame = cap.read()
    if not has_frame:
        break
    frame = cv2.flip(frame,1) #Flip the frame horizontally for convenience
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    
    #Create a 4D blob from the frame
    #What is a blob? A blob is a group of connected pixels in a binary image that share some common property (Eg: grayscale value)
    blob = cv2.dnn.blobFromImage(frame,1.0,(in_width,in_height),mean,swapRB=False,crop=False) 
    #since both model and opencv use BGR, we don't need to swap the channels
    #Run a forward pass through the network to detect faces
    net.setInput(blob)
    detections = net.forward()
    
    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence>conf_threshold:
            x_left_bottom = int(detections[0,0,i,3]*frame_width)
            y_left_bottom = int(detections[0,0,i,4]*frame_height)
            x_right_top = int(detections[0,0,i,5]*frame_width)
            y_right_top = int(detections[0,0,i,6]*frame_height)
            
            cv2.rectangle(frame,(x_left_bottom,y_left_bottom),(x_right_top,y_right_top),(0,255,0))
            label = "Confidence: %.4f"%(confidence)
            label_size,base_line  = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.5,1)
            #explanation of getTextSize function parameters: https://docs.opencv.org/3.4/d6/d6e/group__imgproc__draw.html#ga3d2abfcb995fd2db908c8288199dba82
    
            cv2.rectangle(frame,(x_left_bottom,y_left_bottom-label_size[1]),(x_left_bottom+label_size[0],y_left_bottom+base_line),(255,255,255),cv2.FILLED)
            cv2.putText(frame,label,(x_left_bottom,y_left_bottom),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0))
            
    t,_ = net.getPerfProfile()
    #explanation of getPerfProfile function: going to return the overall time for inference and the timings for each layer
    label = "Inference time: %.2f ms"%((t*1000.0)/cv2.getTickFrequency())
    cv2.putText(frame,label,(0,15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0))
    cv2.imshow(win_name,frame)
