import os
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def detect_red_light(I):
    '''
    This function takes a numpy array <I> and returns a list <bounding_boxes>.
    The list <bounding_boxes> should have one element for each red light in the 
    image. Each element of <bounding_boxes> should itself be a list, containing 
    four integers that specify a bounding box: the row and column index of the 
    top left corner and the row and column index of the bottom right corner (in
    that order). See the code below for an example.
    
    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''
    
    
    bounding_boxes = [] # This should be a list of lists, each of length 4. See format example below. 
    
    '''
    BEGIN YOUR CODE
    '''
    rl = Image.open('../redlight.jpg')
    rl1 = Image.open('../redlight1.jpg')
    rl2 = Image.open('../redlight2.jpg')     
    rl3 = Image.open('../redlight3.jpg')
    rl4 = Image.open('../redlight4.jpg')    

    rl1d=(np.reshape(rl,np.shape(rl)[0]*np.shape(rl)[1]*3)/255.0)#[::-1]
    rl11d=(np.reshape(rl1,np.shape(rl1)[0]*np.shape(rl1)[1]*3)/255.0)#[::-1]
    rl21d=(np.reshape(rl2,np.shape(rl2)[0]*np.shape(rl2)[1]*3)/255.0)#[::-1]
    rl31d=(np.reshape(rl3,np.shape(rl3)[0]*np.shape(rl3)[1]*3)/255.0)#[::-1]
    rl41d=(np.reshape(rl4,np.shape(rl4)[0]*np.shape(rl4)[1]*3)/255.0)#[::-1]
    
    matches = None
    irs = np.reshape(I,np.shape(I)[0]*np.shape(I)[1]*3)/255.0
    for i in range(0, len(irs)):
        matches = None
        if i+len(rl1d) < len(irs) and np.abs(np.sum(np.subtract(irs[i:i+len(rl1d)],rl1d))) < .25: 
            matches = i
        if i+len(rl11d) < len(irs) and np.abs(np.sum(np.subtract(irs[i:i+len(rl11d)],rl11d))) < .25: 
            matches = i
        if i+len(rl21d) < len(irs) and np.abs(np.sum(np.subtract(irs[i:i+len(rl21d)],rl21d))) < .25: 
            matches = i
        if i+len(rl31d) < len(irs) and np.abs(np.sum(np.subtract(irs[i:i+len(rl31d)],rl31d))) < .25: 
            matches = i
        if i+len(rl41d) < len(irs) and np.abs(np.sum(np.subtract(irs[i:i+len(rl41d)],rl41d))) < .25: 
            matches = i
        if  matches != None:
            points=np.unravel_index(i, np.shape(I))
            tl_row = int(points[0])
            tl_col = int(points[1])
            br_row = int(tl_row + np.shape(rl1)[0])
            br_col = int(tl_col + np.shape(rl1)[1])
    
    
    '''
    END YOUR CODE
    '''
    
    for i in range(len(bounding_boxes)):
        assert len(bounding_boxes[i]) == 4
    
    return bounding_boxes

# set the path to the downloaded data: 
data_path = '../RedLights2011_Medium'

# set a path for saving predictions: 
preds_path = '../hw01_preds' 
os.makedirs(preds_path,exist_ok=True) # create directory if needed 

# get sorted list of files: 
file_names = sorted(os.listdir(data_path)) 

# remove any non-JPEG files: 
file_names = [f for f in file_names if '.jpg' in f] 

preds = {}
for i in range(len(file_names)):
    
    # read image using PIL:
    print(os.path.join(data_path,file_names[i]))
    I = Image.open(os.path.join(data_path,file_names[i]))
    
    # convert to numpy array:
    I = np.asarray(I)
    
    preds[file_names[i]] = detect_red_light(I)

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds.json'),'w') as f:
    json.dump(preds,f)
