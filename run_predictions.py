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

    rl1d=(np.reshape(rl,np.shape(rl)[0]*np.shape(rl)[1]*3)/255.0)[::-1]
    cv=np.convolve(np.reshape(I,np.shape(I)[0]*np.shape(I)[1]*3)/255.0,rl1d)

    #plt.plot(range(0, len(cv)), cv)
    #plt.show()

    cv_r=np.resize(cv,np.shape(I))
    peaks1=find_peaks(cv, height=170, distance=3*np.shape(rl)[1])
    print(peaks1)
    if peaks1[1]['peak_heights'] != []:
        print(max(peaks1[1]['peak_heights']))
    peaks = peaks1[0] #np.append(np.append(np.append(peaks1[0],peaks2[0]),peaks3[0]),peaks4[0])
    peaks = np.sort(peaks)
    l_r = np.floor(0.5)
    l_c = np.floor(0.5)
    peaks_inb = [p for p in peaks if p < np.shape(I)[0]*np.shape(I)[1]*3]
    if peaks_inb != []:
        points=np.unravel_index(peaks_inb, np.shape(I))
        for i in range(0,len(points[0])):
            tl_row = int(points[0][i]) 
            tl_col = int(points[1][i]) 
            br_row = int(tl_row + np.shape(rl)[0])
            br_col = int(tl_col + np.shape(rl)[1])
            if tl_row > l_r+np.shape(rl)[0] or tl_col > l_c+np.shape(rl)[1]:
                bounding_boxes.append([tl_row,tl_col,br_row,br_col])
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
