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
    
    irs = np.reshape(I,np.shape(I)[0]*np.shape(I)[1]*3)/255.0
    cv=np.convolve(irs,rl1d)
    cv1=np.convolve(irs,rl11d)
    cv2=np.convolve(irs,rl21d)
    cv3=np.convolve(irs,rl31d)
    cv4=np.convolve(irs,rl41d)

    #plt.plot(range(0, len(cv)), cv)
    #plt.show()

    cv_r=np.resize(cv,np.shape(I))
    peaks=find_peaks(cv, height=170, distance=3*np.shape(rl)[1])
    peaks1=find_peaks(cv1, height=280, distance=3*np.shape(rl1)[1])
    peaks2=find_peaks(cv2, height=797, distance=3*np.shape(rl2)[1])
    peaks3=find_peaks(cv3, height=151, distance=3*np.shape(rl3)[1])
    peaks4=find_peaks(cv4, height=540, distance=3*np.shape(rl4)[1])
    print(peaks)
    print(peaks[1]['peak_heights'])
    sort_peaks=[]
    if len(peaks) >= 2 and peaks[1]['peak_heights'] != []:
        print("max p:", max(peaks[1]['peak_heights']))
        peaks[1]['peak_heights']=peaks[1]['peak_heights']/float(max(peaks[1]['peak_heights']))
        sorted_peaks=peaks[0][np.argsort(peaks[1]['peak_heights'])]
        sort_peaks=np.append(sort_peaks,sorted_peaks)
    if len(peaks1) >= 2 and peaks1[1]['peak_heights'] != []:
        print("max p1:", max(peaks1[1]['peak_heights']))
        peaks1[1]['peak_heights']=peaks1[1]['peak_heights']/float(max(peaks1[1]['peak_heights']))
        sorted_peaks1=peaks1[0][np.argsort(peaks1[1]['peak_heights'])]
        sort_peaks=np.append(sort_peaks,sorted_peaks1)
    if len(peaks2) >= 2 and peaks2[1]['peak_heights'] != []:
        print("max p2:", max(peaks2[1]['peak_heights']))
        peaks2[1]['peak_heights']=peaks2[1]['peak_heights']/float(max(peaks2[1]['peak_heights']))
        sorted_peaks2=peaks2[0][np.argsort(peaks2[1]['peak_heights'])]
        sort_peaks=np.append(sort_peaks,sorted_peaks2)
    if len(peaks3) >= 2 and peaks3[1]['peak_heights'] != []:
        print("max p3:", max(peaks3[1]['peak_heights']))
        peaks3[1]['peak_heights']=peaks3[1]['peak_heights']/float(max(peaks3[1]['peak_heights']))
        sorted_peaks3=peaks3[0][np.argsort(peaks3[1]['peak_heights'])]
        sort_peaks=np.append(sort_peaks,sorted_peaks3)
    if len(peaks4) >= 2 and peaks4[1]['peak_heights'] != []:
        print("max p4:", max(peaks4[1]['peak_heights']))
        peaks4[1]['peak_heights']=peaks4[1]['peak_heights']/float(max(peaks4[1]['peak_heights']))
        sorted_peaks4=peaks4[0][np.argsort(peaks4[1]['peak_heights'])]
        sort_peaks=np.append(sort_peaks,sorted_peaks4)
    print(type(sort_peaks))
    if type(sort_peaks) != type([]):
        sort_peaks = sort_peaks.astype(int)
    l_r = np.floor(0.5)
    l_c = np.floor(0.5)
    peaks_inb = [p for p in sort_peaks if p < np.shape(I)[0]*np.shape(I)[1]*3]
    if peaks_inb != []:
        points=np.unravel_index(peaks_inb, np.shape(I))
        for i in range(0,len(points[0])):
            tl_row = int(points[0][i]) 
            tl_col = int(points[1][i]) 
            br_row = int(tl_row + np.shape(rl1)[0])
            br_col = int(tl_col + np.shape(rl1)[1])
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
