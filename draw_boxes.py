import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from PIL import ImageDraw

data_path = '../RedLights2011_Medium'
preds_path = '../hw01_preds'  

with open(os.path.join(preds_path,'preds.json')) as f:
    print(os.path.join(preds_path,'preds.json'))
    pred = json.load(f)
    for pic in pred.keys():
        im = Image.open(os.path.join(data_path,pic))
        draw = ImageDraw.Draw(im)
        # Create figure and axes
        #fig, ax = plt.subplots()

        # Display the image
        #ax.imshow(im)

        for box in pred[pic]:
            # Create a Rectangle patch
            #rect = patches.Rectangle((box[1], box[2]), box[3]-box[1], box[2]-box[0], linewidth=1, edgecolor='r', facecolor='none')

            # Add the patch to the Axes
            #ax.add_patch(rect)
            xy=[box[1],box[0],box[3],box[2]]
            draw.rectangle(xy, fill=None, outline=255)
        #plt.savefig(os.path.join(preds_path,pic))
        # Clear the current axes.
        #plt.cla() 
        # Clear the current figure.
        #plt.clf() 
        # Closes all the figure windows.
        #plt.close('all')   
        #plt.close(fig)
        del draw
        im.save(os.path.join(preds_path,pic), 'JPEG')
f.close()
