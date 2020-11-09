
import torch
import os
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
from IPython.display import display
# CONSTANT
## Object detection classes for the pretrained torchvision model trained on COCO Dataset 
CLASSES_OD = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# PRINT FUNCTIONS
def print_classes_prediction(y: list) -> list:
    """
        This function print the object that the model from torchvision have detected. 
        Object detection model from torchvision return a list of dictionnary.
        y : prediction returned by a pytorch detection model. 
    """
    for i,img in enumerate(y):
        classes = []
        for c in list(img['labels']):
            classes.append(CLASSES_OD[c])
            
        print('Object in Image',i+1,':',classes)


def print_image_bbox_GT(img: torch.Tensor ,bbox: dict):
    """
        Print Image and bounding box Ground Truth from the Pascal VOC Detection. 
        img : original image ; Tensor Object.
        bbox : Ground Truth ; dict.
    """
    
    # Pytorch tensor image in dataset is size : (3,xwidth,ywidth)
    # Matplotlib need : (xwidth,ywidth,3)
    img = img.transpose(0,2)
    img = img.transpose(0,1)
    # Plot images
    ## Create figure and axes
    fig,ax = plt.subplots(1)
    ax.imshow(img)    
    # Plot bbox
    for o in bbox['annotation']['object']:
        x,y = int(o['bndbox']['xmin']), int(o['bndbox']['ymin'])
        w = int(o['bndbox']['xmax']) - int(o['bndbox']['xmin']) # Width
        h = int(o['bndbox']['ymax']) - int(o['bndbox']['ymin']) # Height
        # Create a Rectangle patch
        rect = patches.Rectangle((x,y),width=w,height=h,linewidth=1,edgecolor='yellow',facecolor='none',label=o['name'])
        # Add the patch to the Axes
        ax.add_patch(rect)
        # Add object name right top of the bbox 
        ax.annotate(o['name'].capitalize(), (int(o['bndbox']['xmax']), int(o['bndbox']['ymin'])), color='yellow', weight='bold', 
                fontsize=10, ha='right', va='bottom')
    plt.show()

def print_image_bbox_prediction(img: torch.Tensor ,pred: dict,threshold=0,color='yellow'):
    """
        Print Image and bounding box prediction of a Torchvision detection model.
        img : original image ; Tensor Object.
        pred : Detection prediction of the images with a torchvision detection model ; dict.
        treshold : this function print the object with a confidence score btwn [treshold-1]
    """
    
    # Error control
    if type(pred)=='list':
        raise Exception('pred must be a dict not a list of prediction')
    if threshold > 1 or threshold <0: 
        raise Exception('treshold must be a float between 0 and 1')
    # Pytorch tensor image in dataset is size : (3,xwidth,ywidth)
    # Matplotlib need : (xwidth,ywidth,3)
    img = img.transpose(0,2)
    img = img.transpose(0,1)
    # Plot images
    ## Create figure and axes
    fig,ax = plt.subplots(1)
    ax.imshow(img)    
    # Plot bbox
    for i,o in enumerate(pred['boxes']):
        if pred['scores'][i] < threshold : # If the confident score is less than treshold then dont print the object.
            break
        x,y = o[0],o[1]
        w = o[2] - o[0] # Width
        h = o[3] - o[1] # Height
        # Create a Rectangle patch
        o_name = CLASSES_OD[pred['labels'][i]] # name of the detected object
        rect = patches.Rectangle((x,y),width=w,height=h,linewidth=1,edgecolor=color,facecolor='none',label=o_name)
        # Add the patch to the Axes
        ax.add_patch(rect)
        # Add object name right top of the bbox 
        ax.annotate(o_name.capitalize(), (o[2], o[1]), color=color, weight='bold', 
                fontsize=10, ha='right', va='bottom')
    plt.show()


def load_images(folder,img_name,size_img=(224,224),normalize=True,print=False) -> torch.Tensor:
    """
        Load random images to test for detection algorithms. 
        Return a resized and normalized tensor image ready to be pass in a torchvision detection model.
        print : Print the image before transformation ; bool 
    """
    if normalize:
        transform=transforms.Compose([
            transforms.Resize(size_img),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
    else:
        transform=transforms.Compose([
            transforms.Resize(size_img),
            transforms.ToTensor(),])

    image_path = os.path.join(folder,img_name)
    img = Image.open(image_path).convert('RGB')
    if print :
        display(img)
    img_t = transform(img)
    x = torch.unsqueeze(img_t,0)
    return x





