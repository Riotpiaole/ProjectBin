from sklearn import svm
import glob
from scipy import misc
import fnmatch
import numpy as np
from config import *

images = []

for image_path in glob.glob ( IMAGE_PATH ):
    label = 1 
    # 0 labeled as anormal
    if  fnmatch.fnmatch( image_path, '*2_*.png' ) :
        label= 2

    if  fnmatch.fnmatch( image_path, '*3_*.png' ) :
        label= 3

    if  fnmatch.fnmatch( image_path, '*4_*.png' ) :
        label= 4

    if  fnmatch.fnmatch( image_path, '*5_*.png' ) :
        label= 5

    if  fnmatch.fnmatch( image_path, '*6_*.png' ) :
        label= 6
    
    image = misc.imread( image_path ,flatten=True).reshape(  (160 * 160))
    image =np.append ( image, label )
    images.append( image )
    

np.random.shuffle ( images )
images = np.array(images)

X = images [ 0 : 15 , : 25600-1 ] 
y = images [ 0:15 , 25600 ] 

clf = svm.SVC()

clf.fit(X, y)