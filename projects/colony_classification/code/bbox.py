from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib as mpl
import random
import math
import sys
import imageio
import re

from skimage import data, color
from skimage.transform import hough_circle
from skimage.feature import peak_local_max, canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte
from skimage import exposure
from skimage import measure
from sklearn.cluster import AgglomerativeClustering

# unsure of package names for different versions
try:
    from skimage import filters
except ImportError:
    from skimage import filter as filters

class HoughPlateDetect:

    def __init__(self, image, minRadii, maxRadii):
        self.image = image    
        self.edges = canny(image, sigma=3, low_threshold = 10, high_threshold = 50)
        self.tryRadii= np.arange(minRadii, maxRadii, 2)
        self.centers = []            #centers of circles from hough
        self.accums = []            #values of hough at centers
        self.radii = []                #radii of test radii that match hough

    def perform_transform(self, peaks_to_lookfor):
        hough_transform = hough_circle(self.edges, self.tryRadii)

        for triedRadii, houghResult in zip(self.tryRadii, hough_transform):
            num_peaks = peaks_to_lookfor
            peaks = peak_local_max(houghResult, num_peaks = num_peaks) #returns list of coordinates of peaks (x,y)

            # add higest accum positions to centers, and the values to accums
            self.centers.extend(peaks)
            self.accums.extend(houghResult[peaks[:, 0], peaks[:,1]])
            self.radii.extend([triedRadii]* num_peaks)
        
    def return_accums(self):
        return self.accums
    
    def return_centers(self): 
        return self.centers

    def return_visualization_parameters(self):
        #return parameters for best circle properties (center position and radius)
        best_accum_index = np.argsort(self.accums)[::-1][0]
        center_x, center_y = self.centers[best_accum_index]
        radius = self.radii[best_accum_index]

        return [center_y, center_x, radius]

    def visualize_hough(self, numCircles):
        hough = color.gray2rgb(self.image)
        
        #plot numCircles circles for hai(highest accumulant index)
        for hai in np.argsort(self.accums)[::-1][:numCircles]:
            
            center_x, center_y = self.centers[hai]
            radius = self.radii[hai]
            cx, cy = circle_perimeter(center_y, center_x, radius)
            hough[cy, cx] = (220, 20, 20)

        return hough

class OtsuSegment:

    def __init__(self, some_image, map):
        self.image = some_image
        self.map = map
        self.val = filters.threshold_otsu(self.map)

    def visualize_otsu(self):    
        
        hist, bins_center = exposure.histogram(self.image)

        plt.figure(figsize=(9, 4)) #wh in inches on screen
        plt.subplot(121)
        plt.imshow(self.image, cmap='gray', interpolation='nearest')
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(self.image > self.val, cmap='gray', interpolation='nearest')
        plt.axis('off')
        # plt.subplot(133)
        # plt.plot(bins_center, hist, lw=2)
        # plt.axvline(self.val, color='k', ls='--')

        plt.tight_layout()
        plt.show()

    def return_otsu(self):
        return self.image > self.val
        

class CircleMask:
    
    def __init__(self, image, parameter, pixelbuffer):
        self.image = image
        self.ccx = parameter[0]
        self.ccy = parameter[1]
        self.cradius = parameter[2]
        self.buffer = pixelbuffer

        ly,lx = image.shape
        y,x  = np.ogrid[0:ly , 0 :lx]
        
        mask = (x- self.ccx)**2 + (y - self.ccy)**2 > (self.cradius - self.buffer)**2
        self.image[mask] = 0
        
    def return_mask(self):
        return self.image

if __name__ == '__main__':

    #bamfile = pysam.AlignmentFile(sys.argv[1], "rb")
    
    plate = sys.argv[1]

    image = imageio.imread(plate)
    #image = img_as_ubyte(img)

    #greyscale_im = GreyScaleConverter(image).return_greyscale(False)
    #print "finished greyscale"
    #print hough.return_accums()
    
    ''' hough transform section'''

    #get plate outline through hough transform
    hough = HoughPlateDetect(image, 1000, 1100)
    hough.perform_transform(2)
    print("done plate detect")

    #get size of this hough transform outline
    parameters = hough.return_visualization_parameters()
    #create sample square image of plate to be used in thresholding
    cen_x = parameters[0]
    cen_y = parameters[1]
    radius = parameters[2]

    leftmost_col = int(cen_x - float(radius)/math.sqrt(2))
    rightmost_col = int(cen_x + float(radius)/math.sqrt(2))
    upmost_row = int(cen_y + float(radius)/math.sqrt(2))
    lowest_row = int(cen_y - float(radius)/math.sqrt(2))

    square = []
    for i in range(lowest_row-1, upmost_row):
        square.append(image[i][leftmost_col -1:rightmost_col -1])

    #print(len(square))
    #print(len(image))
    #optionally view hough result
    #hough_img_overlay = hough.visualize_hough(3)
    #plt.imshow(hough_img_overlay, cmap = plt.cm.gray)
    #plt.show()
       
    #plt.imshow(square, cmap = plt.cm.gray)
    #plt.show()

    otsu = OtsuSegment(image, np.asarray(square))
    #otsu.visualize_otsu()
    
    otsu_img = otsu.return_otsu()
    
    np.save('plate.npy', otsu_img)

    #plt.imshow(otsu_img, cmap = plt.cm.gray)
    #plt.show()
#########################################################################################
    '''circle mask testing, and perimeter elimination'''

    #labeling segmented region
    labels, numlabels = measure.label(otsu.return_otsu(), background = 0, return_num = True)
    
    #count sizes of each from flattened array
    initial_sizes = np.bincount(labels.ravel())
    initial_sizes[0] = 0

    #setting foreground regions < size(25) to background
    small_sizes = initial_sizes < 5
    small_sizes[0] = 0

    #get rid of large foreground objects
    large_sizes = initial_sizes > 10000
    large_sizes[0] = 0

    labels[small_sizes[labels]] = 0
    labels[large_sizes[labels]] = 0

    preprocessed_sizes = np.bincount(labels.ravel())

    #CM = CircleMask(otsu.return_otsu(), parameters, 300)
    #plt.imshow(CM.return_mask(), cmap = plt.cm.gray)
    #plt.show()
    #
    #plt.imshow(labels, cmap = plt.cm.gray)
    #plt.show()

    #apply circle mask to labels grid
    circle_mask = CircleMask(labels, parameters, 100)
    labels_mask = circle_mask.return_mask()
    mask_sizes = np.bincount(labels_mask.ravel())
    mask_sizes[0] = 0

    print(str(np.count_nonzero(mask_sizes)) + " labels before hough mask elimination. \n")
    #getting rid of perimeter colonies
    #binwidth = 5
    #plt.hist(mask_sizes, bins=range(5, max(mask_sizes) + binwidth, binwidth))
    
   # plt.plot(mask_sizes)
   # plt.show()

    #loop through to see which bin counts are truncated (brute force)
    bins_truncated = []
    for bin in range(len(mask_sizes)):
        if mask_sizes[bin] < preprocessed_sizes[bin]:
            bins_truncated.append(bin)
    
    #reset masked label map to remove truncated bins
    for row in range(len(labels_mask)):
        for col in range(len(labels_mask[0])):
            if labels_mask[row][col] in bins_truncated:
                labels_mask[row][col] = 0

    hough_sizes = np.bincount(labels_mask.ravel())
    hough_sizes[0] = 0

    print(str(np.count_nonzero(hough_sizes)) + " after hough mask elimination. \n")

    #plt.imshow(labels_mask, cmap=plt.cm.gray)
    #plt.show()

    # binwidth = 5
    # plt.hist(final_sizes, bins=range(5, max(final_sizes) + binwidth, binwidth))
    # plt.title('Colony size distribution of single agar plate')
    # plt.xlabel('Size')
    # plt.ylabel('Abundance')
    # plt.show()
    #getting intensities of petites in small range using mask_sizes

    irregular_labels = []
    props = measure.regionprops(labels_mask, image)
    
    position_file = open('./' + plate[:-11] + '_ohc.txt', 'r')
    centroid_pos = []
    bits = []
    
    ohc2 = open('./' + plate[:-4] + '_bbox.txt', 'w+')
    bboxes = []
    
    for line in position_file.readlines():
        x_loc = re.search('\((.+?)\,', line)
        if x_loc:
            x = x_loc.group(1)
        y_loc = re.search('\s(.+?)\)', line)
        if y_loc:
            y = y_loc.group(1)
        centroid_pos.append((int(y)*2.724, int(x)*2.724))
        gp = re.search('\)\,\s(.+?)\,', line)
        #gp = re.search('\s(.+?)\]', line)
        if gp:
            bit = gp.group(1)
            bits.append(bit)
    
    for j in range(len(centroid_pos)):
        dist_min = 100000
        index = 0
        for i in range(len(props)):
            dist = np.linalg.norm(np.subtract(centroid_pos[j], props[i].centroid))
            if dist < dist_min:
                dist_min = dist
                index = i
        #print(pos, props[index].centroid, dist)
        box = props[index].bbox
        ohe_data = []
        bit = bits[j]
        if int(bit) == 1:
            ohe_data = [box, 1, 0]
        if int(bit) == 0:
            ohe_data = [box, 0, 1]
        ohc2.write(str(ohe_data) + '\n')
        bboxes.append(box)
    
    position_file.close()
    ohc2.close()
    
    segmented_boxes = []
    for i in range(len(props)):
        segmented_boxes.append(props[i].centroid)
        print(props[i].bbox)
    
    fig,ax = plt.subplots(1)
    ax.imshow(image)
    for b in segmented_boxes:
        rect = patches.Rectangle((b[1], b[0]), 50, 50, linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
    plt.show()
    
    '''
    
    segmented_boxes = []
    for i in range(len(props)):
        segmented_boxes.append(props[i].bbox)

    fig,ax = plt.subplots(1)
    ax.imshow(image)
    for b in segmented_boxes:
        min_row = b[0]
        min_col = b[1]
        width = b[2] - b[0]
        height = b[3] - b[1]
        rect = patches.Rectangle((min_col,min_row), width, height, linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
    for p in centroid_pos:
        plt.plot(p[0], p[1])
    plt.show()
    '''
                