from scipy import misc
from scipy import ndimage as ndi

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import random
import math
import sys
import cv2
import imageio

from skimage import segmentation
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
		self.centers = []			#centers of circles from hough
		self.accums = []			#values of hough at centers
		self.radii = []				#radii of test radii that match hough

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

	an_image = imageio.imread(sys.argv[1])
	print(an_image.shape)
	rgb_weights = [0.2989, 0.5870, 0.1140]

	image = np.dot(an_image[..., :3], rgb_weights).astype(int)
	print(image[0][0:10])
	#image = img_as_ubyte(img)
	print(image.shape)

	plt.imshow(image, cmap=plt.cm.gray)
	plt.show()

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

	print(len(square))
	print(len(image))
	
	#optionally view hough result
	hough_img_overlay = hough.visualize_hough(3)
	plt.imshow(hough_img_overlay, cmap = plt.cm.gray)
	plt.show()

	plt.imshow(square, cmap = plt.cm.gray)
	plt.show()

	otsu = OtsuSegment(image, np.asarray(square))
	otsu.visualize_otsu()

	#plt.imshow(otsu.return_otsu(), cmap = plt.cm.gray)
	#plt.show()
#########################################################################################
	'''circle mask testing, and perimeter elimination'''

	#labeling segmented region
	labels, numlabels = measure.label(otsu.return_otsu(), background = 0, return_num = True)

	#count sizes of each from flattened array
	initial_sizes = np.bincount(labels.ravel())
	initial_sizes[0] = 0

	#setting foreground regions < size(50) to background
	small_sizes = initial_sizes < 50
	small_sizes[0] = 0

	print(initial_sizes)
	print(small_sizes)

	print(small_sizes[labels])

	#get rid of large foreground objects
	large_sizes = initial_sizes >10000
	large_sizes[0] = 0

	labels[small_sizes[labels]] = 0
	labels[large_sizes[labels]] = 0

	preprocessed_sizes = np.bincount(labels.ravel())

	#CM = CircleMask(otsu.return_otsu(), parameters, 300)
	#plt.imshow(CM.return_mask(), cmap = plt.cm.gray)
	#plt.show()
	#
	# plt.imshow(labels, cmap = plt.cm.gray)
	# plt.show()

	#apply circle mask to labels grid
	circle_mask = CircleMask(labels, parameters, 100)
	labels_mask = circle_mask.return_mask()

	mask_sizes = np.bincount(labels_mask.ravel())
	mask_sizes[0] = 0

	print(str(np.count_nonzero(mask_sizes)) + " labels before hough mask elimination. \n")
	#getting rid of perimeter colonies
	#binwidth = 5
	#plt.hist(mask_sizes, bins=range(5, max(mask_sizes) + binwidth, binwidth))
	
	#plt.plot(mask_sizes)
	#plt.show()

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

	#add watershed
	re_binarized = labels_mask > 0
	labels_mask_rebinarized = labels_mask.copy()
	labels_mask_rebinarized[re_binarized] = 1

	dist_transform = cv2.distanceTransform(labels_mask_rebinarized.astype(np.uint8), cv2.DIST_L2, 3)
	local_max_boolean = peak_local_max(dist_transform, min_distance=2, indices=False)
	markers, _ = ndi.label(local_max_boolean)
	segmented_watershed = segmentation.watershed(255 - dist_transform, markers, mask=labels_mask_rebinarized)

	props_otsu = measure.regionprops(labels_mask, image)
	props_watershed = measure.regionprops(segmented_watershed, image)

	#iterate through watershed boxes, eliminahough_img_overlayting those that perfectly overlap (indicating that identical distance pixel values exist), or reassigning
	bboxes_watershed = []
	label_watershed = []

	fractional_overlaps = []
	label_overlaps = []
	target_overlap = []

	for i in range(len(props_watershed)):
		bboxes_watershed.append(props_watershed[i].bbox)
		label_watershed.append(props_watershed[i].label)

	for i, b1 in enumerate(bboxes_watershed):
		minr_1, minc_1, maxr_1, maxc_1 = b1
		fractional_area_intersections = []
		label_intersections = []
		target_overlap.append(label_watershed[i])
		for j, b2 in enumerate(bboxes_watershed):
			minr_2, minc_2, maxr_2, maxc_2 = b2

			if i!=j:
				isOverlapped = (minc_1 <= maxc_2 and minc_2 <= maxc_1 and minr_1 <= maxr_2 and minr_2 <= maxr_1)

				#compute area of intersection and remove bbox if contained within another
				if isOverlapped:
					left = max(minc_1,minc_2)
					right = min(maxc_1, maxc_2)
					top = min(maxr_1,maxr_2)
					bottom = max(minr_1, minr_2)

					intersect_area = abs(left-right) * abs(top-bottom)

					b1_area = abs(maxr_1 - minr_1) * abs(maxc_1 - minc_1)
					fractional_area_intersections.append(intersect_area/float(b1_area))
					label_intersections.append(label_watershed[j])

		fractional_overlaps.append(fractional_area_intersections)
		label_overlaps.append(label_intersections)

	#if a bounding box is completely overlapped, change it's label values
	for i, target_label in enumerate(target_overlap):
		print(sum(fractional_overlaps[i]))
		if sum(fractional_overlaps[i])>=1:
			label_to_assign = label_overlaps[i][np.argmin(np.array(fractional_overlaps[i]))]

			raveled = np.ravel(segmented_watershed)
			arg_array = np.arange(max(raveled)+1)
			bool_array = arg_array == target_label

			segmented_watershed[bool_array[segmented_watershed]]=label_to_assign

	props_watershed_culled = measure.regionprops(segmented_watershed, image)
	bboxes_watershed_culled = []

	for i in range(len(props_watershed_culled)):
		bboxes_watershed_culled.append(props_watershed_culled[i].bbox)

	# plot bounding boxes colour coded green as grande, red as petite
	for b in bboxes_watershed_culled:

		minr, minc, maxr, maxc = b
		bx = (minc, maxc, maxc, minc, minc)
		by = (minr, minr, maxr, maxr, minr)
		plt.plot(bx, by, '-g', linewidth=2.5)

	print(str(np.count_nonzero(hough_sizes)) + " after hough mask elimination. \n")

	plt.imshow(labels_mask, cmap=plt.cm.gray)
	plt.show()

	# binwidth = 5
	# plt.hist(final_sizes, bins=range(5, max(final_sizes) + binwidth, binwidth))
	# plt.title('Colony size distribution of single agar plate')
	# plt.xlabel('Size')
	# plt.ylabel('Abundance')
	# plt.show()
	#getting intensities of petites in small range using mask_sizes

	irregular_labels = []
	props = measure.regionprops(segmented_watershed, image)

	#add another layer of shape processing, removing irregular shapes
	for i in range(len(props)):
		eccentricity = props[i].eccentricity
		size = props[i].area
		#if too irregular, record labels with this irregularity
		if eccentricity > 0.9:
			irregular_labels.append(props[i].label)

	print(irregular_labels)

	for row in range(len(segmented_watershed)):
		for col in range(len(segmented_watershed[0])):
			if segmented_watershed[row][col] in irregular_labels:
				segmented_watershed[row][col] = 0

	# count sizes of each from flattened array
	initial_sizes = np.bincount(segmented_watershed.ravel())
	initial_sizes[0] = 0

	# setting foreground regions < size(50) to background
	small_sizes = initial_sizes < 50
	small_sizes[0] = 0

	# get rid of large foreground objects
	large_sizes = initial_sizes > 10000
	large_sizes[0] = 0

	segmented_watershed[small_sizes[segmented_watershed]] = 0
	segmented_watershed[large_sizes[segmented_watershed]] = 0

	props = measure.regionprops(segmented_watershed, image)

	plt.imshow(segmented_watershed, cmap=plt.cm.gray)
	plt.show()

	formatted_output = []
	for i in range(len(props)):
		minr, minc, maxr, maxc = props[i].bbox

		perline_entry = [minr, minc, maxr, maxc]
		perline_entry.extend([props[i].area, props[i].mean_intensity])

		print(perline_entry)
		formatted_output.append(perline_entry)

	np.savetxt(sys.argv[1][:-4] + "_bbox_size_intensity.txt", formatted_output)

