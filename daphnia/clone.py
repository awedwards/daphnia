from __future__ import division
import numpy as np
import pandas as pd
import os
import cv2
import scipy
from scipy.ndimage import map_coordinates as mc
from skimage import measure
from skimage.filters import gaussian
import random

class Clone(object):

    ''' Class containing attributes for an individual Daphnia clone image
        as well as methods for detecting and analyzing features

        Required initialization variables:

        filepath (str): string coding path to image of daphnia
    '''

    def __init__(self, filepath, **kwargs):
           
        self.filepath = filepath    # points to image
        if os.name == "posix":    # if on Linux or Mac
            self.filebase = filepath.split("/")[-1]
        elif os.name == "nt":    # if on windows
            self.filebase = filepath.split("\\")[-1]
        # attributes are initialized with nan, which can be used as a check
        # to see if certain steps of the pipeline have been completed
        self.animal_area = np.nan
        self.animal_dorsal_area = np.nan
        self.eye_area = np.nan
        self.animal_length = np.nan
        self.pedestal = np.nan
        self.binned_pedestal_data = []
        self.pedestal_area = np.nan
        
        self.pedestal_max_height = np.nan
        self.pedestal_area = np.nan

        self.pedestal_window_max_height = np.nan
        self.pedestal_window_area = np.nan

        self.animal_x_center = np.nan
        self.animal_y_center = np.nan
        self.animal_major = np.nan
        self.animal_minor = np.nan
        self.animal_theta = np.nan
        
        self.eye_x_center = np.nan
        self.eye_y_center = np.nan
        self.eye_major = np.nan
        self.eye_minor = np.nan
        self.eye_theta = np.nan

        # these are directional vectors of anatomical direction starting at origin
        
        self.anterior = np.nan
        self.posterior = np.nan
        self.dorsal = np.nan
        self.ventral = np.nan

        self.flip = False
        
        # these are directional vectors of anatomical direction starting at animal center
        self.ant_vec = np.nan
        self.pos_vec = np.nan
        self.dor_vec = np.nan
        self.ven_vec = np.nan

        # endpoints for masking antenna
        self.ventral_mask_endpoints = np.nan
        self.dorsal_mask_endpoints = np.nan
        self.anterior_mask_endpoints = np.nan

        # coordinates of anatomical points on the animal

        self.eye_dorsal = np.nan
        self.eye_ventral = np.nan
        self.head = np.nan
        self.tail = np.nan
        self.tail_dorsal = np.nan
        self.tail_base = np.nan
        self.tail_tip = np.nan
        self.tail_spine_length = np.nan
        self.dorsal_point = np.nan
        
        self.dorsal_edge = np.nan
        self.checkpoints = np.nan

        self.peak = np.nan
        self.deyecenter_pedestalmax = np.nan
        self.poly_coeff = np.nan
        self.res = np.nan
        self.dorsal_residual = np.nan
        
        # quality check flags
        self.automated_PF = "P"
        self.automated_PF_reason = ''

        self.analyzed = False
        self.accepted = 0
        self.modified = 0
        self.modification_notes = ""
        self.modifier = ""

        # parameters for analysis are read from a file provided by the user and set
        self.count_animal_pixels_blur = kwargs['count_animal_pixels_blur']
        self.count_animal_pixels_n = kwargs['count_animal_pixels_n']
        self.count_animal_pixels_cc_threshold = kwargs['count_animal_pixels_cc_threshold']
        
        self.canny_minval = kwargs['canny_minval']
        self.canny_maxval = kwargs['canny_maxval']

        self.find_eye_blur = kwargs['find_eye_blur']
    
        self.mask_antenna_blur = kwargs['mask_antenna_blur']
        self.edge_pixel_distance_threshold_multiplier = kwargs['edge_pixel_distance_threshold_multiplier']
        self.mask_antenna_coronal_tilt = kwargs['mask_antenna_coronal_tilt']
        self.mask_antenna_anterior_tilt = kwargs['mask_antenna_anterior_tilt']
        self.mask_antenna_posterior_tilt = kwargs['mask_antenna_posterior_tilt']

        self.fit_ellipse_chi2 = kwargs['fit_ellipse_chi2']

        self.find_head_blur = kwargs['find_head_blur']

        self.find_tail_blur = kwargs['find_tail_blur']

        self.dorsal_edge_blur = kwargs['dorsal_edge_blur']
        
        self.analyze_pedestal_moving_avg_window = kwargs['analyze_pedestal_moving_avg_window']
        self.analyze_pedestal_percentile = kwargs['analyze_pedestal_percentile']
        self.analyze_pedestal_polyfit_degree = kwargs['analyze_pedestal_polyfit_degree']
        self.pedestal_n = kwargs['pedestal_n']

    def dist(self,x,y):

        # input: n-dimensional vectors x and y (can be list, tuple or numpy array)
        # output: euclediean distance between vectors x and y

        # returns euclidean distance between two vectors
        x = np.array(x)
        y = np.array(y)
        return np.linalg.norm(x-y)
    
    def fit_ellipse(self, im, chi_2):
        
        # fit an ellipse to the animal pixels
        #
        # input: image
        # returns centroid, major and minor axes and tilt angle of the ellipse

        try:
            #convert segmentation image to list of points
            points = np.array(np.where(im))
            n = points.shape[1]
            
            #calculate mean
            mu = np.mean(points,axis=1)
            x_center = mu[0]
            y_center = mu[1]

            #calculate covariance matrix
            z = points.T - mu*np.ones(points.shape).T
            cov = np.dot(z.T,z)/n
            
            #eigenvalues and eigenvectors of covariance matrix correspond
            #to length of major/minor axes of ellipse
            w,v = np.linalg.eig(cov)

            #calculate 90% confidence intervals using eigenvalues to find length of axes
            maj = np.argmax(w)
            minor = np.argmin(w)
            
            major_l = 2*np.sqrt(chi_2*w[maj])
            minor_l = 2*np.sqrt(chi_2*w[minor])

            # calculate angle of largest eigenvector towards the x-axis to get theta relative to x-axis
            v = v[minor]
            theta = np.arctan(v[1]/v[0])
            
            return x_center, y_center, major_l, minor_l, theta

        except Exception as e:
            print "Error fitting ellipse: " + str(e)
            return

    def find_eye(self, im):
        # input: segmentation image, im (numpy array)
        #
        # detect pixels in the image that belong to the eye of the animal
        
        hc = self.high_contrast(im)
        edges = cv2.Canny(np.array(255*gaussian(hc, self.find_eye_blur), dtype=np.uint8),self.canny_minval,self.canny_maxval)/255

        # initialize eye center
        eye_im = np.where((im < np.percentile(im, 0.025)))
        ex, ey = np.median(eye_im, axis=1)

        # initialize list of points whose neighbors should be checked to a list
        to_check = [(int(ex), int(ey))]

        # initialize two empty lists for pixels that have been checked
        # and pixels assigned as belonging to eye
        checked = []
        eye = []

        count = 0
        
        # Flood-fill algorithm that finds the eye pixels

        while len(to_check)>0:
            
            # grab the first point in the to_check list
            pt = to_check[0]

            # make sure the point and none of its neighbors belong to the eye edge

            if (edges[pt[0]-1, pt[1]] == 0) and (edges[pt[0]+1, pt[1]] == 0) and (edges[pt[0], pt[1]-1] == 0) and (edges[pt[0], pt[1]+1] == 0):
                
                count +=1 # keep track of how big the eye is
                eye.append((pt[0], pt[1])) # add the current point to list of eye points

                # go through all of the neighbor points and add them to the list of points to be checked
                # if they are not already in the list
                if ((pt[0]-1, pt[1]) not in checked) and ((pt[0]-1, pt[1]) not in to_check):
                        to_check.append((pt[0]-1, pt[1]))
                if ((pt[0]+1, pt[1]) not in checked) and ((pt[0]+1, pt[1]) not in to_check):
                        to_check.append((pt[0]+1, pt[1]))
                if ((pt[0], pt[1]-1) not in checked) and ((pt[0], pt[1]-1) not in to_check):
                        to_check.append((pt[0], pt[1]-1))
                if ((pt[0], pt[1]+1) not in checked) and ((pt[0], pt[1]+1) not in to_check):
                        to_check.append((pt[0], pt[1]+1))
            
            # remove the current point from the to_check list and add it to the list of checked points
            checked.append(to_check.pop(0))
        

        self.eye_pts = np.array(eye)

        # calculate the centroid of the eye and store the area
        try:
            self.eye_x_center, self.eye_y_center = np.mean(np.array(eye), axis=0)
            self.eye_area = count
        except (TypeError, IndexError):
            # if no eye points are found, smooth the image with a higher sigma and retry
            self.find_eye_blur+=0.25
            self.find_eye(im)
        

    def count_animal_pixels(self, im):

        # This function estimates the size of the animal
        
        # get binary edge mask of the animal
        hc = self.high_contrast(im)
        edges = cv2.Canny(np.array(255*gaussian(hc, self.count_animal_pixels_blur), dtype=np.uint8),self.canny_minval,self.canny_maxval)/255

        # retrieve landmarks
        cx, cy = self.animal_x_center, self.animal_y_center

        hx1, hy1 = self.ventral_mask_endpoints[0]
        vx, vy = self.ventral_mask_endpoints[1]
        
        hx2, hy2 = self.dorsal_mask_endpoints[0]
        dx, dy = self.dorsal_mask_endpoints[1]

        topx1, topy1 = self.anterior_mask_endpoints[0]
        topx2, topy2 = self.anterior_mask_endpoints[1]

        maskx = []
        masky = []

        # index edge pixels
        idxx, idxy = np.where(edges)

        # mask edge image to get rid of antenna
        for i in np.arange(len(idxx)):
            if self.intersect([cx, cy, idxx[i], idxy[i]], [hx1, hy1, vx, vy]):
                maskx.append(idxx[i])
                masky.append(idxy[i])

            if self.intersect([cx, cy, idxx[i], idxy[i]], [hx2, hy2, dx, dy]):
                maskx.append(idxx[i])
                masky.append(idxy[i])

            if self.intersect([cx, cy, idxx[i], idxy[i]], [topx1, topy1, topx2,topy2]):
                maskx.append(idxx[i])
                masky.append(idxy[i])
        
        edges[[maskx, masky]] = 0
        idxx, idxy = np.where(edges)

        r = 2*self.dist((cx, cy), self.anterior)
        s = np.linspace(0, 2*np.pi, self.count_animal_pixels_n)
        
        x = cx + int(r)*np.sin(s)
        y = cy + int(r)*np.cos(s)
        
        pts = []

        for i in np.arange(len(s)):
            
            p1 = (cx, cy)
            p2 = (x[i], y[i])

            m = (p2[1] - p1[1])/(p2[0] - p1[0])
            b = p1[1] - m*p1[0]

            diff = np.power(idxy - (m*idxx + b), 2)
            near_line_idx = np.where(diff < np.percentile(diff, 2))
            near_line = np.transpose(np.vstack([idxx[near_line_idx], idxy[near_line_idx]]))

            try:
                j = np.argmin(np.linalg.norm(near_line-p2, axis=1))
                pts.append((near_line[j,1], near_line[j,0]))
            except ValueError:
                pass

        cc = [[]]
        idx = 0
        connected = False
        pts = np.array(pts)

        # create connected component object by determining whether or not points are within
        # a distance threshold of neighboring points
        for i in xrange(1, pts.shape[0]-1):
            if (self.dist(pts[i,:], pts[i-1,:]) < self.count_animal_pixels_cc_threshold) or (self.dist(pts[i+1,:],pts[i,:]) < self.count_animal_pixels_cc_threshold):
                try:
                    cc[idx].append(pts[i,:])
                    connected = True
                except IndexError:
                    cc.append([])
                    cc[idx].append(pts[i,:])
                    connected = True
            else:
                try:
                    if len(cc[idx]) < 4:
                        cc.pop(idx)
                    elif connected:
                        idx += 1
                        connected = False
                except IndexError:
                    pass
        
        pts = np.vstack(cc) 
        self.whole_animal_points = pts
        self.animal_area = self.area(pts[:,0], pts[:,1])

    def get_animal_length(self):
        
        # returns the length of the animal by calculating the distance
        # between the head and the tail

        self.animal_length = self.dist(self.head, self.tail)
    
    def get_animal_dorsal_area(self):

        # calculates the area of the dorsal region of the animal

        self.animal_dorsal_area = np.abs(self.area(self.dorsal_edge[:,0], self.dorsal_edge[:,1]))

    def find_features(self, im):

        # driver script that calls all of the landmark-finding functions (except eye)
        #
        # input: im - raw gray-scale image

        ex, ey = self.eye_x_center, self.eye_y_center

        hc = self.high_contrast(im)

        edge_image = cv2.Canny(np.array(255*gaussian(hc, self.mask_antenna_blur), dtype=np.uint8),self.canny_minval,self.canny_maxval)/255
        edge_copy = edge_image.copy()

        edge_index = np.transpose(np.where(edge_image))
        
        # first estimate of animal center
        self.animal_x_center, self.animal_y_center = np.mean(edge_index, axis=0)
        cx, cy = self.animal_x_center, self.animal_y_center

        # distances from edge pixels to center
        d_edgepixel_center = np.linalg.norm(edge_index - np.array([cx, cy]), axis=1)
        
        # filter out edge_index for edge pixels that are too far from the animal
        dhalf_length = np.linalg.norm(np.array((cx,cy)) - np.array((ex, ey)))
        edge_index = edge_index[np.transpose(np.where(d_edgepixel_center < self.edge_pixel_distance_threshold_multiplier*dhalf_length)[0])]
        
        # tail_tip is most likely to be the max dot product of vector from center to edge_pixel and vector from eye_center to center
        tail_tip_index = np.argmax(np.dot(edge_index-np.array([cx,cy]), np.array([cx-ex, cy-ey]))) 
        self.tail_tip = tuple(edge_index[tail_tip_index, :])
        tx, ty = self.tail_tip
        
        cx, cy = (tx + ex)/2, (ty + ey)/2

        hx1, hy1 = 1.2*(ex - cx) + cx, 1.2*(ey - cy) + cy

        # calculate appropriate masking lines for antenna
        vd1 = cx + self.mask_antenna_coronal_tilt*(hy1 - cy), cy + self.mask_antenna_coronal_tilt*(cx - hx1)
        vd2 = cx - self.mask_antenna_coronal_tilt*(hy1 - cy), cy - self.mask_antenna_coronal_tilt*(cx - hx1)

        hx2, hy2 = 1.125*(ex - cx) + cx, 1.125*(ey - cy) + cy
        top1 = hx2 + self.mask_antenna_anterior_tilt*(ey - hy2), hy2 + self.mask_antenna_anterior_tilt*(hx2 - ex)
        top2 = hx2 - self.mask_antenna_anterior_tilt*(ey - hy2), hy2 - self.mask_antenna_anterior_tilt*(hx2 - ex)
      
        tail = 0.4*cx + 0.6*self.tail_tip[0], 0.4*cy + 0.6*self.tail_tip[1]
        bot1 = tail[0] + self.mask_antenna_posterior_tilt*(self.tail_tip[1] - tail[1]), tail[1] + self.mask_antenna_posterior_tilt*(self.tail_tip[0] - tail[0])
        bot2 = tail[0] - self.mask_antenna_posterior_tilt*(self.tail_tip[1] - tail[1]), tail[1] - self.mask_antenna_posterior_tilt*(self.tail_tip[0] - tail[0])
       
        # mask antenna
        edge_copy = self.mask_antenna(edge_copy, (cx, cy), a=[hx1, hy1, vd1[0], vd1[1]], b=[hx1, hy1, vd2[0], vd2[1]], c=[top1[0], top1[1], top2[0], top2[1]])
        # update anatomical directions after masking antenna
        self.get_anatomical_directions(edge_copy)
        
        if self.flip:
            self.flip_dorsal_ventral()

        if self.dist( self.ventral, vd1 ) < self.dist( self.ventral, vd2 ):
            self.ventral_mask_endpoints = ((hx1, hy1), vd1)
        else:
            self.ventral_mask_endpoints = ((hx1, hy1), vd2)
        
        hx, hy = self.ventral_mask_endpoints[0]
        vx, vy = self.ventral_mask_endpoints[1]

        m = (hy - vy)/(hx - vx)
        b = hy - m*hx

        x2 = (cx + m*(cy-b))/(1 + m**2)
        y2 = (m*cx + (m**2)*cy + b)/(1 + m**2)

        shift = np.array( (x2 - cx, y2 - cy) )

        self.dorsal_mask_endpoints = [hx - 1.4*shift[0], hy - 1.4*shift[1], self.tail_tip[0], self.tail_tip[1]]
        self.ventral_mask_endpoints = [self.ventral_mask_endpoints[0][0] + 0.05*shift[0],
            self.ventral_mask_endpoints[0][1] + 0.05*shift[1],
            self.ventral_mask_endpoints[1][0] + 0.05*shift[0],
            self.ventral_mask_endpoints[1][1] + 0.05*shift[1]]
        self.anterior_mask_endpoints = [top1[0], top1[1], top2[0], top2[1]]
        
        self.edge_copy = edge_copy

    def flip_dorsal_ventral(self):

        # hard swap of the dorsal and ventral directions. useful
        # when they have been mislabeled
        
        tmp = self.ventral
        self.ventral = self.dorsal
        self.dorsal = tmp

    def mask_antenna(self, edge, center, **kwargs):

        # the antenna of the daphnia can be in a wide range of
        # random positions. this function masks the antenna to
        # downstream errors
        #
        # input: edge - binary edge image (numpy array)
        #        center - centroid of the animal
        #        
        # returns: edge - binary edge image with antenna masked

        cx, cy = center

        edges_x = np.where(edge)[0]
        edges_y = np.where(edge)[1]
 
        mask_x = []
        mask_y = []
         
        for i in xrange(len(edges_x)):
            for key, value in kwargs.iteritems():
                try:
                    # if the line segment drawn from the animal center to the current edge pixel intersects with 
                    # a mask line (see params), save it to the list of edges to be removed
                    if self.intersect([cx, cy, edges_x[i], edges_y[i]], [value[0], value[1], value[2], value[3]]):
                        mask_x.append(edges_x[i])
                        mask_y.append(edges_y[i])
                except TypeError:
                    continue
        # set masked pixels to 0
        edge[[mask_x, mask_y]] = 0
        return edge

    def get_anatomical_directions(self, im, flag="animal"):

        # function for determining the anterior, posterior, dorsal and ventral
        # directions of the animal relative to the body centroid using the 
        # eye and tip of the tail as landmarks

        # input: im - binary edge image (numpy array)

        x, y, major, minor, theta = self.fit_ellipse(im, self.fit_ellipse_chi2)
        self.animal_x_center, self.animal_y_center, self.animal_major, self.animal_minor, self.animal_theta = x, y, major, minor, theta
        
        major_vertex_1 = (x - 0.5*major*np.sin(theta), y - 0.5*major*np.cos(theta))
        major_vertex_2 = (x + 0.5*major*np.sin(theta), y + 0.5*major*np.cos(theta))

        minor_vertex_1 = (x + 0.5*minor*np.cos(theta), y - 0.5*minor*np.sin(theta))
        minor_vertex_2 = (x - 0.5*minor*np.cos(theta), y + 0.5*minor*np.sin(theta)) 

        if self.dist( major_vertex_1, (self.eye_x_center, self.eye_y_center)) < self.dist(major_vertex_2, (self.eye_x_center, self.eye_y_center)):
            self.anterior = major_vertex_1
            self.posterior = major_vertex_2
        else:
            self.anterior = major_vertex_2
            self.posterior = major_vertex_1

        if self.dist( minor_vertex_1, self.tail_tip ) < self.dist(minor_vertex_2, self.tail_tip):
            self.dorsal = minor_vertex_1
            self.ventral = minor_vertex_2
        else:
            self.dorsal = minor_vertex_2
            self.ventral = minor_vertex_1
        

    def get_orientation_vectors(self):

        # uses the anatomical points and animal center to create a vector for each direction

        self.pos_vec = [self.animal_x_center - self.posterior[0], self.animal_y_center - self.posterior[1]]
        self.dor_vec = [self.animal_x_center - self.dorsal[0], self.animal_y_center - self.dorsal[1]]
        self.ven_vec = [self.animal_x_center - self.ventral[0], self.animal_y_center - self.ventral[1]]
        self.ant_vec = [self.animal_x_center - self.anterior[0], self.animal_y_center - self.anterior[1]]
       
    def eye_vertices(self):
        
        # finds and assigns eye pixel as ventral- and dorsal-most eye pixel

        ep = self.eye_pts
        ex, ey = self.eye_x_center, self.eye_y_center
        dx, dy = self.dor_vec
        vx, vy = self.ven_vec
        
        dorsal_target_x, dorsal_target_y = ex - dx, ey - dy
        ventral_target_x, ventral_target_y = ex - vx, ey - vy
        
        # calculate minimum distance between dorsal/ventral vectors originating at the eye center and all of the eye pixels

        self.eye_dorsal = tuple(ep[np.argmin(np.linalg.norm(ep - (dorsal_target_x, dorsal_target_y), axis=1)), :])
        self.eye_ventral = tuple(ep[np.argmin(np.linalg.norm(ep - (ventral_target_x, ventral_target_y), axis=1)), :])

    def find_head(self, im):

        # finds the head point, defined by the point of the head of the animal that falls on the line 
        # created by the tail landmark and the dorsal point of the eye
        
        # estimate tail position for now, and a better estimate will be made in get_dorsal_edge
        hc = self.high_contrast(im)
        edges = cv2.Canny(np.array(255*gaussian(hc, self.find_head_blur), dtype=np.uint8),self.canny_minval,self.canny_maxval)/255
        tx, ty = self.tail_tip
        target = 3*self.eye_ventral[0] - 2*self.eye_x_center, 3*self.eye_ventral[1] - 2*self.eye_y_center
        ex, ey = 0.5*tx + 0.5*target[0], 0.5*ty + 0.5*target[1]
        
        m = (ty - ey)/(tx - ex)
        
        x, y = np.linspace(tx, ex, 100), np.linspace(ty, ey, 100)

        d = self.dist((tx, ty), (ex, ey))/8

        for i in xrange(int(100)):
            
            p1, p2 = self.orth((x[i], y[i]), d, m, "both")

            if self.dist(self.ventral, p1) < self.dist(self.ventral, p2):
                start = p1
                end = p2
            else:
                start = p2
                end = p1

            e = self.find_edge2(edges, end, start)
            
            if e is not None:
                if self.dist(e, start) < self.dist(p1, p2)/5:
                    self.tail = tuple(e)
                    self.tail_dorsal = self.find_edge2(edges, start, end)
                    break

        if self.tail is None:
            self.tail = self.tail_tip
        
        edx, edy = self.eye_dorsal

        tx, ty = self.tail
        m = (ty - edy)/(tx - edx)
        b = edy - m*edx

        d = self.dist((edx, edy), (tx, ty))
        cx = edx - (-0.15*d*(edx - tx))/d
        cy = edy - (-0.15*d*(edy - ty))/d

        [topx1, topy1, topx2, topy2] = self.anterior_mask_endpoints

        if self.intersect((edx, edy, cx, cy), (topx1, topy1, topx2, topy2)):
            res = self.intersection((edx, edy, cx, cy), (topx1, topy1, topx2, topy2))
            p1 = res[0], res[1]
            p2 = edx, edy
        else:
            p1 = cx, cy
            p2 = edx, edy
        
        try:
            hx, hy = self.find_edge2(edges, p2, p1)
            self.head = hx, hy
        
        except TypeError:
            
            # if head edge can't be found, just estimate based on dorsal eye point
            self.head = (edx - (-0.05*d*(edx - tx))/d, edy - (-0.05*d*(edy - ty))/d)

    def find_tail(self, im):

        hc = self.high_contrast(im)
        edges = cv2.Canny(np.array(255*gaussian(hc, self.find_tail_blur), dtype=np.uint8),self.canny_minval,self.canny_maxval)/255

        tx, ty = self.tail_tip
        
        ventral_edge = self.traverse_ventral_edge(edges, self.tail_tip, self.anterior, np.array(self.tail_tip) - np.array(self.ven_vec))
        
        dorsal_edge = self.dorsal_edge

        old_diffs = np.min(np.linalg.norm(dorsal_edge - ventral_edge[0,:], axis=1))

        for i in np.arange(1, len(ventral_edge)):
            diffs = np.min(np.linalg.norm(dorsal_edge - ventral_edge[i,:], axis=1))
            if diffs - old_diffs > 2:
                self.tail = ventral_edge[i,:]
                self.tail_dorsal = dorsal_edge[np.argmin(np.linalg.norm(dorsal_edge - self.tail, axis=1)), :]
                break
            old_diffs = diffs
        self.tail = tuple(self.tail)
        self.tail_dorsal = tuple(self.tail_dorsal)
        self.tail_base = ((self.tail[0]+self.tail_dorsal[0])/2, (self.tail[1] + self.tail_dorsal[1])/2)
        self.get_tail_spine_length()

    def get_tail_spine_length(self):
        self.tail_spine_length = self.dist(self.tail_tip, self.tail)

    def initialize_dorsal_edge(self, im, dorsal_edge_blur = None, edges = None):

        if not dorsal_edge_blur:
            dorsal_edge_blur = self.dorsal_edge_blur
        
        hx, hy = self.head
        tx_d, ty_d = self.tail_dorsal
        cx, cy = self.animal_x_center, self.animal_y_center

        if edges is None:
            hc = self.high_contrast(im) 
            edges = cv2.Canny(np.array(255*gaussian(hc, dorsal_edge_blur), dtype=np.uint8), 0, 50)/255
            #edges = self.mask_antenna(edges, (cx, cy),
            #        dorsal=self.dorsal_mask_endpoints,
            #        ventral=self.ventral_mask_endpoints,
            #        anterior=self.anterior_mask_endpoints)
        
        self.edges = edges

        if (self.head[0] == self.tail_dorsal[0]):
            m = np.inf
        else:
            m,b = self.line_fit(self.head, self.tail_dorsal)

        d = self.dist((hx,hy), (self.dorsal_mask_endpoints[0], self.dorsal_mask_endpoints[1]))

        checkpoints = [self.head]
        counter=1

        for i in np.arange(0,1,0.1):
            mp = (1-i)*hx + i*tx_d, (1-i)*hy + i*ty_d
            x,y = self.orth(mp, d, m, flag="dorsal")
            p2 = self.find_edge2(edges, mp, (x,y))
            if p2 is not None:
                checkpoints.append(p2)
                counter+=1
    
        checkpoints.append(self.tail_dorsal)
        self.checkpoints = np.array(checkpoints)

    def prune_checkpoints(self):
        
        checkpoints = self.checkpoints
        prune = True

        while prune:
            prune = False
            prune_list = []
            for k in np.arange(1, len(checkpoints)-1):

                if (checkpoints[k-1][0] == checkpoints[k+1][0]):
                    m = np.inf
                    err = 0
                else:
                    m,b = self.line_fit(checkpoints[k-1], checkpoints[k+1])
                    x,y = checkpoints[k]
                    err = np.abs(b + m*x - y)/np.sqrt(1 + m**2)
                
                if err > 20:
                    prune = True
                    prune_list.append(k)
            
            mask = np.ones(len(checkpoints), dtype=bool)
            mask[prune_list] = False
            checkpoints = checkpoints[mask, :]
        
        self.checkpoints = np.vstack(checkpoints)

    def fit_dorsal_edge(self, im, dorsal_edge_blur = None, edges = None):
        
        if not dorsal_edge_blur:
            dorsal_edge_blur = self.dorsal_edge_blur

        cx, cy = self.animal_x_center, self.animal_y_center
        
        if edges is None:
            hc = self.high_contrast(im) 
            edges = cv2.Canny(np.array(255*gaussian(hc, dorsal_edge_blur), dtype=np.uint8), 0, 50)/255

        self.prune_checkpoints()
        checkpoints = self.checkpoints
        
        # for some reason, calling traverse_dorsal_edge in reverse on the head portion works better
        dorsal_edge = self.traverse_dorsal_edge(edges, np.array(checkpoints[1]), np.array(checkpoints[0]))[::-1]
        
        # traverse_dorsal_edge prunes the target checkpoint from the list (so there aren't duplicates in the final list),
        # but since traverse_dorsal_edge is being called in reverse for the head portion, we want to prune the starting
        # point and add the target instead
        dorsal_edge = np.vstack((np.array(checkpoints[0]), dorsal_edge))
        dorsal_edge = dorsal_edge[:-1,:] 
        
        for k in np.arange(1,len(checkpoints)-1):
       
            x,y = checkpoints[k,:]
            x_1, y_1 = checkpoints[k+1,:]

            dorsal_edge = np.vstack([dorsal_edge, self.traverse_dorsal_edge(edges, checkpoints[k,:], checkpoints[k+1,:])])
        
        self.dorsal_edge = np.vstack([dorsal_edge, self.traverse_dorsal_edge(edges, self.tail_dorsal, self.tail_tip)])    

    def remove_tail_spine(self):
        
        self.dorsal_edge = self.dorsal_edge[0:np.argmin(np.linalg.norm(self.dorsal_edge - self.tail_dorsal, axis=1)), :]
        self.dorsal_edge = self.interpolate(self.dorsal_edge)
        self.checkpoints[-1,:] = self.dorsal_edge[-1,:]
    
    def line_fit(self,p1,p2):
        
        # returns slope and y-intercept of line between two points
        m = (p2[1]-p1[1])/(p2[0]-p1[0])

        return m, p2[1] - m*p2[0]

    def orth(self, p, d, m, flag="center"):

        if flag == "center":
            cx, cy = self.animal_x_center, self.animal_y_center
        elif flag == "dorsal":
            cx, cy = self.ventral

        x1 = p[0] + np.sqrt((d**2)/(1 + 1/(m**2)))
        y1 = p[1] - (1/m)*(x1 - p[0])

        x2 = p[0] - np.sqrt((d**2)/(1 + 1/(m**2)))
        y2 = p[1] - (1/m)*(x2 - p[0])
        
        if flag == "both":
            return (x1, y1), (x2, y2)

        if self.dist((x1, y1), (cx, cy)) < self.dist((x2, y2), (cx, cy)):
            return x2, y2
        else:
            return x1, y1

    def traverse_ventral_edge(self, edges, current, target, ventral, n=200):
        
        ventral_edge = [list(current)]

        target_vector = np.array(target) - np.array(current)
        target_vector = self.norm_vec(target_vector)

        ventral_vector = np.array(ventral) - np.array(current)
        ventral_vector = self.norm_vec(ventral_vector)

        window = 1
        w,h = edges.shape

        for i in np.arange(n):
            
            idx = self.index_on_pixels(edges[int(np.max([0, current[0]-window])):int(np.min([w, current[0]+window+1])),
                int(np.max([0, current[1]-window])):int(np.min([h, current[1]+window+1]))]) - (window,window)
            idx = idx[~np.all(idx == 0, axis=1)]

            try:
                nxt = current + idx[np.argmax(np.dot(idx, target_vector) + np.dot(idx, ventral_vector))]

                if (list(nxt) in ventral_edge) or (self.dist(nxt, target) > self.dist(current, target)):
                    raise(ValueError)
                else:
                    current = nxt
                    target_vector = np.array(target) - np.array(current)
                    target_vector = self.norm_vec(target_vector)

                    ventral_vector = np.array(ventral) - np.array(current)
                    ventral_vector = self.norm_vec(ventral_vector)

                    ventral_edge.append(list(current))
                    window=1
            except ValueError:
                 window += 1

        return np.vstack(ventral_edge)

    def traverse_dorsal_edge(self, edges, current,target):

        idx = self.index_on_pixels(edges)

        cx, cy = self.animal_x_center, self.animal_y_center

        dorsal_edge = [list(current)]
        
        target_vector = np.array(target) - np.array(current)
        target_vector = self.norm_vec(target_vector)
        
        dorsal_vector = np.array(current) - np.array((cx, cy))
        dorsal_vector = self.norm_vec(dorsal_vector)

        nxt_vector = target_vector

        window=1
        w,h = edges.shape

        while (self.dist(current, target) > 2) and (window < 10):

            idx = self.index_on_pixels(edges[ int(np.max([0,current[0]-window])):int(np.min([w,current[0]+window+1])),
                int(np.max([0, current[1]-window])):int(np.min([h,current[1]+window+1]))]) - (window,window)
            idx = idx[~np.all(idx == 0, axis=1)]

            try:
                
                nxt = current + idx[np.argmax(np.dot(idx, dorsal_vector) + np.dot(idx, target_vector) + np.dot(idx, nxt_vector))]

                if (list(nxt) in dorsal_edge) or (self.dist(nxt, target) > self.dist(current, target)):
                    raise(ValueError)
                else:

                    target_vector = np.array(target) - np.array(current)
                    target_vector = self.norm_vec(target_vector)

		    dorsal_vector = np.array(current) - np.array((cx, cy))
		    dorsal_vector = self.norm_vec(dorsal_vector)

		    nxt_vector = np.array(nxt) - np.array(current)
		    nxt_vector = self.norm_vec(nxt_vector)
		    
		    current = nxt
		    dorsal_edge.append(list(current))
		    window=1
            
            except ValueError:
                window += 1
        if (list(target) in dorsal_edge):
            dorsal_edge.remove(list(target))

        return dorsal_edge
    
    def qscore(self):

        # returns the 'qscore' of the 

        # if the tail and head happen to have the same x-position, we 
        # get a divide-by-zero error. To avoid this, rotate 90 degrees.
        
        if (self.head[0] == self.tail_dorsal[0]):
            head = self.rotate([0,0], np.array(self.head), np.pi/4)
            tail_dorsal = self.rotate([0,0], np.array(self.tail_dorsal), np.pi/4)
            d = self.dorsal_edge
            d = np.transpose(np.vstack(self.rotate([0,0], d, np.pi/4)))
        else:
            head = self.head
            tail_dorsal = self.tail_dorsal
            d = self.dorsal_edge
        
        m1, b1 = self.line_fit(head, tail_dorsal) 
        self.q = np.abs(b1 + m1*d[:,0] -d[:,1])/np.sqrt(1 + m1**2)
                
        m2 = -1/m1
        b2 = d[:,1] - m2*d[:,0]

        x = (b2 - b1)/(m1 - m2)
        y = m1*x + b1
        
        self.qi = np.linalg.norm(np.transpose(np.vstack([x,y])) - head, axis=1)/self.dist(head, tail_dorsal)
        self.check_dorsal_edge_fit()

    def index_on_pixels(self,a):
        
        # returns the indices of positive pixels in a binary image

        return np.transpose(np.vstack(np.where(a)))
    
    def norm_vec(self, v):

        # normalizes vectors by dividing by the max value of the vector

        return v/np.max(np.abs(v))

    def get_pedestal_max_height(self, data):

        # returns the height of the pedestal at its maximum
        #
        # input: data - x,y positions of all pedestal points
        
        self.pedestal_max_height = np.max(data[:,1])

    def get_pedestal_area(self, data):

        # returns the size of the pedestal in pixels
        #
        # input: data - x,y positions of all pedestal points
        
        self.pedestal_area = np.sum(0.5*(self.dist(self.head, self.dorsal_point)/400)*(data[1:][:,0] - data[0:-1][:,0])*(data[1:][:,1] + data[0:-1][:,1]))
        
    def get_pedestal_theta(self, data, n=200):

        # returns angle of pedestal relative to body
        #
        # input: data - x,y positions of all pedestal points
        
        x = (n - data[np.argmax(data[:,1]), 0]) * self.dist(self.head, self.dorsal_point)/400
        hyp = self.dist((n,0), (x, np.max(data[:,1])))
        self.pedestal_theta = np.arcsin((n - x)/hyp)*(180/np.pi)

    def find_edge2(self, edges, p1, p2):

        # finds closest edge-pixel to a line segment

        # input: edges - binary mask of all edges
        #        p1, p2 - endpoints of line segment

        idxx, idxy = np.where(edges)
        
        # if there are no edges in the image (e.g. blur is too high), exit
        if (len(idxx) == 0) or (len(idxy) == 0):
            return

        # calculate slope and intercept of line that p1 and p2 lie on
        m = (p2[1] - p1[1])/(p2[0] - p1[0])
        b = p1[1] - m*p1[0]

        diff = np.power(idxy - (m*idxx + b), 2)
        near_line_idx = np.where(diff < np.percentile(diff, 2))
        near_line = np.transpose(np.vstack([idxx[near_line_idx], idxy[near_line_idx]]))
        
        near_line = near_line[np.linalg.norm(near_line - p2, axis=1) - self.dist(p1, p2) < 0]

        try:
            j = np.argmin(np.linalg.norm(near_line-p2, axis=1))
            return near_line[j,:]
        except ValueError:
            return

    def high_contrast(self, im):

        # applies contrast filter with clipLimit=2.0, tileGridSize (8,8)
        #
        # input: image (numpy array)
        #
        # output: high contrast image (numpy array)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(im)
    
    def area(self, x, y):

        # shoelace method for calculating area of a polygon
        #
        # input: two lists of x- and y-coordinates
        #
        # return: area of polygon in pixels (float)

        x = np.asanyarray(x)
        y = np.asanyarray(y)
        n = len(x)

        up = np.arange(-n+1, 1)
        down = np.arange(-1, n-1)

        return (x * (y.take(up) - y.take(down))).sum() / 2
    
    def intersect(self, s1, s2):

        # check if two line segments intersect

        # input: s1 - pair of x,y coordinates for line #1
        #        s2 - pair of x,y coordinates for line #2

        # output: boolean indicating whether lines intersect


        x1, y1, x2, y2 = s1
        x3, y3, x4, y4 = s2

        if (max([x1, x2]) < min([x3, x4])): return False

        m1 = (y1 - y2)/(x1 - x2)
        m2 = (y3 - y4)/(x3 - x4)

        if (m1 == m2): return False
        
        b1 = y1 - m1*x1
        b2 = y3 - m2*x3

        xa = (b2 - b1) / (m1 - m2)

        if ( (xa < max( [min([x1, x2]), min([x3, x4])] )) or (xa > min( [max([x1, x2]), max([x3, x4])] )) ):
            return False

        return True

    def intersection(self, s1, s2):

        # returns the point of intersection for two line segments

        if not self.intersect(s1, s2): return np.nan

        x1, y1, x2, y2 = s1
        x3, y3, x4, y4 = s2

        return (((x1*y2-y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4))/
                ((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)),
                ((x1*y2-y1*x2)*(y3-y4) - (y1-y2)*(x3*y4-y3*x4))/
                ((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)))

    def norm(self, x):
        
        # normalizes a vector by scaling it to be between 0-1

        return (x - np.min(x)) / (np.max(x) - np.min(x))
    
    def check_dorsal_edge_fit(self):

        # fits a 4th order polynomial to the dorsal edge and checks the error
        
        poly_coeff, residual, _, _, _ = np.polyfit(self.qi, self.q, 4, full=True)
        
        self.dorsal_residual = residual[0]

        if self.dorsal_residual > 20000:
            self.automated_PF = 'F'
            self.automated_PF_reason = 'high dorsal residual error'

    def analyze_pedestal(self):

        # script for taking spatially-normalized pedestal points and
        # analyzing them. the script finds the peak of the pedestal,
        # fits a polynomial to the points and finds the difference in
        # height and poly fit.
    
        qi = np.array(self.qi)
        q = np.array(self.q)
        
        ipeak = np.argmax(q)
        self.peak = q[ipeak]
        
        threshold = np.percentile(q, self.analyze_pedestal_percentile)

        poly_train = np.where(q<threshold)
        roi = np.where(q >= threshold)
        
        X = np.transpose(np.vstack([ qi[poly_train], q[poly_train] ]))

        # fit polynomial to pedestal points
        self.poly_coeff, res, _, _, _ = np.polyfit(X[:,0], X[:,1], self.analyze_pedestal_polyfit_degree, full=True)
        self.res = res[0]

        poly = np.poly1d(self.poly_coeff)
        
        # get difference between y-coordinates and poly fit coordinates to
        # estimate the growth of the pedestal
        yy = poly(qi)
        diff = q - yy
          
        # calculate stats on pedestal
        self.calc_pedestal_area(qi[roi], diff[roi])
        self.pedestal_max_height = np.max(diff[roi])
        self.get_deye_pedestalmax()
        
    def rotate(self, origin, points, phi):

        ox, oy = origin
        try:
            px, py = points[:,0], points[:,1]
        except IndexError:
            px, py = points
        qx = ox + np.cos(phi) * (px - ox) - np.sin(phi) * (py - oy)
        qy = oy + np.sin(phi) * (px - ox) + np.cos(phi) * (py - oy)

        return qx, qy

    def get_deye_pedestalmax(self):

        self.deyecenter_pedestalmax = self.dist((self.eye_x_center, self.eye_y_center), self.peak)

    def flip(self, p, ip):
        
        x1, y1 = p[0, :]
        x2, y2 = p[p.shape[0]-1, :]
        ex, ey = self.eye_x_center, self.eye_y_center

        if self.dist((x1, y1), (ex, ey)) > self.dist((x2, y2), (ex, ey)):
            ip = np.max(ip) - ip
            return np.flip(p, axis = 0), ip[::-1]
        
    def calc_pedestal_area(self, x, y):
        
        # calculates the area of pedestal

        self.pedestal_area = np.abs(np.sum((y[1:] + y[:-1])*(x[1:] - x[:-1])/2))

    def interpolate(self, dorsal_edge):

        p = dorsal_edge
        diff = np.linalg.norm(p[1:,:] - p[0:-1,:], axis=1)
        biggest_gaps = np.where(diff == np.max(diff))[0]
        
        while (p.shape[0] < self.pedestal_n) and (len(biggest_gaps) > 0):
 
            i = random.choice(biggest_gaps)
            biggest_gaps[np.where(biggest_gaps==i)[0][0]+1:] += 1
            biggest_gaps = np.delete(biggest_gaps, np.where(biggest_gaps == i))

            new_point = (p[i] + p[i+1])/2
            p = np.vstack([p[:i+1, :], new_point, p[i+1:,:]])

            if len(biggest_gaps) == 0:
                diff = np.linalg.norm(p[1:,:] - p[0:-1,:], axis=1)
                try:
                    biggest_gaps = np.where(diff == np.max(diff))[0]
                except ValueError:
                    biggest_gaps = []
        return p
