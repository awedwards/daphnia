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
    
    def __init__(self, filepath):
        
        self.filepath = filepath

        self.animal_area = np.nan
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
        
        # these are directional vectors of anatomical direction starting at animal center
        self.ant_vec = np.nan
        self.pos_vec = np.nan
        self.dor_vec = np.nan
        self.ven_vec = np.nan

        # endpoints for masking antenna
        self.ventral_mask_endpoints = np.nan
        self.dorsal_mask_endpoints = np.nan
        self.anterior_mask_endpoints = np.nan

        # these are actual points on the animal

        self.eye_dorsal = np.nan
        self.eye_ventral = np.nan
        self.head = np.nan
        self.tail = np.nan
        self.tail_dorsal = np.nan
        self.tail_base = np.nan
        self.tail_tip = np.nan
        self.tail_spine_length = np.nan
        self.dorsal_point = np.nan

        self.peak = np.nan
        self.deyecenter_pedestalmax = np.nan
        self.poly_coeff = np.nan
        self.res = np.nan

        # quality check flags
        self.automated_PF = "U"
        self.automated_PF_reason = ''
        self.manual_PF = "U"
        self.manual_PF_reason = ''
        self.manual_PF_curator = ''

        self.analyzed = False

    def dist(self,x,y):

        # returns euclidean distance between two vectors
        x = np.array(x)
        y = np.array(y)
        return np.linalg.norm(x-y)
    
    def fit_ellipse(self, im, chi_2):
        
        # fit an ellipse to the animal pixels

        try:
            # input: segmentation image
            # return xcenter,ycenter,major_axis_length,minor_axis_length,theta

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

    def find_eye(self, im, find_eye_blur=0.5, canny_minval=0, canny_maxval=50, **kwargs):

        hc = self.high_contrast(im)
        edges = cv2.Canny(np.array(255*gaussian(hc, find_eye_blur), dtype=np.uint8), canny_minval, canny_maxval)/255

        # initialize eye center
        eye_im = np.where((im < np.percentile(im, 0.025)))
        ex, ey = np.median(eye_im, axis=1)

        to_check = [(int(ex), int(ey))]
        checked = []
        eye = []

        count = 0

        while len(to_check)>0:
            pt = to_check[0]
            if (edges[pt[0]-1, pt[1]] == 0) and (edges[pt[0]+1, pt[1]] == 0) and (edges[pt[0], pt[1]-1] == 0) and (edges[pt[0], pt[1]+1] == 0):
                count +=1
                eye.append((pt[0], pt[1]))
                if ((pt[0]-1, pt[1]) not in checked) and ((pt[0]-1, pt[1]) not in to_check):
                        to_check.append((pt[0]-1, pt[1]))
                if ((pt[0]+1, pt[1]) not in checked) and ((pt[0]+1, pt[1]) not in to_check):
                        to_check.append((pt[0]+1, pt[1]))
                if ((pt[0], pt[1]-1) not in checked) and ((pt[0], pt[1]-1) not in to_check):
                        to_check.append((pt[0], pt[1]-1))
                if ((pt[0], pt[1]+1) not in checked) and ((pt[0], pt[1]+1) not in to_check):
                        to_check.append((pt[0], pt[1]+1))
            
            checked.append(to_check.pop(0))
        

        self.eye_pts = np.array(eye)
        try:
            self.eye_x_center, self.eye_y_center = np.mean(np.array(eye), axis=0)
            self.eye_area = count
        except (TypeError, IndexError):
            self.find_eye(im, find_eye_blur=find_eye_blur+0.25)
    
    def count_animal_pixels(self, im, count_animal_pixels_blur=1.0, count_animal_pixels_n=100, count_animal_pixels_cc_threshold=10, canny_minval=0, canny_maxval=50, **kwargs):
        
        hc = self.high_contrast(im)
        edges = cv2.Canny(np.array(255*gaussian(hc, count_animal_pixels_blur), dtype=np.uint8), canny_minval, canny_maxval)/255

        cx, cy = self.animal_x_center, self.animal_y_center

        hx1, hy1 = self.ventral_mask_endpoints[0]
        vx, vy = self.ventral_mask_endpoints[1]
        
        hx2, hy2 = self.dorsal_mask_endpoints[0]
        dx, dy = self.dorsal_mask_endpoints[1]

        topx1, topy1 = self.anterior_mask_endpoints[0]
        topx2, topy2 = self.anterior_mask_endpoints[1]

        
        maskx = []
        masky = []

        idxx, idxy = np.where(edges)
        

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
        s = np.linspace(0, 2*np.pi, count_animal_pixels_n)
        
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

        for i in xrange(1, pts.shape[0]-1):
            if (self.dist(pts[i,:], pts[i-1,:]) < count_animal_pixels_cc_threshold) or (self.dist(pts[i+1,:],pts[i,:]) < count_animal_pixels_cc_threshold):
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

        self.animal_length = self.dist(self.head, self.tail)

    def find_features(self, im, mask_antenna_blur=1.5,
            mask_antenna_cnx_comp_threshold=150,
            mask_antenna_coronal_tilt = 0.7,
            mask_antenna_anterior_tilt=20,
            mask_antenna_posterior_tilt=2,
            canny_minval=0,
            canny_maxval=50, **kwargs):

        ex, ey = self.eye_x_center, self.eye_y_center

        hc = self.high_contrast(im)

        edge_image = cv2.Canny(np.array(255*gaussian(hc, mask_antenna_blur), dtype=np.uint8), canny_minval, canny_maxval)/255
        edge_labels = measure.label(edge_image, background = 0)

        edge_copy = edge_image.copy()
        
        labels = np.ndarray.flatten(np.argwhere(np.bincount(np.ndarray.flatten(edge_labels[np.nonzero(edge_labels)])) > mask_antenna_cnx_comp_threshold))
        big_cc = np.isin(edge_labels, labels) 
        big_cc_x = np.where(big_cc)[0]
        big_cc_y = np.where(big_cc)[1]

        idx = np.argmax(np.linalg.norm(np.vstack(np.where(big_cc)).T - np.array((self.eye_x_center, self.eye_y_center)), axis=1))
        tx = big_cc_x[idx]
        ty = big_cc_y[idx]

        self.tail_tip = (tx, ty)

        cx, cy = (tx + ex)/2, (ty + ey)/2

        hx1, hy1 = 1.2*(ex - cx) + cx, 1.2*(ey - cy) + cy

        vd1 = cx + mask_antenna_coronal_tilt*(hy1 - cy), cy + mask_antenna_coronal_tilt*(cx - hx1)
        vd2 = cx - mask_antenna_coronal_tilt*(hy1 - cy), cy - mask_antenna_coronal_tilt*(cx - hx1)

        hx2, hy2 = 1.125*(ex - cx) + cx, 1.125*(ey - cy) + cy
        top1 = hx2 + mask_antenna_anterior_tilt*(ey - hy2), hy2 + mask_antenna_anterior_tilt*(hx2 - ex)
        top2 = hx2 - mask_antenna_anterior_tilt*(ey - hy2), hy2 - mask_antenna_anterior_tilt*(hx2 - ex)

        tail = 0.4*cx + 0.6*self.tail_tip[0], 0.4*cy + 0.6*self.tail_tip[1]
        bot1 = tail[0] + mask_antenna_posterior_tilt*(self.tail_tip[1] - tail[1]), tail[1] + mask_antenna_posterior_tilt*(self.tail_tip[0] - tail[0])
        bot2 = tail[0] - mask_antenna_posterior_tilt*(self.tail_tip[1] - tail[1]), tail[1] - mask_antenna_posterior_tilt*(self.tail_tip[0] - tail[0])
       
        edge_copy = self.mask_antenna(edge_copy, (cx, cy), a=[hx1, hy1, vd1[0], vd1[1]], b=[hx1, hy1, vd2[0], vd2[1]], c=[top1[0], top1[1], top2[0], top2[1]])
        self.get_anatomical_directions(edge_copy)

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
    
    def mask_antenna(self, edge, center, **kwargs):
        
        cx, cy = center

        edges_x = np.where(edge)[0]
        edges_y = np.where(edge)[1]
 
        #TO DO: Vectorize
        mask_x = []
        mask_y = []
     
        for i in xrange(len(edges_x)):
            for key, value in kwargs.iteritems():
                if self.intersect([cx, cy, edges_x[i], edges_y[i]], [value[0], value[1], value[2], value[3]]):
                    mask_x.append(edges_x[i])
                    mask_y.append(edges_y[i])
        
        edge[[mask_x, mask_y]] = 0
        return edge

    def get_anatomical_directions(self, im, fit_ellipse_chi2=3, flag="animal", **kwargs):

        x, y, major, minor, theta = self.fit_ellipse(im, fit_ellipse_chi2)
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

        self.eye_dorsal = ep[np.argmin(np.linalg.norm(ep - (dorsal_target_x, dorsal_target_y), axis=1)), :]
        self.eye_ventral = ep[np.argmin(np.linalg.norm(ep - (ventral_target_x, ventral_target_y), axis=1)), :]

    def find_head(self, im, find_head_blur=1.0, canny_minval=0, canny_maxval=50, **kwargs):

        hc = self.high_contrast(im)
        edges = cv2.Canny(np.array(255*gaussian(hc, find_head_blur), dtype=np.uint8), canny_minval, canny_maxval)/255

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
            self.head = edx - (-0.05*d*(edx - tx))/d, edy - (-0.05*d*(edy - ty))/d

    def find_tail(self, im, find_tail_blur=1.5, find_tail_n=100, canny_minval=0, canny_maxval=50,**kwargs):
         
        hc = self.high_contrast(im)
        edges = cv2.Canny(np.array(255*gaussian(hc, find_tail_blur), dtype=np.uint8), canny_minval, canny_maxval)/255

        tx, ty = self.tail_tip
        target = 3*self.eye_ventral[0] - 2*self.eye_x_center, 3*self.eye_ventral[1] - 2*self.eye_y_center
        ex, ey = 0.5*tx + 0.5*target[0], 0.5*ty + 0.5*target[1]
        
        m = (ty - ey)/(tx - ex)
        
        x, y = np.linspace(tx, ex, find_tail_n), np.linspace(ty, ey, find_tail_n)

        d = self.dist((tx, ty), (ex, ey))/8

        for i in xrange(int(find_tail_n)):
            
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
                    self.tail = e
                    self.tail_dorsal = self.find_edge2(edges, start, end)
                    self.tail_base = (self.tail[0] + self.tail_dorsal[0])/2, (self.tail[1] + self.tail_dorsal[1])/2
                    break

        if self.tail_dorsal is None:
            self.tail_dorsal = np.nan

        if self.tail is None:
            self.tail = self.tail_tip
        
        self.tail_spine_length = self.dist(self.tail_base, self.tail_tip)

    def get_dorsal_edge(self, im, dorsal_edge_blur=0.8, canny_minval=0, canny_maxval=50,**kwargs):
        
        hx, hy = self.head
        tx_d, ty_d = self.tail_dorsal
        cx, cy = self.animal_x_center, self.animal_y_center

        hc = self.high_contrast(im)
        edges = cv2.Canny(np.array(255*gaussian(hc, dorsal_edge_blur), dtype=np.uint8), 0, 50)/255

        edges = self.mask_antenna(edges, (cx, cy),
                dorsal=self.dorsal_mask_endpoints,
                ventral=self.ventral_mask_endpoints,
                anterior=self.anterior_mask_endpoints)

        m,b = self.line_fit(self.head, self.tail_dorsal)

        d = self.dist((hx,hy), (self.dorsal_mask_endpoints[0], self.dorsal_mask_endpoints[1]))

        checkpoints = {}

        for i in np.arange(0,1,0.1):
            mp = (1-i)*hx + i*tx_d, (1-i)*hy + i*ty_d
            x,y = self.orth(mp, d, m, flag="dorsal")
            p2 = self.find_edge2(edges, mp, (x,y))
            checkpoints[int(i*10)] = p2
        
        m,b = self.line_fit(checkpoints[4], self.tail_dorsal)

        dorsal_edge = self.traverse_dorsal_edge(edges, np.array(self.head), np.array(checkpoints[0]), flag="head")

        for k in np.arange(9):
            try:
                x,y = checkpoints[k]
                x_1, y_1 = checkpoints[k+1]
                err = np.abs(b + m*x -y)/np.sqrt(1 + m**2)
                err_plus_one = np.abs(b + m*x_1 - y_1)/np.sqrt(1 + m**2)

                if (k>4) and ((err > self.dist(self.head, self.tail)/16) or (err_plus_one > self.dist(self.head, self.tail)/16)):
                    raise TypeError
                else:
                    dorsal_edge = np.vstack([dorsal_edge, self.traverse_dorsal_edge(edges, checkpoints[k], checkpoints[k+1])])
            except TypeError:
                continue

        try:
            x,y = checkpoints[9]
            err = np.abs(b + m*x - y)/np.sqrt(1+m**2)
            
            if err < self.dist(self.head, self.tail)/16:
                dorsal_edge = np.vstack([dorsal_edge, self.traverse_dorsal_edge(edges, checkpoints[9], self.tail_dorsal)])
            else:
                raise TypeError
        except TypeError:
                dorsal_edge = np.vstack([dorsal_edge, self.tail_dorsal])
        self.dorsal_edge = self.interpolate(dorsal_edge)
        

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

    def traverse_dorsal_edge(self, edges, current,target,flag="dorsal",**kwargs):

        idx = self.index_on_pixels(edges)

        cx, cy = self.animal_x_center, self.animal_y_center

        dorsal_edge = [list(current)]
        
        target_vector = np.array(target - current)
        target_vector = self.norm_vec(target_vector)
        
        dorsal_vector = np.array(current - (cx, cy))
        dorsal_vector = self.norm_vec(dorsal_vector)

        nxt_vector = target_vector

        window=1
        w,h = edges.shape

        while (self.dist(current, target) > 2) and (window < 10):

            idx = self.index_on_pixels(edges[ np.max([0,current[0]-window]):np.min([w,current[0]+window+1]),
                                      np.max([0, current[1]-window]):np.min([h,current[1]+window+1])]) - (window,window)
            idx = idx[~np.all(idx == 0, axis=1)]

            try:
                
                nxt = current + idx[np.argmax(np.dot(idx, dorsal_vector) + np.dot(idx, target_vector) + np.dot(idx, nxt_vector))]

                if (list(nxt) in dorsal_edge) or (self.dist(nxt, target) > self.dist(current, target)):
                    raise(ValueError)
                else:

                    target_vector = np.array(target - current)
                    target_vector = self.norm_vec(target_vector)

		    dorsal_vector = np.array(current - (cx, cy))
		    dorsal_vector = self.norm_vec(dorsal_vector)

		    nxt_vector = np.array(nxt - current)
		    nxt_vector = self.norm_vec(nxt_vector)
		    
		    current = nxt
		    dorsal_edge.append(list(current))
		    window=1
            
            except ValueError:
                window += 1

        if not (list(target) in dorsal_edge):
            dorsal_edge.append(list(target))

        return dorsal_edge
    
    def index_on_pixels(self,a):
        return np.transpose(np.vstack(np.where(a)))
    
    def norm_vec(self, v):
        return v/np.max(np.abs(v))

    def get_pedestal_max_height(self, data):
        
        self.pedestal_max_height = np.max(data[:,1])

    def get_pedestal_area(self, data):
        
        self.pedestal_area = np.sum(0.5*(self.dist(self.head, self.dorsal_point)/400)*(data[1:][:,0] - data[0:-1][:,0])*(data[1:][:,1] + data[0:-1][:,1]))
        
    def get_pedestal_theta(self, data, n=200):
        
        x = (n - data[np.argmax(data[:,1]), 0]) * self.dist(self.head, self.dorsal_point)/400
        hyp = self.dist((n,0), (x, np.max(data[:,1])))
        self.pedestal_theta = np.arcsin((n - x)/hyp)*(180/np.pi)

    def find_edge(self, im, p1, p2, t1=0.1, npoints=400, lp=np.nan, t2=2):

        xx, yy = np.linspace(p1[1], p2[1], npoints), np.linspace(p1[0], p2[0], npoints)
        zi = mc(im, np.vstack((yy, xx)), mode="nearest")
        zi = pd.rolling_mean(zi, 4)

        for i in xrange(len(zi)):
            if zi[i] > t1:
                if lp is np.nan: return(yy[i], xx[i])
                elif self.dist((yy[i], xx[i]), lp) < t2:
                    return (yy[i], xx[i])
                else:
                    continue

    def find_edge2(self, edges, p1, p2):

        idxx, idxy = np.where(edges)
        
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
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(im)
    
    def area(self, x, y):

        x = np.asanyarray(x)
        y = np.asanyarray(y)
        n = len(x)

        up = np.arange(-n+1, 1)
        down = np.arange(-1, n-1)

        return (x * (y.take(up) - y.take(down))).sum() / 2
    
    def intersect(self, s1, s2):

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

        return (x - np.min(x)) / (np.max(x) - np.min(x))

    def analyze_pedestal(self, analyze_pedestal_moving_avg_window=12, analyze_pedestal_percentile=80, analyze_pedestal_polyfit_degree=3, pedestal_n=400, **kwargs):
    
        # ma = window for moving average
        # w_p = lower percentile for calculating polynomial model
        # deg = degree of polynomial model
        self.interpolate()
        p = self.pedestal
 
        # smooth pedestal
        window = int(analyze_pedestal_moving_avg_window)
        s = pd.rolling_mean(p, window)
        s[0:window, :] = p[0:window, :]

        p1 = s[0, :]
        p2 = s[-1, :]

        m = (p2[1] - p1[1])/(p2[0] - p1[0])
        b = p1[1] - m*p1[0]
        h = np.abs(-m*s[:,0] + s[:,1] - b)/np.sqrt(m**2 + 1)
        
        ipeak = np.argmax(h)
        self.peak = p[ipeak]
        
        m1 = ((s[ipeak-1,1]-s[ipeak+1,1])/(s[ipeak-1,0]-s[ipeak+1,0]))
        
        origin = [0,0]

        qx, qy = self.rotate(origin, s, np.pi - np.arctan(m1))

        if qy[ipeak] < qy[0]:
            qx, qy = self.rotate(origin, s, 2*np.pi - np.arctan(m1))
        
        qx -= np.min(qx)
        qy -= np.min(qy)

        threshold = np.percentile(qy, analyze_pedestal_percentile)
        poly_train = np.where(qy<threshold)
        roi = np.where(qy >= threshold)

        X = np.transpose(np.vstack([ qx[poly_train], qy[poly_train] ]))
        self.poly_coeff, res, _, _, _ = np.polyfit(X[:,0], X[:,1], analyze_pedestal_polyfit_degree, full=True)
        self.res = res[0]

        poly = np.poly1d(self.poly_coeff)
        
        yy = poly(qx)
        diff = qy - yy
          
        self.calc_pedestal_area(qx[roi], diff[roi])
        self.pedestal_max_height = np.max(diff[roi])
        self.get_deye_pedestalmax()

    def rotate(self, origin, points, phi):

        ox, oy = origin
        px, py = points[:,0], points[:,1]

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

    def interpolate(self, dorsal_edge, pedestal_n=500, **kwargs):

        p = dorsal_edge
        diff = np.linalg.norm(p[1:,:] - p[0:-1,:], axis=1)
        biggest_gaps = np.where(diff == np.max(diff))[0]

        while p.shape[0] < pedestal_n:
 
            i = random.choice(biggest_gaps)
            biggest_gaps[np.where(biggest_gaps==i)[0][0]+1:] += 1
            biggest_gaps = np.delete(biggest_gaps, np.where(biggest_gaps == i))

            new_point = (p[i] + p[i+1])/2
            p = np.vstack([p[:i+1, :], new_point, p[i+1:,:]])

            if len(biggest_gaps) == 0:
                diff = np.linalg.norm(p[1:,:] - p[0:-1,:], axis=1)
                biggest_gaps = np.where(diff == np.max(diff))[0]
        return p
