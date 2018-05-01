from __future__ import division
import numpy as np
import pandas as pd
import os
import cv2
import scipy
from scipy.ndimage import map_coordinates as mc
from skimage import measure
from skimage.filters import gaussian

class Clone(object):
    
    def __init__(self, filepath):
        
        self.filepath = filepath

        self.total_animal_pixels = np.nan
        self.animal_area = np.nan
        self.total_eye_pixels = np.nan
        self.eye_area = np.nan
        self.animal_length_pixels = np.nan
        self.animal_length = np.nan
        self.pedestal = np.nan
        self.ipedestal = np.nan
        self.binned_pedestal_data = []
        self.pedestal_area = np.nan
        self.pedestal_theta = np.nan
        self.snake = np.nan
        self.pixel_to_mm = np.nan
        
        self.pedestal_max_height_pixels = np.nan
        self.pedestal_area_pixels = np.nan
        self.pedestal_max_height = np.nan
        self.pedestal_area = np.nan

        self.pedestal_window_max_height_pixels = np.nan
        self.pedestal_window_area_pixels = np.nan
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
        self.head = np.nan
        self.tail = np.nan
        self.tail_tip = np.nan
        self.dorsal_point = np.nan

        self.peak = np.nan
        self.deyecenter_pedestalmax_pixels = np.nan
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
            self.total_eye_pixels = count
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

        r = 2*self.dist((cx, cy), self.anterior)

        s = np.linspace(0, 2*np.pi, count_animal_pixels_n)
        x = cx + int(r)*np.sin(s)
        y = cy + int(r)*np.cos(s)

        pts = []

        for i in xrange(len(s)):
            

            p1 = (x[i], y[i])
            p2 = (cx, cy)

            if self.intersect((p1[0], p1[1], cx, cy), (hx1, hy1, vx, vy)):
                res = self.intersection((p1[0], p1[1], cx, cy), (hx1, hy1, vx, vy))
                p1 = (res[0], res[1])

            if self.intersect((p1[0], p1[1], cx, cy), (hx2, hy2, dx, dy) ):
                res = self.intersection((p1[0], p1[1], cx, cy), (hx2, hy2, dx, dy))
                p1 = (res[0], res[1])

            if self.intersect((p1[0], p1[1], cx, cy), (topx1, topy1, topx2, topy2) ):
                res = self.intersection((p1[0], p1[1], cx, cy), (topx1, topy1, topx2, topy2))
                p1 = (res[0], res[1])

            edge_pt = self.find_edge(edges, p1, p2)

            if edge_pt is not np.nan:
                pts.append((edge_pt[1], edge_pt[0]))

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
                        idx -= 1
                    elif connected:
                        idx += 1
                        connected = False
                except IndexError:
                    pass
        
        pts = np.vstack(cc) 
        self.whole_animal_points = pts
        self.total_animal_pixels = self.area(pts[:,0], pts[:,1])

    def get_animal_length(self):

        self.animal_length_pixels = self.dist(self.head, self.tail)

    def mask_antenna(self, im, mask_antenna_blur=1.5,
            mask_antenna_cnx_comp_threshold=125,
            mask_antenna_coronal_tilt = 0.7,
            mask_antenna_anterior_tilt=20,
            mask_antenna_posterior_tilt=2,
            canny_minval=0,
            canny_maxval=1, **kwargs):

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
       
        edges_x = np.where(edge_image)[0]
        edges_y = np.where(edge_image)[1]

        #TO DO: Vectorize
        mask_x1 = []
        mask_y1 = []
        mask_x2 = []
        mask_y2 = []
        top_mask_x = []
        top_mask_y = []

        for i in xrange(len(edges_x)):
            if self.intersect([cx, cy, edges_x[i], edges_y[i]], [hx1, hy1, vd1[0], vd1[1]]):
                mask_x1.append(edges_x[i])
                mask_y1.append(edges_y[i])
            if self.intersect([cx, cy, edges_x[i], edges_y[i]], [hx1, hy1, vd2[0], vd2[1]]):
                mask_x2.append(edges_x[i])
                mask_y2.append(edges_y[i])
            if self.intersect([cx, cy, edges_x[i], edges_y[i]], [top1[0], top1[1], top2[0], top2[1]]):
                top_mask_x.append(edges_x[i])
                top_mask_y.append(edges_y[i])

        edge_copy[[mask_x1, mask_y1]] = 0
        edge_copy[[mask_x2, mask_y2]] = 0
        edge_copy[[top_mask_x, top_mask_y]] = 0

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

        self.dorsal_mask_endpoints = ((hx - 1.4*shift[0], hy - 1.4*shift[1]), self.tail_tip)
        self.ventral_mask_endpoints = ((self.ventral_mask_endpoints[0][0] + 0.05*shift[0],
            self.ventral_mask_endpoints[0][1] + 0.05*shift[1]),
            (self.ventral_mask_endpoints[1][0] + 0.05*shift[0],
                self.ventral_mask_endpoints[1][1] + 0.05*shift[1]))
        self.anterior_mask_endpoints = (top1, top2)
        self.edge_copy = edge_copy

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
       
    def find_head(self, im, find_head_blur=1.0, canny_minval=0, canny_maxval=50, **kwargs):

        hc = self.high_contrast(im)
        edges = cv2.Canny(np.array(255*gaussian(hc, find_head_blur), dtype=np.uint8), canny_minval, canny_maxval)/255

        ep = self.eye_pts
        ex, ey = self.eye_x_center, self.eye_y_center
        dx, dy = self.dor_vec
        target_x, target_y = ex - dx, ey - dy

        self.eye_dorsal = ep[np.argmin(np.linalg.norm(ep - (target_x, target_y), axis=1)), :]
        edx, edy = self.eye_dorsal

        tx, ty = self.tail
        m = (ty - edy)/(tx - edx)
        b = edy - m*edx

        d = self.dist((edx, edy), (tx, ty))
        cx = edx - (-0.15*d*(edx - tx))/d
        cy = edy - (-0.15*d*(edy - ty))/d

        (topx1, topy1), (topx2, topy2) = self.anterior_mask_endpoints

        if self.intersect((edx, edy, cx, cy), (topx1, topy1, topx2, topy2)):
            res = self.intersection((edx, edy, cx, cy), (topx1, topy1, topx2, topy2))
            p1 = res[0], res[1]
            p2 = edx, edy
        else:
            p1 = cx, cy
            p2 = edx, edy
        
        try:
            hx, hy = self.find_edge(edges, p1, p2)
            self.head = hx, hy
        
        except TypeError:
            
            # if head edge can't be found, just estimate based on dorsal eye point
            self.head = edx - (-0.05*d*(edx - tx))/d, edy - (-0.05*d*(edy - ty))/d

    def find_tail(self, im, find_tail_blur=1.5, find_tail_n=100, canny_minval=0, canny_maxval=50,**kwargs):
         
        hc = self.high_contrast(im)
        edges = cv2.Canny(np.array(255*gaussian(hc, find_tail_blur), dtype=np.uint8), canny_minval, canny_maxval)/255

        tx, ty = self.tail_tip
        ex, ey = 0.5*tx + 0.5*self.eye_x_center, 0.5*ty + 0.5*self.eye_y_center
        
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

            e = self.find_edge(edges, start, end)

            if self.dist(e, start) < self.dist(p1, p2)/45:
                self.tail = e
                break

        if self.tail == None:
            self.tail = self.tail_tip

    def initialize_pedestal(self, im, initialize_pedestal_blur=1.25, pedestal_n=400, canny_minval=0, canny_maxval=50,**kwargs):
        
        ex, ey = self.eye_x_center, self.eye_y_center
        tx, ty = self.tail_tip[0], self.tail_tip[1]

        hc = self.high_contrast(im)
        edges = cv2.Canny(np.array(255*gaussian(hc, initialize_pedestal_blur), dtype=np.uint8), canny_minval, canny_maxval)/255

        m = (ty - ey)/(tx - ex)
        b = ey - m*ex

        d = self.dist((ex, ey), (self.dorsal_mask_endpoints[0][0], self.dorsal_mask_endpoints[0][1]))
        
        x, y = self.orth((ex, ey), d, m, flag="dorsal")
        p1 = self.find_edge(edges, (x, y), (ex, ey))

        mp = 0.67*ex + 0.33*tx, 0.67*ey + 0.33*ty

        x, y = self.orth(mp, d, m, flag="dorsal")
        p2 = self.find_edge(edges, (x, y), mp)

        m2 = (p1[1] - p2[1])/(p1[0] - p2[0])
        xx1, yy1 = self.orth(p1, d*0.25, m2)
        xx2, yy2 = self.orth(p2, d*0.25, m2)

        bsx, bsy = np.linspace(p1[0], p2[0], pedestal_n), np.linspace(p1[1], p2[1], pedestal_n)
        xx, yy = np.linspace(xx1, xx2, pedestal_n), np.linspace(yy1, yy2, pedestal_n)

        self.baseline = np.array([bsx, bsy]).T
        self.pedestal = np.array([xx, yy]).T

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

    def fit_pedestal(self, im, fit_pedestal_blur=1, canny_minval=0, canny_maxval=50,**kwargs):

        if self.pedestal is np.nan: self.initialize_pedestal(im)
        
        ps = self.pedestal
        bs = self.baseline

        hc = self.high_contrast(im)
        edges = cv2.Canny(np.array(255*gaussian(hc, fit_pedestal_blur), dtype=np.uint8), canny_minval, canny_maxval)/255
    
        snakex = ps[:,0]
        snakey = ps[:,1]

        d = []
        idx = []
        
        n = len(snakex)
        t2 = self.dist(bs[0,:], bs[-1,:])/60

        for i in xrange(1,n):

            p2 = snakex[i], snakey[i]
            p1 = bs[i,0], bs[i,1]
            
            if len(d) > 0:
                lp = d[-1]
            else:
                lp = np.nan
            e = self.find_edge(edges, p2, p1, lp=lp, t2=t2)
            
            if e is not None:
                d.append(e)
                idx.append(i) 
        
        self.pedestal, self.ipedestal = d, idx
    
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

        self.interpolate(pedestal_n)
        p = self.interp_pedestal

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
        threshold = np.percentile(h, analyze_pedestal_percentile)

        for j in xrange(ipeak, len(h)):
            if h[j] <= threshold:
                qub = j
                break

        for j in xrange(ipeak, 0, -1):
            if h[j] <= threshold:
                qlb = j
                break

        self.peak = p[ipeak]
        
        m1 = ((s[ipeak-1,1]-s[ipeak+1,1])/(s[ipeak-1,0]-s[ipeak+1,0]))
        
        origin = [0,0]

        qx, qy = self.rotate(origin, s, np.pi - np.arctan(m1))

        if qy[ipeak] < qy[0]:
            qx, qy = self.rotate(origin, s, 2*np.pi - np.arctan(m1))
        
        qx -= np.min(qx)
        qy -= np.min(qy)

        X = np.transpose(np.vstack([np.concatenate([qx[:qlb], qx[qub:]]), np.concatenate([qy[:qlb], qy[qub:]])]))

        self.poly_coeff, res, _, _, _ = np.polyfit(X[:,0], X[:,1], analyze_pedestal_polyfit_degree, full=True)
        self.res = res[0]
        poly = np.poly1d(self.poly_coeff)
        
        yy = poly(qx)
        diff = qy - yy
        diff[np.where(diff<0)] = 0
        
        lb, ub = 0, 0

        for j in xrange(ipeak, len(diff)):
            if diff[j] == 0:
                ub = j
                break
        for j in xrange(ipeak, 0, -1):
            if diff[j] == 0:
                lb = j
                break
          
        self.calc_pedestal_area(qx[lb:ub], diff[lb:ub])
        
        try:
            self.pedestal_max_height_pixels = np.max(diff[lb:ub])
        except ValueError:
            # if there is no section of pedestal that is higher than predictive model, set max height to 0
            self.pedestal_max_height_pixels = 0.0

        self.get_deye_pedestalmax()

    def rotate(self, origin, points, phi):

        ox, oy = origin
        px, py = points[:,0], points[:,1]

        qx = ox + np.cos(phi) * (px - ox) - np.sin(phi) * (py - oy)
        qy = oy + np.sin(phi) * (px - ox) + np.cos(phi) * (py - oy)

        return qx, qy

    def get_deye_pedestalmax(self):

        self.deyecenter_pedestalmax_pixels = self.dist((self.eye_x_center, self.eye_y_center), self.peak)

    def flip(self, p, ip):
        
        x1, y1 = p[0, :]
        x2, y2 = p[p.shape[0]-1, :]
        ex, ey = self.eye_x_center, self.eye_y_center

        if self.dist((x1, y1), (ex, ey)) > self.dist((x2, y2), (ex, ey)):
            ip = np.max(ip) - ip
            return np.flip(p, axis = 0), ip[::-1]
        
    def calc_pedestal_area(self, x, y):
        
        # calculates the area of pedestal

        self.pedestal_area_pixels = np.abs(np.sum((y[1:] + y[:-1])*(x[1:] - x[:-1])/2))

    def interpolate(self, pedestal_n):

        p = self.pedestal
        p = np.array([list(x) for x in p])
        idx = np.array(self.ipedestal)
        new_idx = [0]
        new_p = [p[0,:]]

        start = idx[0]
        for i in xrange(1, start+1):
            new_idx.append(i)
            new_p.append(p[0,:])

        for i in xrange(len(idx) - 1):
            d = idx[i+1] - idx[i]

            if d > 1:
                x1, y1 = p[i, :]
                x2, y2 = p[i+1, :]

                xx, yy = np.linspace(x1, x2, d), np.linspace(y1, y2, d)

                [new_p.append([xx[x], yy[x]]) for x in xrange(0, len(xx))]
                [new_idx.append(x) for x in xrange(idx[i] + 1, idx[i+1] + 1)]
            else:
                new_p.append(p[i+1, :])
                new_idx.append(idx[i+1])

        end = idx[-1]
        
        for i in xrange(end+1, int(pedestal_n)):
            new_idx.append(i)
            new_p.append(p[p.shape[0]-1,:])
            
        self.interp_pedestal = np.vstack(new_p)
        self.interp_idx = np.array(new_idx)
