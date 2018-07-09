import os
import cv2
import utils
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.widgets import Slider
import numpy as np
from skimage.filters import gaussian
from clone import Clone

DATA_COLS = ["filepath",
            "animal_dorsal_area",
            "eye_area",
            "animal_length",
            "animal_x_center",
            "animal_y_center",
            "animal_major",
            "animal_minor",
            "animal_theta",
            "eye_x_center",
            "eye_y_center",
            "anterior",
            "posterior",
            "dorsal",
            "ventral",
            "ant_vec",
            "pos_vec",
            "dor_vec",
            "ven_vec",
            "eye_dorsal",
            "head",
            "tail",
            "tail_tip",
            "tail_base",
            "tail_dorsal",
            "tail_spine_length",
            "ventral_mask_endpoints",
            "dorsal_mask_endpoints",
            "anterior_mask_endpoints",
            "pedestal_max_height",
            "pedestal_area",
            "poly_coeff",
            "res",
            "peak",
            "deyecenter_pedestalmax",
            "dorsal_residual",
            "automated_PF",
            "automated_PF_reason"]

METADATA_FIELDS = ["filebase",
            "barcode",
            "cloneid",
            "pond",
            "id",
            "treatment",
            "replicate",
            "rig",
            "datetime",
            "inductiondate",
            "manual_PF",
            "manual_PF_reason",
            "manual_PF_curator",
            "pixel_to_mm",
            "season",
            "animal_dorsal_area_mm",
            "animal_length_mm",
            "eye_area_mm",
            "tail_spine_length_mm",
            "deyecenter_pedestalmax_mm",
            "pedestal_area_mm",
            "pedestal_max_height_mm",
            "experimenter",
            "inducer"]

class PointFixer:
    
    def __init__(self, clone, display):
        
        im = cv2.imread(clone.filepath, cv2.IMREAD_GRAYSCALE)
        self.original_image = im
        self.image = im
        self.edges = False
        self.edge_blur = 1.0
        
        self.display = display
        self.clone = clone

        self.params = {}
        self.clone.find_eye(self.image, **self.params)
        self.clone.find_features(self.image, **self.params)
        self.clone.get_orientation_vectors()
        self.clone.eye_vertices()
        self.clone.find_head(self.image, **self.params)
        self.clone.initialize_dorsal_edge(self.image, **self.params)
        self.clone.fit_dorsal_edge(self.image, **self.params)
        self.clone.find_tail(self.image)
        self.clone.remove_tail_spine()
        self.clone.get_animal_length()
        self.clone.get_animal_dorsal_area()
        self.de = clone.dorsal_edge
        
        self.selected = None
        self.checkpoints = self.clone.checkpoints  
        self.cid = self.display.figure.canvas.mpl_connect('button_press_event',self)
        self.draw() 
    
    def __call__(self, event):
        if event.inaxes!=self.display.axes: return
        
        if (self.selected is None):
        
            self.get_closest_checkpoint(event.ydata, event.xdata) 
            self.display.scatter(self.selected[1], self.selected[0], c="green")
            self.display.figure.canvas.draw()
        else:
            
            self.display.clear()
            
            self.set_closest_checkpoint(event.ydata, event.xdata)
            self.clone.checkpoints = self.checkpoints
            self.clone.fit_dorsal_edge(self.original_image)
            self.clone.remove_tail_spine()
            self.de = self.clone.dorsal_edge
            self.checkpoints = self.clone.checkpoints
            self.selected = None
            
            self.draw()

    def reset_button_press(self, event):
        
        if self.edges:
            self.edge_button_press(1)
        
        if self.clone.flip:
            self.clone.flip = not self.clone.flip

        self.clone.find_features(self.original_image, **self.params)
        self.clone.get_orientation_vectors()
        self.clone.find_head(self.original_image, **self.params)
        self.edge_blur = 1.0
        self.clone.dorsal_blur = 1.0
        self.clone.initialize_dorsal_edge(self.original_image, **self.params)
        self.clone.fit_dorsal_edge(self.original_image, **self.params)
        self.clone.find_tail(self.original_image)
        self.clone.remove_tail_spine()
        self.de = self.clone.dorsal_edge
        self.selected = None 
        self.checkpoints = self.clone.checkpoints
        
        self.draw()

    def flip_button_press(self, event):
   
        self.display.imshow(self.image, cmap="gray")
        params = {}
        self.clone.flip = not self.clone.flip
        self.clone.find_features(self.original_image, **self.params)
        self.clone.get_orientation_vectors()
        self.clone.eye_vertices()
        self.clone.find_head(self.original_image, **self.params)
        self.clone.initialize_dorsal_edge(self.original_image, **self.params)
        self.clone.fit_dorsal_edge(self.original_image, **self.params)
        self.clone.find_tail(self.original_image)
        self.clone.remove_tail_spine()
        self.de = self.clone.dorsal_edge
        self.checkpoints = self.clone.checkpoints 
        self.draw()

    def edge_button_press(self, event):

        self.edges = not self.edges

        if self.edges:
            hc = self.clone.high_contrast(self.original_image)
            self.image = cv2.Canny(np.array(255*gaussian(hc, self.edge_blur), dtype=np.uint8), 0, 50)/255
            self.draw()
        if not self.edges:
            self.image = self.original_image
            self.draw()
    
    def set_blur_slider(self, event):
        
        self.edge_blur = event
        self.clone.initialize_dorsal_edge(self.original_image, dorsal_edge_blur = self.edge_blur, **self.params)
        self.clone.fit_dorsal_edge(self.original_image, dorsal_edge_blur = self.edge_blur, **self.params)
        self.edges = False
        self.checkpoints = self.clone.checkpoints
        self.de = self.clone.dorsal_edge
        self.edge_button_press(1)

    def draw(self):
        
        self.display.clear()
        
        self.display.imshow(self.image, cmap="gray")
        self.display.scatter(self.de[:,1], self.de[:,0], c="blue")
        
        if self.edges:
            checkpoint_color = "yellow"
        else:
            checkpoint_color = "black"
        
        self.display.scatter(self.checkpoints[:,1], self.checkpoints[:,0], c=checkpoint_color)
        self.display.axis('off')

        self.display.set_title(clone.filebase)

        self.display.figure.canvas.draw()

    def get_closest_checkpoint(self, x, y):

        self.selected = self.checkpoints[np.argmin(np.linalg.norm(self.checkpoints-(x,y), axis=1)), :]
    
    def set_closest_checkpoint(self, x, y):
        
        val = self.checkpoints[np.argmin(np.linalg.norm(self.checkpoints - self.selected, axis=1)), :]
        self.checkpoints[np.argmin(np.linalg.norm(self.checkpoints - self.selected, axis=1)), :] = (x, y)
        
        if self.clone.dist(val,self.clone.head) < 0.0001:
            self.clone.head = (x, y)
        if self.clone.dist(val, self.clone.tail_dorsal) < 0.0001:
            self.clone.tail_dorsal = (x,y)

class Viewer:

    def __init__(self, clone_list):
        
        self.clone_list = clone_list
        self.curr_idx = 0
        self.clone = self.clone_list[self.curr_idx]
        
        self.fig = plt.figure(figsize=(15,10))
        self.display = self.fig.add_subplot(111)
        self.display.axis('off')

        self.obj = PointFixer(self.clone, self.display)
        self.populate_figure()

    def populate_figure(self):

        axreset = plt.axes([0.40, 0.01, 0.1, 0.075])
        self.resetbutton = Button(axreset, 'Reset')
        self.resetbutton.on_clicked(self.obj.reset_button_press)

        axflip = plt.axes([0.50, 0.01, 0.1, 0.075])
        self.flipbutton = Button(axflip, 'Flip')
        self.flipbutton.on_clicked(self.obj.flip_button_press)

        axedges = plt.axes([0.60, 0.01, 0.1, 0.075])
        self.edgebutton = Button(axedges, 'Toggle Edges')
        self.edgebutton.on_clicked(self.obj.edge_button_press)

        axblur = plt.axes([0.20, 0.01, 0.1, 0.075])
        self.blurslider = Slider(axblur, 'Blur Slider', 0, 3)
        self.blurslider.set_val(self.obj.edge_blur)
        self.blurslider.on_changed(self.obj.set_blur_slider)

        if self.curr_idx+1 < len(self.clone_list):
            axnext = plt.axes([0.85, 0.01, 0.1, 0.075])
            self.nextbutton = Button(axnext, 'Next')
            self.nextbutton.on_clicked(self.next_button_press)

        if self.curr_idx > 0:
            axprev = plt.axes([0.75, 0.01, 0.1, 0.075])
            self.prevbutton = Button(axprev, 'Previous')
            self.prevbutton.on_clicked(self.prev_button_press)


    def prev_button_press(self,event):

        self.curr_idx -= 1
        self.clone = self.clone_list[self.curr_idx]
        
        plt.clf()
        
        self.display = self.fig.add_subplot(111)
        self.display.axis('off')

        self.obj = PointFixer(self.clone, self.display)
        self.populate_figure()

    def next_button_press(self,event):

        self.curr_idx += 1
        self.clone = self.clone_list[self.curr_idx]
        
        plt.clf()

        self.display = self.fig.add_subplot(111)
        self.display.axis('off')

        self.obj = PointFixer(self.clone, self.display)
        self.populate_figure()        
        
#

gui_params = utils.myParse("gui_params.txt")

file_list = []

print "Reading in image file list"
with open(gui_params["image_list_filepath"], "rb") as f:
    line = f.readline().strip()
    while line:
        file_list.append(line)
        line = f.readline().strip()

print "Reading in analysis file"
df = utils.csv_to_df(gui_params["input_analysis_file"])

clone_list = []

for f in file_list:
    try:
        fileparts = f.split("/")

        clone = utils.dfrow_to_clone( df, np.where(df.filebase == fileparts[-1])[0][0] )
        clone.filepath = f
        clone_list.append(clone)
    except Exception:
        clone_list.append(Clone(f))


v = Viewer(clone_list) 
plt.show()


