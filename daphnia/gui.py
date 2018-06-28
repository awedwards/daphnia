import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.widgets import Slider
import numpy as np
from skimage.filters import gaussian
from clone import Clone

#f = "/Volumes/daphnia/data/full_101956_DBunk_131_juju_1A_RigB_20170726T173118.bmp"
#f = "/Volumes/daphnia/data/full_101262_DBunk_132_juju1_2A_RigB_20170622T163323.bmp"
#f = "/Volumes/daphnia/data/full_110928_DBunk_131_ctrl_3B_RigB_20171013T113523.bmp"
#f = "/Volumes/daphnia/data/full_110773_DBunk_321_ctrl_1B_RigB_20171004T171435.bmp"

#f = "/Volumes/daphnia/data/full_110887_D8_298_juju_2A_RigA_20171013T110416.bmp"
#f = "/Volumes/daphnia/data/full_101784_D8_137_ctrl_1A_RigB_20170727T141928.bmp"
#f = "/Volumes/daphnia/data/full_100871_AD8_29_juju2_1C_RigB_20170615T133402.bmp"
#f = "/Volumes/daphnia/data/full_110195_D8_213_juju_2B_RigB_20170809T162438.bmp"
#f = "/Volumes/daphnia/data/full_101855_D8_65_juju_2B_RigA_20170727T123919.bmp"

#f = "/Volumes/daphnia/data/full_100493_D10_A14_juju1_2B_RigA_20170606T151522.bmp"
f = "/Users/edwardsa/Documents/bergland/good_daphnia_images/image102.png"

clone = Clone(f)

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
        self.cid = display.figure.canvas.mpl_connect('button_press_event',self)
        self.draw() 
    
    def __call__(self, event):
        if event.inaxes!=self.display.axes: return
        
        if (self.selected is None):
        
            self.get_closest_checkpoint(event.ydata, event.xdata) 
            self.display.scatter(self.selected[1], self.selected[0], c="green")
            self.display.figure.canvas.draw()
        else:
            self.display.clear()
            self.display.imshow(self.image,cmap="gray")
            self.set_closest_checkpoint(event.ydata, event.xdata)
            self.clone.checkpoints = self.checkpoints
            self.clone.fit_dorsal_edge(self.original_image)
            self.clone.remove_tail_spine()
            self.de = self.clone.dorsal_edge
            self.checkpoints = self.clone.checkpoints
            self.selected = None
            
            self.draw()

    def reset_button_press(self, event):

        self.display.clear()
        
        if self.edges:
            self.edge_button_press(1)

        params = {}
        
        self.clone.find_head(self.original_image, **self.params)
        self.edge_blur = 1.0
        self.clone.dorsal_blur = 1.0
        self.clone.initialize_dorsal_edge(self.original_image, **self.params)
        self.clone.fit_dorsal_edge(self.original_image, **self.params)
        self.clone.find_tail(self.original_image)
        self.clone.remove_tail_spine()
        self.de = clone.dorsal_edge
        self.selected = None 
        self.checkpoints = self.clone.checkpoints
        self.draw()

    def flip_button_press(self, event):
   
        self.display.clear()
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
        self.de = clone.dorsal_edge
        self.checkpoints = self.clone.checkpoints 
        self.draw()

    def edge_button_press(self, event):

        self.display.clear()
        
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
        
        self.display.imshow(self.image, cmap="gray")
        self.display.scatter(self.de[:,1], self.de[:,0], c="blue")
        if self.edges:
            checkpoint_color = "yellow"
        else:
            checkpoint_color = "black"

        self.display.scatter(self.checkpoints[:,1], self.checkpoints[:,0], c=checkpoint_color)
        self.display.axis('off')
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

fig = plt.figure(figsize=(20,20))
display = fig.add_subplot(111)
display.axis('off')

obj = PointFixer(clone, display)

axreset = plt.axes([0.40, 0.01, 0.1, 0.075])
resetbutton = Button(axreset, 'Reset')
resetbutton.on_clicked(obj.reset_button_press)

axflip = plt.axes([0.50, 0.01, 0.1, 0.075])
flipbutton = Button(axflip, 'Flip')
flipbutton.on_clicked(obj.flip_button_press)

axedges = plt.axes([0.60, 0.01, 0.1, 0.075])
edgebutton = Button(axedges, 'Toggle Edges')
edgebutton.on_clicked(obj.edge_button_press)

axblur = plt.axes([0.20, 0.01, 0.1, 0.075])
blurslider = Slider(axblur, 'Blur Slider', 0, 3)
blurslider.set_val(obj.edge_blur)
blurslider.on_changed(obj.set_blur_slider)

plt.show()
