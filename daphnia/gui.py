import os
import cv2
import utils
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-dark-palette')
from matplotlib.widgets import Button, Slider, TextBox, LassoSelector
from matplotlib.path import Path
import numpy as np
import pandas as pd
from skimage.filters import gaussian
from clone import Clone

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

ANALYSIS_METADATA_FIELDS = ["edge_pixel_distance_threshold_multiplier",
            "count_animal_pixels_cc_threshold",
            "mask_antenna_posterior_tilt",
            "mask_antenna_anterior_tilt",
            "pedestal_n",
            "mask_antenna_blur",
            "analyze_pedestal_percentile",
            "canny_maxval",
            "count_animal_pixels_blur",
            "mask_antenna_coronal_tilt",
            "find_head_blur",
            "count_animal_pixels_n",
            "fit_ellipse_chi2",
            "analyze_pedestal_polyfit_degree",
            "analyze_pedestal_moving_avg_window",
            "dorsal_edge_blur",
            "canny_minval",
            "find_tail_blur",
            "find_eye_blur"]

class PointFixer:
    
    def __init__(self, clone, display):
        
        print "Reading image"
        im = cv2.imread(clone.filepath, cv2.IMREAD_GRAYSCALE)

        if im is None:
            print "Image " + clone.filepath + " not found. Check your image filepath."
            raise IOError
        
        self.original_image = im
        self.image = im

        self.display = display
        self.clone = clone
        
        self.edges = False
        self.edge_blur = 1.0

        hc = self.clone.high_contrast(self.original_image)
        
        self.original_edge_image = cv2.Canny(np.array(255*gaussian(hc, self.edge_blur), dtype=np.uint8), 0, 50)/255
        self.edge_image = self.original_edge_image.copy()

        self.edge_image  = self.clone.mask_antenna(self.edge_image, (self.clone.animal_x_center, self.clone.animal_y_center),
                dorsal=self.clone.dorsal_mask_endpoints,
                ventral=self.clone.ventral_mask_endpoints,
                anterior=self.clone.anterior_mask_endpoints)
        
        #self.edge_image = self.original_edge_image

        self.params = {}
        self.de = clone.dorsal_edge
        
        self.show_dorsal_edge = True
        self.add_checkpoint = False
        self.mask_clicked = False
        self.unmask_clicked = False
        
        self.selected = None
        self.checkpoints = self.clone.checkpoints  
        self.cid = self.display.figure.canvas.mpl_connect('button_press_event',self)

        self.draw()
    
    def __call__(self, event):
        
        if event.inaxes!=self.display.axes: return
        
        if self.add_checkpoint:
            self.get_closest_checkpoint(event.ydata, event.xdata, n=2)
            self.insert_new_checkpoint(event.ydata, event.xdata)
            self.clone.checkpoints = self.checkpoints
            self.fit_dorsal()
            self.add_checkpoint = False
            self.selected = None
            self.draw()
        
        elif not (self.mask_clicked or self.unmask_clicked):

            if (self.selected is None):
            
                self.get_closest_checkpoint(event.ydata, event.xdata)
                self.display.scatter(self.selected[1], self.selected[0], c="green")
                self.display.figure.canvas.draw()
        
            else:
            
                self.display.clear()
                
                self.set_closest_checkpoint(event.ydata, event.xdata)
                self.clone.checkpoints = self.checkpoints
                self.fit_dorsal()
                self.selected = None
            
                self.draw()

    def fit_dorsal(self):

        self.clone.fit_dorsal_edge(self.original_image, dorsal_edge_blur=self.edge_blur,edges=self.edge_image)
        self.clone.remove_tail_spine()
        self.de = self.clone.interpolate(self.clone.dorsal_edge)
        self.checkpoints = self.clone.checkpoints
        self.clone.qscore()
        self.clone.analyze_pedestal()

    def flip_button_press(self, event):
   
        self.display.imshow(self.image, cmap="gray")
        self.clone.flip = not self.clone.flip
        self.clone.find_eye(self.original_image)
        self.clone.find_features(self.original_image)
        self.clone.get_orientation_vectors()
        self.clone.eye_vertices()
        self.clone.find_head(self.original_image)
        self.clone.initialize_dorsal_edge(self.original_image)
        self.clone.fit_dorsal_edge(self.original_image)
        self.edge_image = self.clone.edges
        self.clone.find_tail(self.original_image)
        self.clone.remove_tail_spine()
        self.de = self.clone.interpolate(self.clone.dorsal_edge)
        self.checkpoints = self.clone.checkpoints 
        self.draw()

    def edge_button_press(self, event):
        
        self.edges = not self.edges
        if self.edges:
            self.image = self.edge_image
            self.draw()

        if not self.edges:
            self.image = self.original_image
            self.draw()

    def draw(self):
        
        self.display.clear()
        
        buttoncolor=[0.792156862745098, 0.8823529411764706, 1.0]

        axaddcheck = plt.axes([0.025, 0.085, 0.1, 0.075])
        self.addcheckbutton = Button(axaddcheck, 'Add Checkpoint', color=buttoncolor,hovercolor=buttoncolor)
        self.addcheckbutton.on_clicked(self.add_checkpoint_button_press)

        if self.add_checkpoint:
            self.addcheckbutton.color = "green"
            self.addcheckbutton.hovercolor = "green"
        
        self.display.imshow(self.image, cmap="gray")
        
        axaccept = plt.axes([0.875, 0.7, 0.1, 0.075])
        self.acceptbutton = Button(axaccept, 'Accept Changes', color=buttoncolor,hovercolor=buttoncolor)
        self.acceptbutton.on_clicked(self.accept)
 
        if self.clone.accepted:
            self.acceptbutton.color = "blue"
            self.acceptbutton.hovercolor = "blue"
       
        axmodified = plt.axes([0.875, 0.6, 0.1, 0.075])
        self.modifiedbutton = Button(axmodified, 'Mark as Modified', color=buttoncolor, hovercolor=buttoncolor)
        self.modifiedbutton.on_clicked(self.modified_clicked)
        
        if self.clone.modified:
            self.modifiedbutton.color = "blue"
            self.modifiedbutton.hovercolor = "blue"

        axmask = plt.axes([0.025, 0.185, 0.05, 0.075])
        self.maskbutton = Button(axmask, 'Mask', color=buttoncolor, hovercolor=buttoncolor)
        self.maskbutton.on_clicked(self.mask)
        
        axunmask = plt.axes([0.075, 0.185, 0.05, 0.075])
        self.unmaskbutton = Button(axunmask, 'Unmask', color=buttoncolor, hovercolor=buttoncolor)
        self.unmaskbutton.on_clicked(self.unmask)

        if self.mask_clicked:
            self.maskbutton.color = "green"
            self.maskbutton.hovercolor = "green"
        
        if self.unmask_clicked:
            self.unmaskbutton.color = "green"
            self.unmaskbutton.hovercolor = "green"

        if self.show_dorsal_edge:
            try:
                self.display.scatter(self.de[:,1], self.de[:,0], c="blue")
            except TypeError:
                pass

            if self.edges:
                checkpoint_color = "yellow"
            else:
                checkpoint_color = "black"
            
            try:
                self.display.scatter(self.checkpoints[:,1], self.checkpoints[:,0], c=checkpoint_color)
            except TypeError:
                pass
        
        self.display.axis('off')
        self.display.set_title(self.clone.filepath, color="black")
        self.display.figure.canvas.draw()

    def get_closest_checkpoint(self, x, y, n=1):

        self.selected = self.checkpoints[np.argsort(np.linalg.norm(self.checkpoints-(x,y), axis=1))[0:n], :]
        if n==1:
            self.selected = self.selected[0]

    def set_closest_checkpoint(self, x, y):
        val = self.checkpoints[np.argmin(np.linalg.norm(self.checkpoints - self.selected, axis=1)), :]
        self.checkpoints[np.argmin(np.linalg.norm(self.checkpoints - self.selected, axis=1)), :] = (x, y)
        
        if self.clone.dist(val,self.clone.head) < 0.0001:
            self.clone.head = (x, y)
            self.clone.get_animal_length()
        if self.clone.dist(val, self.clone.tail_dorsal) < 0.0001:
            self.clone.tail_dorsal = (x,y)
    
    def insert_new_checkpoint(self, x, y):
        
        idx1 = np.argwhere(self.checkpoints == self.selected[0])[0][0]
        idx2 = np.argwhere(self.checkpoints == self.selected[1])[0][0]
        
        x = int(x)
        y = int(y)
        
        if (np.min([idx1, idx2]) == 0) and (np.dot(self.checkpoints[0,:] - np.array((x,y)), self.checkpoints[1,:] - np.array((x,y))) > 0):
                self.checkpoints = np.vstack([(x,y), self.checkpoints])
                self.clone.head = (x, y)
        elif (np.max([idx1, idx2]) == len(self.checkpoints)-1) and (np.dot(self.checkpoints[-1,:] - np.array((x,y)), self.checkpoints[-2,:] - np.array((x,y))) > 0):
                self.checkpoints = np.vstack([self.checkpoints, (x,y)])
                self.clone.tail_dorsal = np.array((x, y))

        else:
            self.checkpoints = np.vstack([self.checkpoints[0:np.min([idx1, idx2])+1,:], (x, y), self.checkpoints[np.max([idx1, idx2]):]])

    def delete_selected_checkpoint(self, event):

        if self.selected is not None:
            self.checkpoints = np.delete(self.checkpoints, np.argmin(np.linalg.norm(self.checkpoints - self.selected, axis=1)), axis=0)
            self.clone.checkpoints = self.checkpoints
            self.fit_dorsal()
            self.selected = None
            self.draw()

    def toggle_dorsal_button_press(self, event):

        self.show_dorsal_edge = not self.show_dorsal_edge
        self.draw()

    def add_checkpoint_button_press(self, event):
        
        self.add_checkpoint = not self.add_checkpoint
        self.draw()

    def accept(self, event):
        
        if self.clone.accepted == 0:
            self.clone.accepted = 1
        else:
            self.clone.accepted = 0
            
        self.draw()
    
    def modified_clicked(self, event):

        if self.clone.modified == 0:
            self.clone.modified = 1
        else:
            self.clone.modified = 0

        self.draw()

    def lasso_then_mask(self, verts):
        path = Path(verts)
            
        x,y = np.where(self.edge_image)
        
        for i in xrange(len(x)):
            if path.contains_point([y[i],x[i]]):
                self.edge_image[x[i]][y[i]] = 0
        
        self.image = self.edge_image

        self.clone.initialize_dorsal_edge(self.original_image, dorsal_edge_blur=self.edge_blur, edges=self.edge_image)
        self.fit_dorsal()
        self.draw()

    def mask(self, event):
        
        self.mask_clicked = not self.mask_clicked
        
        if self.mask_clicked:
            
            self.unmask_clicked = False

            self.lasso = LassoSelector(self.display, onselect=self.lasso_then_mask)
            self.edges = False
            self.edge_button_press(1)
        else:
            self.lasso = 0
        
        self.draw()
    
    def lasso_then_unmask(self, verts):
        
        path = Path(verts)

        x,y = np.where(self.original_edge_image)

        for i in xrange(len(x)):
            if path.contains_point([y[i],x[i]]):
                self.edge_image[x[i]][y[i]] = 1
        
        self.image = self.edge_image

        self.clone.initialize_dorsal_edge(self.original_image, dorsal_edge_blur=self.edge_blur, edges=self.edge_image)
        self.fit_dorsal()
        self.draw()
        
    def unmask(self, event):    
        
        self.unmask_clicked = not self.unmask_clicked
        
        if self.unmask_clicked:
            
            self.mask_clicked = False
            self.lasso = LassoSelector(self.display, onselect=self.lasso_then_unmask)
            self.edges = False
            self.edge_button_press(1)

        else:
            self.lasso = 0

        self.draw()

class Viewer:

    def __init__(self, gui_params, params):

        self.curation_data = utils.load_manual_curation(params['curation_csvpath'])
        self.males_list = utils.load_male_list(params['male_listpath'])
        self.induction_dates = utils.load_induction_data(params['induction_csvpath'])
        self.season_data = utils.load_pond_season_data(params['pond_season_csvpath'])
        self.early_release = utils.load_release_data(params['early_release_csvpath'])
        self.late_release = utils.load_release_data(params['late_release_csvpath'])
        self.duplicate_data = utils.load_duplicate_data(params['duplicate_csvpath'])
        self.experimenter_data, self.inducer_data = utils.load_experimenter_data(params['experimenter_csvpath'])
        
        self.gui_params = gui_params
        self.params = params
        self.params.update(self.gui_params)
        
        file_list = []
        metadata_list = []
        print "Reading in image file list"

        with open(self.params["image_list_filepath"], "rb") as f:
            line = f.readline().strip()
            while line:
                file_list.append(line)
                line = f.readline().strip()

        print "Reading in analysis file"

        try:
            self.data = utils.csv_to_df(self.params["output_analysis_file"])
        except IOError:
            self.data = utils.csv_to_df(self.params["input_analysis_file"])
        
        if not hasattr(self.data, 'accepted'):
            self.data['accepted'] = np.zeros(len(self.data))
       
        try:
            self.shape_data = utils.read_shape_long(self.params["output_shape_file"]).set_index('filebase')
        except IOError:
            self.shape_data = utils.read_shape_long(self.params["input_shape_file"]).set_index('filebase')
        
        all_clone_list = []
        clone_list = []

        for f in file_list:
            try:
                fileparts = f.split("/")
                clone = utils.dfrow_to_clone(self.data, np.where(self.data.filebase == fileparts[-1])[0][0], self.params)
                clone.filepath = f
            
                if not (int(self.params['skip_accepted']) and clone.accepted):
                    clone_list.append(clone)
                all_clone_list.append(clone)

            except IndexError:
                print "No entry found in datafile for " + str(f)

        if len(clone_list) == 0:
            print "Either image list is empty or they have all been 'accepted'"
            return

        for i in xrange(len(clone_list)):
            
            clone_list[i].dorsal_edge = np.transpose(np.vstack((np.transpose(self.shape_data.loc[clone_list[i].filebase].x),
                np.transpose(self.shape_data.loc[clone_list[i].filebase].y))))
            clone_list[i].q = self.shape_data.loc[clone_list[i].filebase].q
            clone_list[i].qi = self.shape_data.loc[clone_list[i].filebase].qi
            idx = self.shape_data.loc[clone_list[i].filebase].checkpoint==1
            clone_list[i].checkpoints = clone_list[i].dorsal_edge[idx,:]

        self.all_clone_list = all_clone_list
        self.clone_list = clone_list

        self.curr_idx = 0
        self.clone = self.clone_list[self.curr_idx]
        
        self.saving = 0
        self.fig = plt.figure(figsize=(15,10))
        self.fig.patch.set_facecolor("lightgrey")
        self.display = self.fig.add_subplot(111)
        self.display.axis('off')
        
        try: 
            self.obj = PointFixer(self.clone, self.display)
        except IOError:
            
            try:
                self.clone.modification_notes += " Image file not found."
            except TypeError:
                self.clone.modification_notes = "Image file not found."

            if self.curr_idx < len(self.clone_list)-1:
                self.next_button_press(1)
        
        self.obj.clone.modifier = self.gui_params["default_modifier"]
        self.populate_figure()
        self.add_checkpoint = False
        
        return

    def populate_figure(self):
        
        self.display.clear()
        
        button_color = [0.792156862745098, 0.8823529411764706, 1.0]

        if self.saving == 1:
            self.saving_text = self.fig.text(0.875, 0.879, 'Saving', fontsize=10,fontweight='bold',color=[0.933,0.463,0])
        else:
            try:
                self.saving_text.remove()
            except AttributeError:
                pass

        self.notes_text = self.fig.text(0.852, 0.575, 'Notes', fontsize=10, color="black") 
        
        axmodnotes = plt.axes([0.852, 0.45, 0.125, 0.12])

        initial_modnotes = str(self.obj.clone.modification_notes)
        if initial_modnotes == "nan":
            initial_modnotes = ""

        self.notestextbox = TextBox(axmodnotes,'',initial=initial_modnotes)
        self.notestextbox.on_submit(self.set_notes)
        
        axmodifier = plt.axes([0.875, 0.37, 0.1, 0.035])
        self.modifiertextbox = TextBox(axmodifier,'Initials',initial=str(self.params['default_modifier']))
        self.modifiertextbox.on_submit(self.change_default_modifier)

        axflip = plt.axes([0.50, 0.01, 0.1, 0.075])
        self.flipbutton = Button(axflip, 'Flip', color=button_color, hovercolor=button_color)
        self.flipbutton.on_clicked(self.obj.flip_button_press)

        axedges = plt.axes([0.60, 0.01, 0.1, 0.075])
        self.edgebutton = Button(axedges, 'Toggle Edges', color=button_color, hovercolor=button_color)
        self.edgebutton.on_clicked(self.obj.edge_button_press)

        axtogdorsal = plt.axes([0.70, 0.01, 0.1, 0.075])
        self.togdorsalbutton = Button(axtogdorsal, 'Toggle Dorsal Fit', color=button_color, hovercolor=button_color)
        self.togdorsalbutton.on_clicked(self.obj.toggle_dorsal_button_press)
        
        axdel = plt.axes([0.025, 0.01, 0.1, 0.075])
        self.delcheckbutton = Button(axdel, 'Delete Checkpoint', color=button_color, hovercolor=button_color)
        self.delcheckbutton.on_clicked(self.obj.delete_selected_checkpoint)
        
        axsave = plt.axes([0.875, 0.8, 0.1, 0.075])
        self.savebutton = Button(axsave, 'Save', color=button_color, hovercolor=button_color)
        self.savebutton.on_clicked(self.save)
        
        axblur = plt.axes([0.25, 0.01, 0.1, 0.035])
        self.blurtextbox = TextBox(axblur, 'Gaussian Blur StDev',initial=str(self.obj.edge_blur))
        self.blurtextbox.on_submit(self.set_edge_blur)

        axreset = plt.axes([0.40, 0.01, 0.1, 0.075])
        self.resetbutton = Button(axreset, 'Reset', color=button_color, hovercolor=button_color)
        self.resetbutton.on_clicked(self.reset_button_press)

        if self.curr_idx+1 < len(self.clone_list):
            axnext = plt.axes([0.875, 0.01, 0.1, 0.075])
            self.nextbutton = Button(axnext, 'Next', color=button_color, hovercolor=button_color)
            self.nextbutton.on_clicked(self.next_button_press)

        if self.curr_idx > 0:
            axprev = plt.axes([0.875, 0.085, 0.1, 0.075])
            self.prevbutton = Button(axprev, 'Previous', color=button_color, hovercolor=button_color)
            self.prevbutton.on_clicked(self.prev_button_press)
         
        self.obj.draw()

    def set_notes(self, text):

        try:
            self.obj.clone.modification_notes = str(text)
            self.obj.draw()
        except ValueError:
            print "Invalid value for notes"

    def set_edge_blur(self, text):
        
        try:
            self.obj.edge_blur = float(text)
        except ValueError:
            print "Invalid value for gaussian blur sigma"

        self.obj.clone.initialize_dorsal_edge(self.obj.original_image, dorsal_edge_blur = self.obj.edge_blur)
        self.obj.clone.fit_dorsal_edge(self.obj.original_image, dorsal_edge_blur = self.obj.edge_blur)
        self.obj.de = self.obj.clone.interpolate(self.obj.clone.dorsal_edge)
        self.obj.edge_image = self.obj.clone.edges
        self.obj.checkpoints = self.obj.clone.checkpoints
        
        self.obj.edges = False
        self.obj.edge_button_press(1)
        
        self.obj.draw()
    
    def change_default_modifier(self, text):
        
        self.gui_params['default_modifier'] = str(text)
        
        with open("gui_params.txt","w+") as f:
            for k, v in gui_params.items():
                f.write(str(k) + ", " + str(v) + "\n")

        self.obj.clone.modifier = self.gui_params["default_modifier"]

    def prev_button_press(self,event):

        self.clone_list[self.curr_idx] = self.obj.clone

        self.curr_idx -= 1
        self.clone = self.clone_list[self.curr_idx]
        
        self.fig.clear()

        self.display = self.fig.add_subplot(111)
        self.display.axis('off')

        self.obj = PointFixer(self.clone, self.display)
        self.obj.clone.modifier = self.gui_params["default_modifier"]

        self.populate_figure()
        return


    def next_button_press(self,event):
        try:
            self.clone_list[self.curr_idx] = self.obj.clone
        except AttributeError:
            self.clone_list[self.curr_idx] = self.clone

        self.curr_idx += 1
        self.clone = self.clone_list[self.curr_idx]
        
        self.fig.clear()

        self.display = self.fig.add_subplot(111)
        self.display.axis('off')

        try:

            self.obj = PointFixer(self.clone, self.display)
            self.obj.clone.modifier = self.gui_params["default_modifier"]
            self.populate_figure()

        except IOError:
            
            try:
                self.clone.modification_notes += " Image file not found."
            except TypeError:
                self.clone.modification_notes = "Image file not found."
            
            if self.curr_idx < len(self.clone_list)-1:
                self.next_button_press(event)
        
    def reset_button_press(self, event):
        
        if self.obj.clone.flip:
            self.obj.clone.flip = not self.obj.clone.flip
        
        self.obj.clone.find_features(self.obj.original_image, **self.obj.params)
        self.obj.clone.get_orientation_vectors()
        self.obj.clone.find_head(self.obj.original_image, **self.obj.params)
        self.obj.edge_blur = 1.0
        self.blurtextbox.set_val('1.0')
        self.obj.clone.dorsal_blur = 1.0
        self.obj.clone.initialize_dorsal_edge(self.obj.original_image, **self.obj.params)
        self.obj.clone.fit_dorsal_edge(self.obj.original_image, **self.obj.params)
        self.obj.clone.find_tail(self.obj.original_image)
        self.obj.edge_image = self.obj.clone.edges
        self.obj.clone.remove_tail_spine()
        self.obj.de = self.obj.clone.interpolate(self.obj.clone.dorsal_edge)
        self.obj.selected = None 
        self.obj.checkpoints = self.obj.clone.checkpoints
        
        self.obj.draw()


    def save(self, event):

        self.clone_list[self.curr_idx] = self.obj.clone

        for all_c in xrange(len(self.all_clone_list)):
            for c in xrange(len(self.clone_list)):
                if self.all_clone_list[all_c].filebase == self.clone_list[c].filebase:
                    self.all_clone_list[all_c] = self.clone_list[c]

        self.saving = 1
        self.populate_figure()

        with open(self.params["input_analysis_file"],"rb") as analysis_file_in, open(self.params["output_analysis_file"],"wb") as analysis_file_out:
            
            line = analysis_file_in.readline()
            DATA_COLS = line.split("\t").strip()

            while line:
                written = False
                for clone in self.all_clone_list:
                    if (line.split("\t")[0] == clone.filebase) and clone.accepted:

                        metadata = utils.build_metadata_dict(clone.filepath,
                                self.curation_data,
                                self.males_list,
                                self.induction_dates,
                                self.season_data,
                                self.early_release,
                                self.late_release,
                                self.duplicate_data,
                                self.experimenter_data,
                                self.inducer_data,
                                pixel_to_mm=clone.pixel_to_mm)
                        analysis_file_out.write(utils.clone_to_line(clone, DATA_COLS, metadata.keys(), metadata)+"\n")
                        written = True

                if not written:    
                    analysis_file_out.write(line)

                line = analysis_file_in.readline()
        
        with open(self.params["input_shape_file"],"rb") as shape_file_in, open(self.params["output_shape_file"],"wb") as shape_file_out:
            
            line = shape_file_in.readline()

            while line:
                for clone in self.clone_list:
                    if (line.split("\t")[0] == clone.filepath) and clone.accepted:
                        
                        for i in np.arange(len(clone.dorsal_edge)):
                            if len(np.where((clone.checkpoints==clone.dorsal_edge[i,:]).all(axis=1))[0]) > 0:
                                checkpoint = 1
                            else:
                                checkpoint = 0
                            shape_file_out.write('\t'.join([clone.filepath, str(i), str(clone.dorsal_edge[i, 0]), str(clone.dorsal_edge[i,1]), str(clone.qi[i]), str(clone.q[i]), str(checkpoint)]) + "\n")
                            line = shape_file_in.readline() # so we skip past all of the lines we are overwriting

                    else:
                        shape_file_out.write(line)
                        line = shape_file_in.readline()
        
        self.saving = 0
        self.populate_figure()
"""
        with open(self.params["input_analysis_metadata_file"],"rb") as analysis_file_in, open(self.params["output_analysis_metadata_file"],"wb") as analysis_file_out:
            
            line = analysis_file_in.readline()

            while line:
                for clone in self.clone_list:
                    if (line.split("\t")[0] == clone.filepath) and clone.accepted:
                        analysis_file_out.write(clone_to_line(clone, DATA_COLS, METADATA_FIELDS, {m:getattr(clone,m) for m in METADATA_FIELDS}))
                else:
                    analysis_file_out.write(line)

                line = analysis_file_in.readline()
"""
#

gui_params = utils.myParse("gui_params.txt")
params = utils.myParse("params.txt")

v = Viewer(gui_params, params)
plt.show()
