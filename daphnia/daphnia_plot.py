import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def plot_eye_mask(clone, eye_mask=0, eye_mask_color="yellow", eye_mask_alpha=0.1, eye_mask_markersize=1,**kwargs):
    
    if eye_mask:
        plt.scatter(clone.eye_pts[:,1], clone.eye_pts[:,0],c=eye_mask_color,alpha=eye_mask_alpha,s=eye_mask_markersize)

def plot_antenna_mask(clone, im, antenna_mask=0, antenna_mask_color="blue", antenna_mask_linewidth=3, **kwargs):

    if antenna_mask:

        edge_x = np.where(clone.edge_copy)[0]
        edge_y = np.where(clone.edge_copy)[1]

        im[edge_x, edge_y] = 0
        
        plt.plot( (clone.ventral_mask_endpoints[0][1], clone.ventral_mask_endpoints[1][1]),
                (clone.ventral_mask_endpoints[0][0], clone.ventral_mask_endpoints[1][0]),
                c=antenna_mask_color, linewidth=antenna_mask_linewidth)

        plt.plot( (clone.dorsal_mask_endpoints[0][1], clone.dorsal_mask_endpoints[1][1]),
                (clone.dorsal_mask_endpoints[0][0], clone.dorsal_mask_endpoints[1][0]),
                c=antenna_mask_color, linewidth=antenna_mask_linewidth)

        plt.plot( (clone.anterior_mask_endpoints[0][1], clone.anterior_mask_endpoints[1][1]),
                (clone.anterior_mask_endpoints[0][0], clone.anterior_mask_endpoints[1][0]),
                c=antenna_mask_color, linewidth=antenna_mask_linewidth)

        plt.plot( (clone.posterior_mask_endpoints[0][1], clone.posterior_mask_endpoints[1][1]),
                (clone.posterior_mask_endpoints[0][0], clone.posterior_mask_endpoints[1][0]),
                c=antenna_mask_color, linewidth=antenna_mask_linewidth)

def plot_landmarks(clone, landmarks=1,
        landmark_head=1,
        landmark_tail=1,
        landmark_tail_tip=1,
        landmark_animal_center=1,
        landmark_eye_center=1,
        landmark_eye_dorsal=1,
        landmark_color="red",
        landmark_style="pixel",
        landmark_size=3, **kwargs):
    
    if landmarks:
        
        if landmark_style == "pixel":
            style = ','

        if landmark_head:
            plt.scatter(clone.head[1], clone.head[0], c=landmark_color, marker=style, s=landmark_size)

        if landmark_tail: 
            plt.scatter(clone.tail[1], clone.tail[0], c=landmark_color, marker=style, s=landmark_size)

        if landmark_tail_tip:
            plt.scatter(clone.tail_tip[1], clone.tail_tip[0], c=landmark_color, marker=style, s=landmark_size)
        
        if landmark_animal_center:
            plt.scatter(clone.animal_y_center, clone.animal_x_center, c=landmark_color, marker=style, s=landmark_size)

        if landmark_eye_center:
            plt.scatter(clone.eye_y_center, clone.eye_x_center, c=landmark_color, marker=style, s=landmark_size)

        if landmark_eye_dorsal:
            plt.scatter(clone.eye_dorsal[1], clone.eye_dorsal[0], c=landmark_color, marker=style, s=landmark_size)

def plot_animal_length(clone, animal_length_plot=1, animal_length_plot_color="black", animal_length_linewidth=2, **kwargs):

    if animal_length_plot:

        plt.plot( (clone.tail[1], clone.head[1]), (clone.tail[0], clone.head[0]), c=animal_length_plot_color, linewidth=animal_length_linewidth)

def plot_tail_spine_length(clone, tail_spine_length_plot=1, tail_spine_length_plot_color="black", tail_spine_length_linewidth=2, **kwargs):

    if tail_spine_length_plot:

        plt.plot( (clone.tail_base[1], clone.tail_tip[1]), (clone.tail_base[0], clone.tail_tip[0]), c=tail_spine_length_plot_color, linewidth=tail_spine_length_linewidth)

def plot_animal_perimeter(clone, animal_perimeter=1, animal_perimeter_style="line", animal_perimeter_color="green", animal_perimeter_linewidth=2,**kwargs):

    if animal_perimeter:

        pts = np.array(clone.whole_animal_points)
        if animal_perimeter_style == "line":
            for i in np.arange(pts.shape[0]-1):
                plt.plot( (pts[i,0], pts[i+1,0]), (pts[i,1],pts[i+1,1]), c=animal_perimeter_color, linewidth=animal_perimeter_linewidth)
            plt.plot( (pts[0, 0], pts[-1, 0]), (pts[0, 1], pts[-1, 1]), c=animal_perimeter_color)

        elif animal_perimeter_style == "points":
            
            plt.scatter(pts[:,0], pts[:,1], c=animal_perimeter_color)

def plot_pedestal(clone, pedestal_plot=1,
    pedestal_plot_color="blue",
    pedestal_plot_alpha=1.0,
    pedestal_plot_marker_size=2,
    pedestal_plot_window_highlight=0,
    pedestal_plot_window_highlight_color="red", **kwargs):

    if pedestal_plot:
        p = clone.pedestal
        p = np.array([list(x) for x in p])
        plt.scatter(p[:,1], p[:,0], c=pedestal_plot_color, s=pedestal_plot_marker_size, alpha=pedestal_plot_alpha)

def plot(clone, im, plot_params):

    f = plt.figure(figsize=(20,20))

    plot_eye_mask(clone, **plot_params) 
    plot_antenna_mask(clone, im, **plot_params)
    plot_landmarks(clone, **plot_params)
    plot_animal_length(clone, **plot_params)
    plot_tail_spine_length(clone, **plot_params)
    plot_animal_perimeter(clone, **plot_params)
    plot_pedestal(clone, **plot_params)
    
    plt.imshow(im, cmap="gray")
    plt.axis('off')

    plt.savefig( os.path.join( plot_params["daphnia_plot_dir"], plot_params["daphnia_plot_name"] + "." + plot_params["daphnia_plot_format"]),
            format=plot_params["daphnia_plot_format"],
            dpi=int(plot_params["daphnia_plot_resolution"]))
