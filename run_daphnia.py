import click
import cv2
from daphnia.clone import Clone

@click.command()
@click.option('--params', default='', help='Path to parameter file.')
@click.option('--output', default='daphnia_output.txt', help='Path to output file.')
@click.option('--pedoutput',default='pedestal_output.txt', help='Path to output file for pedestal fit data.')
@click.argument('i', nargs=-1,type=click.Path(exists=True))

DATA_COLS = ["filepath",
        "total_animal_pixels",
        "total_eye_pixels",
        "animal_length_pixels",
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
        "ventral_mask_endpoints",
        "dorsal_mask_endpoints",
        "anterior_mask_endpoints",
        "posterior_mask_endpoints",
        "pedestal_max_height_pixels",
        "pedestal_area_pixels",
        "poly_coeff",
        "res",
        "peak",
        "deyecenter_pedestalmax_pixels"]

def analyze_clone(clone):

    try:

        im = cv2.imread(clone.filepath, cv2.IMREAD_GRAYSCALE)

        clone.find_eye(im)
        clone.get_eye_area()

        clone.mask_antenna(im)
        clone.count_animal_pixels(im)
        clone.get_animal_area()
        
        clone.find_tail()
        clone.get_orientation_vectors()
    
        clone.get_eye_vector("dorsal")
        clone.get_animal_length()

        clone.initialize_pedestal(im)
        clone.fit_pedestal(im)
        clone.analyze_pedestal()

    except Exception as e:
        print "Error analyzing " + clone.filepath + ": " + str(e)

def main(params, i):

    if params:
        params_dict = {}
        with open(params) as f:
            line = f.readline()
            while line:
                param_name, param_value = line.split(',')
                params_dict[param_name] = param_value
                line = f.readline()

    for image_filepath in i:
        clone = Clone(image_filepath)
        analyze_clone(clone)

