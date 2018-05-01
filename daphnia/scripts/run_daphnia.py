import click
import cv2
import sys, os
from daphnia.clone import Clone
from daphnia.daphnia_plot import plot as daphnia_plot
from ast import literal_eval

def analyze_clone(clone, im, params):

#    try:
        clone.find_eye(im, **params)

        clone.mask_antenna(im, **params)
        clone.count_animal_pixels(im, **params)
        
        clone.find_tail(im, **params)
        clone.get_orientation_vectors()
        clone.find_head(im, **params)

        clone.get_animal_length()

        clone.initialize_pedestal(im, **params)
        clone.fit_pedestal(im, **params)
        clone.analyze_pedestal(**params)

#    except Exception as e:
#        print "Error analyzing " + clone.filepath + ": " + str(e)

def write_clone(clone, cols, output, ped_output):
    
    try:
   	if os.stat(output).st_size == 0:
            raise OSError
    except (OSError):
	print "Starting new output file"
	with open(output, "wb+") as f:
	    f.write( "\t".join(cols) + "\n")

    try:
   	if os.stat(ped_output).st_size == 0:
            raise OSError
    except (OSError):
	print "Starting new pedestal output file"
	with open(ped_output, "wb+") as f:
	    f.write( "\t".join(["filepath","pedestal_data"]) + "\n")

    try:
        with open(output, "ab+") as f:

            tmpdata = []

            for c in cols:

                val = str(getattr(clone, c))

                if val is not None:
                    tmpdata.append( val )
                else:
                    tmpdata.append("")

            f.write( "\t".join(tmpdata) + "\n")

        with open(ped_output, "ab+") as f:

            f.write('\t'.join([clone.filepath, str(clone.pedestal)]))
    
    except (IOError, AttributeError) as e:
        print "Can't write data for " + clone.filepath + " to file: " + str(e)

def myParse(params):
    
    params_dict = {}

    with open(params) as f:
        for line in f:
            if not line.startswith("#"):
                try:
                    param_name, param_value = line.split(',') 
                    try:
                        params_dict[param_name] = literal_eval(param_value.strip())
                    except (ValueError, SyntaxError):
                        params_dict[param_name] = param_value.strip()
                except ValueError:
                    pass
    return params_dict

@click.command()
@click.option('--params', default='params.txt', help='Path to parameter file.')
@click.option('--plot', is_flag=True, help='Generates overlay on input image')
@click.option('--plot_params', default='plot_params.txt', help='Path to plot parameter file.')
@click.argument('images', nargs=-1,type=click.Path(exists=True))
def daphnia(params, images, plot, plot_params):
    
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
            "pedestal_max_height_pixels",
            "pedestal_area_pixels",
            "poly_coeff",
            "res",
            "peak",
            "deyecenter_pedestalmax_pixels"]


    params_dict = myParse(params)

    if plot:
        plot_params_dict = myParse(plot_params)

    for image_filepath in images:
        
        click.echo('Analyzing {0}'.format(image_filepath))
        
        clone = Clone(image_filepath)
        im = cv2.imread(clone.filepath, cv2.IMREAD_GRAYSCALE)
        analyze_clone(clone, im, params_dict)
        write_clone(clone, DATA_COLS, params_dict['output'], params_dict['ped_output'])

        if plot:
            daphnia_plot(clone, im, plot_params_dict)
            

if __name__ == '__main__':
    daphnia()
