import click
import cv2
import sys, os
from daphnia.clone import Clone

def analyze_clone(clone, params):

    try:

        im = cv2.imread(clone.filepath, cv2.IMREAD_GRAYSCALE)

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

    except Exception as e:
        print "Error analyzing " + clone.filepath + ": " + str(e)

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

@click.command()
@click.option('--params', default='params.txt', help='Path to parameter file.')
@click.option('--output', default='daphnia_output.txt', help='Path to output file.')
@click.option('--pedoutput',default='pedestal_output.txt', help='Path to output file for pedestal fit data.')
@click.argument('i', nargs=-1,type=click.Path(exists=True))
def daphnia(params, i, output, pedoutput):
    
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


    if params:
        params_dict = {}

        with open(params) as f:
            line = f.readline()
            while line:
                param_name, param_value = line.split(',')
                params_dict[param_name] = float(param_value.strip())
                line = f.readline()
    
    for image_filepath in i:
        
        clone = Clone(image_filepath)
        analyze_clone(clone, params_dict)
        write_clone(clone, DATA_COLS, output, pedoutput)

        click.echo('Analyzing {0}'.format(image_filepath))

if __name__ == '__main__':
    daphnia()
