import click
import cv2
import sys, os
from daphnia.clone import Clone
from daphnia.daphnia_plot import plot as daphnia_plot
from ast import literal_eval
import daphnia.utils as utils
import numpy as np

def analyze_clone(clone, im, params):

    try:
        print "Detecting eye"
        clone.find_eye(im, **params)
        print "Masking antenna"
        clone.find_features(im, **params)
        print "Estimating area"
        #clone.count_animal_pixels(im, **params)
        clone.get_orientation_vectors()
        clone.eye_vertices()
        
        clone.find_tail(im, **params)
        clone.find_head(im, **params)

        clone.get_animal_length()
        print "Fitting and analyzing pedestal"
        clone.get_dorsal_edge(im,**params)
        clone.qscore()
    except Exception as e:
        print "Error analyzing " + str(clone.filepath) + ": " + str(e)

def write_clone(clone, cols, metadata_fields, metadata, output, shape_output):

    try:
        if (os.stat(output).st_size == 0):
            raise OSError
    except (OSError):
        print "Starting new output file"
        try:
            with open(output, "wb+") as f:
                f.write( "\t".join(metadata_fields)+ "\t" + "\t".join(cols) + "\n")
        except IOError:
            print "Can't find desired location for saving data"
            pass
    
    try:
        if (os.stat(shape_output).st_size == 0):
            raise OSError
    except (OSError):
        print "Starting new pedestal output file"
        
        try:
            with open(shape_output, "wb+") as f:
                f.write( "\t".join(["filepath","i","x","y","qi","q"]) + "\n")
        except IOError:
            print "Can't find desired location for saving data"

    try:
        with open(output, "ab+") as f:

            tmpdata = []
            for mf in metadata_fields:
                try:
                    if mf == "animal_area_mm":
                        val = getattr(clone,'animal_area')/np.power(metadata["pixel_to_mm"],2)
                    elif mf == "eye_area_mm":
                        val = getattr(clone,'eye_area')/np.power(metadata["pixel_to_mm"],2)
                    elif mf == "animal_length_mm":
                        val = getattr(clone,'animal_length')/metadata["pixel_to_mm"]
                    elif mf == "tail_spine_length_mm":
                        val = getattr(clone,'tail_spine_length')/metadata["pixel_to_mm"]
                    elif mf == "pedestal_area_mm":
                        val = getattr(clone,'pedestal_area')/np.power(metadata["pixel_to_mm"],2)
                    elif mf == "pedestal_max_height_mm":
                        val = getattr(clone,'pedestal_max_height')/metadata["pixel_to_mm"]
                    elif mf == "deyecenter_pedestalmax_mm":
                        val = getattr(clone,'deyecenter_pedestalmax')/metadata["pixel_to_mm"]
                    else:
                        val = metadata[mf]

                except KeyError:
                    val = ''
                
                tmpdata.append( str(val) )

            for c in cols:

                val = str(getattr(clone, c))

                if val is not None:
                    tmpdata.append( val )
                else:
                    tmpdata.append("")

            f.write( "\t".join(tmpdata) + "\n")
        try:
            with open(shape_output, "ab+") as f:
                for i in np.arange(len(clone.dorsal_edge)):
                    f.write('\t'.join([clone.filepath, str(i), str(clone.dorsal_edge[i,0]), str(clone.dorsal_edge[i,1]), str(clone.qi[i]), str(clone.q[i])]) + '\n')

        except Exception as e:
            print "Error writing pedestal to file: " + str(e)
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
            "animal_area",
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
            "tail_spine_length",
            "ventral_mask_endpoints",
            "dorsal_mask_endpoints",
            "anterior_mask_endpoints",
            "pedestal_max_height",
            "pedestal_area",
            "poly_coeff",
            "res",
            "peak",
            "deyecenter_pedestalmax"]

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
            "animal_area_mm",
            "animal_length_mm",
            "eye_area_mm",
            "tail_spine_length_mm",
            "deyecenter_pedestalmax_mm",
            "pedestal_area_mm",
            "pedestal_max_height_mm"]

    params_dict = myParse(params)

    if plot:
        plot_params_dict = myParse(plot_params)
    
    if params_dict['load_metadata']:
        curation_data = utils.load_manual_curation(params_dict['curation_csvpath'])
        males_list = utils.load_male_list(params_dict['male_listpath'])
        induction_dates = utils.load_induction_data(params_dict['induction_csvpath'])
        season_data = utils.load_pond_season_data(params_dict['pond_season_csvpath'])

    for image_filepath in images:
        
        click.echo('Analyzing {0}'.format(image_filepath))
        
        clone = Clone(image_filepath)
        im = cv2.imread(clone.filepath, cv2.IMREAD_GRAYSCALE)
        analyze_clone(clone, im, params_dict)
        
        if params_dict['load_metadata']:
            try:
                metadata = utils.build_metadata_dict(image_filepath, curation_data, males_list, induction_dates, season_data)
            except Exception:
                print "Error gathering metadata: " + str(e)
        write_clone(clone, DATA_COLS, METADATA_FIELDS, metadata, params_dict['output'], params_dict['shape_output'])

        if plot:
            clone.filebase = metadata['filebase']
            daphnia_plot(clone, im, plot_params_dict)

if __name__ == '__main__':
    daphnia()
