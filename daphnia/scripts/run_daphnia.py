import click
import cv2
import sys, os
from daphnia.clone import Clone
from daphnia.daphnia_plot import plot as daphnia_plot
from ast import literal_eval
import daphnia.utils as utils
import numpy as np

def analyze_clone(clone, im):

    try:
        print "Detecting eye"
        clone.find_eye(im)
        print "Masking antenna"
        clone.find_features(im)
        print "Estimating area"
        clone.get_orientation_vectors()
        clone.eye_vertices()

        clone.find_head(im)
        clone.initialize_dorsal_edge(im)
        clone.fit_dorsal_edge(im)
        clone.find_tail(im)
        clone.remove_tail_spine()

        clone.get_animal_length()
        clone.get_animal_dorsal_area()

        print "Fitting and analyzing pedestal"
        clone.qscore()

        clone.analyze_pedestal()
    except Exception as e:
        print "Error analyzing " + str(clone.filepath) + ": " + str(e)

@click.command()
@click.option('--params', default='params.txt', help='Path to parameter file.')
@click.option('--plot', is_flag=True, help='Generates overlay on input image')
@click.option('--plot_params', default='plot_params.txt', help='Path to plot parameter file.')
@click.argument('images', nargs=-1,type=click.Path(exists=True))

def daphnia(params, images, plot, plot_params):
    
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
            "eye_ventral",
            "head",
            "tail",
            "tail_tip",
            "tail_base",
            "tail_spine_length",
            "tail_dorsal",
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
            "accepted",
            "modified",
            "modification_notes",
            "modifier",
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

    params_dict = utils.myParse(params)

    if plot:
        
        plot_params_dict = utils.myParse(plot_params)
    
    if params_dict['load_metadata']:
        
        curation_data = utils.load_manual_curation(params_dict['curation_csvpath'])
        males_list = utils.load_male_list(params_dict['male_listpath'])
        induction_dates = utils.load_induction_data(params_dict['induction_csvpath'])
        season_data = utils.load_pond_season_data(params_dict['pond_season_csvpath'])
        early_release = utils.load_release_data(params_dict['early_release_csvpath'])
        late_release = utils.load_release_data(params_dict['late_release_csvpath'])
        duplicate_data = utils.load_duplicate_data(params_dict['duplicate_csvpath'])
        experimenter_data, inducer_data = utils.load_experimenter_data(params_dict['experimenter_csvpath'])

    for image_filepath in images:
        
        click.echo('Analyzing {0}'.format(image_filepath))
        clone = Clone(image_filepath, **params_dict)
        im = cv2.imread(clone.filepath, cv2.IMREAD_GRAYSCALE)
        analyze_clone(clone, im)
        
        if params_dict['load_metadata']:
            metadata = utils.build_metadata_dict(image_filepath, curation_data, males_list, induction_dates, season_data, early_release, late_release, duplicate_data, experimenter_data, inducer_data)

        utils.write_clone(clone, DATA_COLS, metadata.keys(), metadata, params_dict['output'], params_dict['shape_output'])
        utils.write_analysis_metadata(clone, params_dict, params_dict['analysis_metadata_output'])
        if plot:
            clone.filebase = metadata['filebase']
            daphnia_plot(clone, im, plot_params_dict)

if __name__ == '__main__':
    daphnia()
