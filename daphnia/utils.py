from __future__ import division
from clone import Clone
import pickle
import os
import numpy as np
import pandas as pd
import re
from collections import defaultdict
from openpyxl import load_workbook 
from ast import literal_eval
import cv2
import scipy

def save_pkl(obj, path, name):
    with open(os.path.join(path, name) + '.pkl','wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_pkl(name, path):
    with open(os.path.join(path,name) + '.pkl','rb') as f:
        return pickle.load(f)

def parsePond(s):
    
    # this method parses clone ids and finds pond name and id
    #
    # if only the pond is given (e.g. Cyrus), the id defaults to pond name

    pattern = '^(A?[DW]?\s?\d{1,2}|Cyril|DBunk|Male|C14|Chard)[\s?|\.?|_?]?(A?\d{1,3}A?)?'
    m = re.search(pattern, s)
    pond, cloneid = m.groups()

    if cloneid is None:
        cloneid = pond

    return pond, cloneid

def parse(s):

    pattern = '^([A-Za-z]{4,10})_(\d{6})?_?(DBunk_?\s?\d{1,3}|Male_\d|A?[DW]?\s?\d{1,2}[_|\s|.]?A?\d{1,3}A?|Cyril|C14|Chard)_(juju\d?|ctrl|NA|(?:\d*\.)?\d+)_(\d[A-Z]|NA)_(Rig[AB])_(\d{8}T\d{6}).bmp$'
    
    m = re.search(pattern,s)
    filetype,barcode,cloneid,treatment,replicate,rigId,datetime = m.groups()
    
    return {"filetype":filetype,
            "barcode":barcode,
            "cloneid":cloneid,
            "treatment":treatment,
            "replicate":replicate,
            "rig":rigId,
            "datetime":datetime}

def build_metadata_dict(filepath, curation_dict, male_list, induction_dict, season_dict, early_release, late_release, duplicate_data, experimenter_data, inducer_data, pixel_to_mm=None):

    _, filebase = os.path.split(filepath)
    md = parse(filebase)
    md['filebase'] = filebase
    md['treatment'] = convert_treatment(md['treatment'])

    # add induction
    for k in ['pond', 'id', 'season']:
        try:
            md[k] = season_dict[md['cloneid']][k]
        except KeyError:
            md[k] = ''
    
    if pixel_to_mm is None:
        try:
            im = cv2.imread(filepath.replace("full","fullMicro"), cv2.IMREAD_GRAYSCALE)
            md['pixel_to_mm'] = calc_pixel_to_mm(im)
        except Exception as e:
            print "Error extracting conversion factor from micrometer: " + str(e)
            md['pixel_to_mm'] = np.nan
    else:
        md['pixel_to_mm'] = pixel_to_mm

    md['manual_PF'] = 'U'
    md['manual_PF_reason'] = ""
    
    if md['barcode'] in male_list:
        md['manual_PF'] = "F"
        md['manual_PF_reason'] = "male"
        md['manual_PF_curator'] = "awe"
    
    if not (md['manual_PF'] == 'F'):
        try:
            row = curation_dict[md['filebase']]
            md['manual_PF'] = row['manual_PF'].upper()
            md['manual_PF_reason'] = row['manual_PF_reason']
            md['manual_PF_curator'] = row['manual_PF_curator'].lower()
        except KeyError:
            md['manual_PF'] = "P"
            md['manual_PF_reason'] = ''
            md['manual_PF_curator'] = "awe"
   
    try:
        md['inductiondate'] = induction_dict[md['barcode']]
    except KeyError:
        md['inductiondate'] = 'NA'

    if md['barcode'] in early_release:
        md['manual_PF_reason'] += "; early release"
    if md['barcode'] in late_release:
        md['manual_PF_reason'] += "; late release"

    try:
        if duplicate_data[md['filebase']] == "Use":
            md['manual_PF_reason'] += "; duplicate"
        else:
            md['manual_PF'] = 'F'
            md['manual_PF_reason'] = ", ".join([md['manual_PF_reason'], "duplicate", duplicate_data[md['filebase']]])
    except KeyError:
        pass

    try:
        md['experimenter'] = experimenter_data[(md['barcode'], md['datetime'][0:8])]
        md['inducer'] = inducer_data[(md['barcode'], md['datetime'][0:8])]
    except KeyError:
        md['experimenter'] = 'NA'
        md['inducer'] = 'NA'
    
    return md

def convert_treatment(treatment):
        
    if treatment is not np.nan:

	if treatment == 'ctrl':
	    return 0.0
	elif treatment == 'juju1':
	    return 0.1
	elif treatment == 'juju2':
	    return 0.25
	elif (treatment == 'juju3') or (treatment == 'juju'):
	    return 0.5
	elif treatment == 'juju4':
	    return 1.0
    
    return treatment

def recursivedict(): 
    # initializes a default dictionary with an arbitrary number of dimensions

    return defaultdict(recursivedict)

def load_induction_data(filepath):
    
    print "Loading induction data\n"
    inductiondates = {}
    inductionfiles = os.listdir(filepath)

    for i in inductionfiles:
        if not i.startswith("._") and (i.endswith(".xlsx") or i.endswith(".xls")):
            print "Loading " + i
            wb = load_workbook(os.path.join(filepath,i), data_only=True)
            data = wb["Inductions"].values
            cols = next(data)[0:]
            data = list(data)
	    df = pd.DataFrame(data, columns=cols)
            df = df[df.ID_Number.notnull()]

            for j,row in df.iterrows():
                if not str(row['ID_Number']) == "NaT":
                    time = pd.Timestamp(row['Induction_Date'])
                    inductiondates[str(int(row['ID_Number']))] = time.strftime("%Y%m%dT%H%M%S")

    return inductiondates

def load_pond_season_data(filepath):

    print "Loading pond and season metadata\n"
    df = pd.read_csv( filepath )

    pond_season_dict = {}

    for i, row in df.iterrows():
        pond_season_dict[row['cloneid']] = {'pond':row['pond'], 'id':row['id'], 'season':row['season']}

    return pond_season_dict

def load_manual_scales(filepath):

     # load manual_scales
    manual_scales = {}
    with open(os.path.join(filepath, "manual_scales.txt"),'rb') as f:
        line = f.readline()
        while line:
            filename,conversion = line.strip().split(',')
            manual_scales[filename] = conversion
            line = f.readline()
    
    return manual_scales

def load_release_data(filepath):
    
    with open(filepath, "rb") as f:
        line = f.readline()
    
    return line.split("\t")

def load_duplicate_data(filepath):

    duplicate_dict = {}

    df = pd.read_csv(filepath, index_col="filebase")

    for i, row in df.iterrows():
        duplicate_dict["full_"+i] = row['Notes']

    return duplicate_dict

def build_clonelist(datadir, analysisdir, inductiondatadir, pondseasondir, ext=".bmp"):
    
    # input: paths to data, segmented data and metadata files

    clones = recursivedict()
   
    inductiondates = load_induction_data(inductiondatadir)
    pond_season_md = load_pond_season_data(pondseasondir)
    manual_scales = load_manual_scales(analysisdir)

    files = os.listdir(datadir)
    
    for f in files:
        
        if f.startswith("._"):
            continue
        
        elif f.endswith(ext) and f.startswith("full_"):
            
            filebase = f[5:]

            print "Adding " + f + " to clone list"
            imagetype,barcode,clone_id,treatment,replicate,rig,datetime = parse(f)
            
            if barcode is not None:
          
                if str(barcode) in inductiondates.iterkeys():
                    induction = inductiondates[str(barcode)]
                else:
                    induction = None
                
                clones[barcode][datetime][imagetype] = Clone( filebase,
                        imagetype,
                        barcode,
                        clone_id,
                        treatment,
                        replicate,
                        rig,
                        datetime,
                        induction,
                        pond_season_md[clone_id]['pond'],
                        pond_season_md[clone_id]['id'],
                        pond_season_md[clone_id]['season'],
                        datadir)
        
                if imagetype == "close":
                    clones[barcode][datetime][imagetype].pixel_to_mm = 1105.33
                try:
                    clones[barcode][datetime][imagetype].pixel_to_mm = manual_scales[clones[barcode].micro_filepath]
                except (KeyError, AttributeError):
                    pass
    
    return clones

def csv_to_df(csvfile, sep="\t"):
    return pd.read_csv(csvfile, sep=sep)

def df_to_clonelist(df, datadir = None):

    clones = recursivedict()
    clf = load_SVM()

    for index, row in df.iterrows():
        clone = Clone( row['filebase'],
		'full',
                row['barcode'],
                row['cloneid'],
                row['treatment'],
                row['replicate'],
                row['rig'],
                row['datetime'],
                row['inductiondate'],
                row['pond'],
                row['id'],
                row['season'],
                datadir,
                clf=clf)

        for k in row.keys():
            try:
                setattr(clone, k, literal_eval(row[k]))
            except (ValueError, SyntaxError):
                setattr(clone, k, row[k])

        clones[str(row['barcode'])][str(row['datetime'])]['full'] = clone
    
    return clones

def dfrow_to_clone(df, irow, params, datadir = None):
    
    row = df.iloc[irow]
    clone = Clone(row['filepath'], **params)

    for k,v in row.iteritems(): 
        try:
            setattr(clone, k, literal_eval(str(v)))
        except (ValueError, SyntaxError):
            setattr(clone, k, v)
    
    return clone

def update_clone_list(clones, loadedclones):

     for barcode in loadedclones.iterkeys():
        for dt in loadedclones[barcode].iterkeys():
            clones[barcode][dt]['full'] = loadedclones[barcode][dt]['full']
            clones[barcode][dt]['full'].analyzed = False
     return clones  

def save_clonelist(clones, path, outfile, cols):
   
    with open(os.path.join(path, outfile), "wb+"):
        f.write( "\t".join(cols) + "\n")
        
    for barcode in clones.iterkeys():
        for dt in clones[barcode].iterkeys():
            clone = clones[barcode][dt]["full"]
            write_clone(clone, cols, path, outfile)

def update_attr(src, dest, attr):

    setattr(dest, attr, getattr(src, attr))

def load_male_list(csvpath):

    df = pd.read_csv(csvpath, header=None)
    return df[0].values

def load_manual_curation(csvpath):

    df = pd.read_csv(csvpath)
    
    curation_data = {}

    for i, row in df.iterrows():
        curation_data[row['filebase'] + ".bmp"] = row
    
    return curation_data

def load_experimenter_data(csvpath):

    df = pd.read_csv(csvpath)
    experimenter_data = {}
    inducer_data = {}

    for i, row in df.iterrows():
        experimenter_data[(str(row['Barcode']),str(row['Date']))] = row['Initials']
        inducer_data[(str(row['Barcode']),str(row['Date']))] = row['Inductions']
    return experimenter_data, inducer_data

def crop(img):
        
    # this method is for cropping out the scale from micrometer images

    # aperture edges mess up image normalization, so we need to figure out which
    # (if any) corners have aperture edges, as well as how far each of the edges
    # extends (since it is not always symmetric)
    
    w,h = img.shape
    
    corners = []
    docrop = False

    # if there are 5 pixels in a row that are very dark, it is most likely a corner
    if np.sum(img[0, 0:np.int(h/2)] < 50) > 5 and np.sum(img[0:np.int(w/2),0] < 50) > 5:
	docrop = True
	corners.append(["topleft",
			np.max(np.where(img[0, 0:np.int(h/2)] < 50)),
			np.max(np.where(img[0:np.int(w/2),0] < 50))])

    if np.sum(img[0, np.int(h/2):] < 50) > 5 and np.sum(img[0:np.int(w/2),h-1] < 50) > 5:
	docrop = True
	corners.append(["topright",
			np.int(h/2) + np.min(np.where(img[0, np.int(h/2):] < 50)),
			np.max(np.where(img[0:np.int(w/2),h-1] < 50))])

    if np.sum(img[w-1, np.int(h/2):] < 50) > 5 and np.sum(img[np.int(w/2):,h-1] < 50) > 5:
	docrop = True
	corners.append(["bottomright",
			np.int(h/2) + np.min(np.where(img[w-1, np.int(h/2):] < 50)),
			np.int(w/2) + np.min(np.where(img[np.int(w/2):,h-1] < 50))])

    if np.sum(img[w-1,0:np.int(h/2)]<50) >5 and np.sum(img[np.int(w/2):,0] <50) > 5:
	docrop = True
	corners.append(["bottomleft",
			np.max(np.where(img[w-1,0:np.int(h/2)] < 50)),
			np.int(w/2) + np.min(np.where(img[np.int(w/2):,0] < 50))])
    
    if len(corners) == 0:
	return img
    else:

	# this method tries to crop the left and righr corners column-wise first
	try:
	    leftbound = max([x[1] for x in corners if "left" in x[0]])
	except ValueError:
	    leftbound = 0
	
	try:
	    rightbound = min([x[1] for x in corners if "right" in x[0]])
	except ValueError:
	    rightbound = h-1
	
	if (leftbound > int(h*0.25) or rightbound < int(h*0.75)) or (leftbound == int(h/2)-1 and  rightbound == int(h/2)):

	    #if the left and right corners can't be cropped column-wise (e.g. there's a solid border along the bottom)

	    if len(corners) == 4:
		img = cv2.medianBlur(img,5)
		circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
					   param1=50,param2=50,minRadius=300)
		if circles is np.nan:
		    return crop(img[int(w/2):,:])
		else:
		    circle = np.mean(np.array(circles[0]),axis=0)
		    x,y,r = circle
		    return crop(img[int(max(y-0.7*r,0)):int(min(y+0.7*r,h)),
					 int(max(x-0.7*r,0)):int(min(x+0.7*r,w))])
	    
	    cornernames = [x[0] for x in corners]
	    
	    if len(corners) == 3:
		if "topright" not in cornernames:
		    for x in corners:
			if x[0]=="topleft": leftb = x[1] 
		    for x in corners:
			if x[0]=="bottomright": lowerb = x[2]
		    return crop(img[:lowerb,leftb:])
		
		elif "bottomright" not in cornernames:
		    for x in corners:
			if x[0]=="bottomleft": leftb = x[1]
		    for x in corners:
			if x[0]=="topright": topb = x[2]
		    return crop(img[topb:,leftb:])
		
		elif "topleft" not in cornernames:
		    for x in corners:
			if x[0]=="topright": rightb = x[1]
		    for x in corners:
			if x[0]=="bottomleft": lowerb = x[2]
		    return crop(img[:lowerb,:rightb])
		
		elif "bottomleft" not in cornernames:
		    for x in corners:
			if x[0]=="bottomright": rightb = x[1]
		    for x in corners:
			if x[0]=="topleft": topb = x[2]
		    return crop(img[topb:,:rightb])
	    
	    elif all(["bottom" in x[0] for x in corners]):
		threshold = min([x[2] for x in corners])
		return crop(img[0:threshold,:])

	    elif all(["top" in x[0] for x in corners]):
		threshold = max([x[2] for x in corners])
		return crop(img[threshold:,:])

	    elif all(["right" in x[0] for x in corners]):
		threshold = min([x[1] for x in corners])
		return crop(img[:,0:threshold])

	    elif all(["left" in x[0] for x in corners]):
		threshold = max([x[1] for x in corners])
		return img[:,threshold:]
        else: return crop(img[:,leftbound:rightbound])

def calc_pixel_to_mm(im):

    # calculates the pixel to millimeter ratio for a clone given an image of
    # a micrometer associated with clone

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cropped = crop(im)
    
    w,h = cropped.shape
    cl1 = clahe.apply(cropped)
    highcontrast = cl1.copy()

    edge_threshold = 175
    sum_edges = w*h
    lines = None

    while (edge_threshold > 0 and not np.any(lines)):
	edges = cv2.Canny(highcontrast,0,edge_threshold,apertureSize = 3)
	sum_edges = np.sum(edges)
	edge_threshold -= 25
	min_line_length = 200

	while (min_line_length > 0) and not np.any(lines) and (sum_edges/(255*w*h) < 0.5):
	    lines = cv2.HoughLines(edges,1,np.pi/180,200,min_line_length)    
	    min_line_length -= 50
    
    if lines is None:
	print "Could not detect ruler"
	return

    measurements = []
    for line in lines[0]:
	rho,theta = line
	a = np.cos(theta)
	b = np.sin(theta)
	x0 = a*rho
	y0 = b*rho
	x1 = int(x0 + 1000*(-b))
	y1 = int(y0 + 1000*(a))
	x2 = int(x0 - 1000*(-b))
	y2 = int(y0 - 1000*(a))
	
	# y=mx+b
	try:
	    m = (y2-y1)/(x2-x1)
	except ZeroDivisionError:
	    continue
	
	b = y2 - m*x2

	x1 = int(0.33*h)
	y1 = int(x1*m + b)
	x2 = int(0.67*h)
	y2 = int(x2*m + b)


	npoints = max(np.abs(y2-y1),np.abs(x2-x1))

	x, y = np.linspace(y1, y2, npoints), np.linspace(x1, x2, npoints)
	# Extract the pixel values along the line
	zi = scipy.ndimage.map_coordinates(highcontrast, np.vstack((x,y)),mode='nearest')
	#mean shift the pixels
	zi = zi-pd.rolling_mean(zi,4)
	df = pd.DataFrame(zi)
	mva = pd.rolling_mean(zi,4)
	mva = mva[~np.isnan(mva)]
	
	#transform to frequency domain
	fourier = np.fft.fft(mva)
	n = fourier.size
	freqs = np.fft.fftfreq(n)
	idx = np.argmax(np.abs(fourier))
	freq = freqs[idx]

	#this is so that really noisy frequencies don't get captured
	try:
	    if np.abs(1/freq) < 50:
		measurements.append(np.abs(1/freq)*40)
	except ZeroDivisionError:
	    continue
    return np.mean(measurements)

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

def clone_to_line(clone, cols, metadata_fields=None, metadata=None):
    
    if metadata_fields is not None:
        cols = metadata_fields + cols

    tmpdata = []
    for mf in cols:
        try:
            if mf == "animal_dorsal_area_mm":
                val = getattr(clone, "animal_dorsal_area")/np.power(metadata["pixel_to_mm"], 2)
            elif mf == "eye_area_mm":
                val = getattr(clone, "eye_area")/np.power(metadata["pixel_to_mm"], 2)
            elif mf == "animal_length_mm":
                val = getattr(clone, "animal_length")/metadata["pixel_to_mm"]
            elif mf == "tail_spine_length_mm":
                val = getattr(clone, "tail_spine_length")/metadata["pixel_to_mm"]
            elif mf == "pedestal_area_mm":
                val = getattr(clone, "pedestal_area")/np.power(metadata["pixel_to_mm"], 2)
            elif mf == "pedestal_max_height_mm":
                val = getattr(clone, "pedestal_max_height")/metadata["pixel_to_mm"]
            elif mf == "deyecenter_pedestalmax_mm":
                val = getattr(clone, "deyecenter_pedestalmax")/metadata["pixel_to_mm"]
            else:
                val = getattr(clone, mf)

        except (AttributeError, TypeError, ValueError):
            val = "nan"

        tmpdata.append(str(val))

    return "\t".join(tmpdata)

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
                f.write( "\t".join(["filebase","i","x","y","qi","q","checkpoint"]) + "\n")
        except IOError:
            print "Can't find desired location for saving data"
    
    try:
        with open(output, "ab+") as f:

            tmpdata = []
            for mf in metadata_fields:
                
                try:
                    if mf == "animal_dorsal_area_mm":
                        val = getattr(clone, "animal_dorsal_area")/np.power(metadata["pixel_to_mm"], 2)
                    elif mf == "eye_area_mm":
                        val = getattr(clone, "eye_area")/np.power(metadata["pixel_to_mm"], 2)
                    elif mf == "animal_length_mm":
                        val = getattr(clone, "animal_length")/metadata["pixel_to_mm"]
                    elif mf == "tail_spine_length_mm":
                        val = getattr(clone, "tail_spine_length")/metadata["pixel_to_mm"]
                    elif mf == "pedestal_area_mm":
                        val = getattr(clone, "pedestal_area")/np.power(metadata["pixel_to_mm"], 2)
                    elif mf == "pedestal_max_height_mm":
                        val = getattr(clone, "pedestal_max_height")/metadata["pixel_to_mm"]
                    elif mf == "deyecenter_pedestalmax_mm":
                        val = getattr(clone, "deyecenter_pedestalmax")/metadata["pixel_to_mm"]
                    else:
                        val = metadata[mf]
                except Exception:
                    val = "nan"
                
                tmpdata.append(str(val))



            for c in cols:

                val = str(getattr(clone, c))

                if val is not None:
                    tmpdata.append(val)
                else:
                    tmpdata.append("nan")
            
            f.write("\t".join(tmpdata) + "\n")
        
        try:
            with open(shape_output, "ab+") as f:
                for i in np.arange(len(clone.dorsal_edge)):

                    if len(np.where((clone.checkpoints==clone.dorsal_edge[i,:]).all(axis=1))[0]) > 0:
                        checkpoint = 1
                    else:
                        checkpoint = 0
                    f.write('\t'.join([clone.filebase, str(i), str(clone.dorsal_edge[i, 0]), str(clone.dorsal_edge[i,1]), str(clone.qi[i]), str(clone.q[i]), str(checkpoint)]) + "\n")
        
        except Exception as e:
            print "Error writing dorsal edge to file: " + str(e)
    except (IOError, AttributeError) as e:
        print "Can't write data for " + clone.filepath + " to file: " + str(e)

def read_shape_long( shape_file ):

    return pd.read_csv( shape_file, dtype={'filebase':str, 'i':float, 'x':float, 'y':float, 'qi':float, 'q':float},sep="\t")

def read_masked_regions_long( masked_regions_file ):

    masked_regions = {}

    df = pd.read_csv( masked_regions_file, dtype={'filebase':str, 'i':int, 'x':float, 'y':float, 'masking_or_unmasking':str}, sep="\t")
    fbs = np.unique(df.filebase)

    for fb in fbs:

        regions = df[df.filebase==fb]
        n_regions = np.max(regions.i)
        masked_regions[fb] = {}

        for reg in xrange(n_regions+1):
            region = regions[regions.i == reg]
            region_type = np.unique(region.masking_or_unmasking)[0]
            x = region.x.values
            y = region.y.values
            masked_regions[fb][reg] = (region_type, np.transpose(np.vstack((x,y))))
    
    return masked_regions

def write_analysis_metadata(clone, params_dict, metadata_output_file):
    
    params_key_list = [] 
    params_val_list = []

    for k in params_dict.keys():
       try:
           params_val_list.append(str(getattr(clone,k)))
           params_key_list.append(k)
       except AttributeError:
           continue
    try:
        if (os.stat(metadata_output_file).st_size == 0):
            raise OSError
    except (OSError):
        print "Starting new analysis metadata output file"

        try:
            with open(metadata_output_file, "wb+") as f:
                f.write( "\t".join(params_key_list) + "\n")
        except IOError:
            print "Can't find desired location for saving analysis metadata"
    
    try:
        with open(metadata_output_file, "ab+") as f:
            f.write("\t".join(params_val_list)+"\n")
    except Exception as e:
        print "Error writing analysis metadata to file: " + str(e)
