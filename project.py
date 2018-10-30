import csv
import numpy
import pandas
import os
import subprocess

OPENFACE_LOCATION = '/home/black/OpenFace-master/build/bin/FeatureExtraction'

input_folder = '/home/black/pipeline_4_July/10secin'
output_folder = '/home/black/pipeline_4_July/10secout/Drunk'
files = []
for f in os.listdir('10secin/'):
 	files.append(input_folder+'/'+f)

#print(files)
c1 = '"{OPENFACE_LOCATION}" {videos} -out_dir '+ output_folder +' -2Dfp -3Dfp -pose -aus -gaze';
videoPaths = "";
for f in range(0,len(files)):
    videoPaths+='-f "'+files[f]+'" ';

com1 = c1.format(OPENFACE_LOCATION = OPENFACE_LOCATION , videos= videoPaths);
#print(com1)
subprocess.call(com1, shell=True)
