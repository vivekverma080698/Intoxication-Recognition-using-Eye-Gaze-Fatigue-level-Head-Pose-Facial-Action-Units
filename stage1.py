import os
import subprocess

def STAGE1():
    OPENFACE_LOCATION = '/home/black/OpenFace-master/build/bin/FeatureExtraction'

    input_folder_video = './video'
    output_folder = './Features'

    files = []
    for f in os.listdir('./video'):
        files.append(input_folder_video+'/'+f)

    c1 = '"{OPENFACE_LOCATION}" {videos} -out_dir '+ output_folder +' -2Dfp -3Dfp -pose -aus -gaze'
    videoPaths = ""
    for f in range(0,len(files)):
        videoPaths+='-f "'+files[f]+'" '

    com1 = c1.format(OPENFACE_LOCATION = OPENFACE_LOCATION , videos= videoPaths)
    subprocess.call(com1, shell=True)
