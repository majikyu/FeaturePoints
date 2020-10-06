"""

Select Algorithm by option '-m' or '-method', choose num from 0 to 7
0 -> 'ORB'
1 -> 'AGAST'
2 -> 'FAST'
3 -> 'MSER'
4 -> 'AKAZE'
5 -> 'BRISK'
6 -> 'KAZE'
7 -> 'BLOB'

"""
import cv2
import os
from pathlib import Path
import time
import argparse

class CountFP():
    def __init__(self):
        self.finder = list()
        self.finder.append(cv2.ORB_create())
        self.finder.append(cv2.AgastFeatureDetector_create())
        self.finder.append(cv2.FastFeatureDetector_create())
        self.finder.append(cv2.MSER_create())
        self.finder.append(cv2.AKAZE_create())
        self.finder.append(cv2.BRISK_create())
        self.finder.append(cv2.KAZE_create())
        self.finder.append(cv2.SimpleBlobDetector_create())

    def print_fpnum(self, f, img, name):
        kp = []
        for detecter in self.finder:
            kp.append(str(len(detecter.detect(img))))

        f.write("{},{}\n".format(name, ",".join(kp)))

def AlgorithmSelecter(algo):
    if algo == 0: 
        return cv2.ORB_create()
    elif algo == 1:
        return cv2.AgastFeatureDetector_create()
    elif algo == 2:
        return cv2.FastFeatureDetector_create()
    elif algo == 3:
        return cv2.MSER_create()
    elif algo == 4:
        return cv2.AKAZE_create()
    elif algo == 5:
        return cv2.BRISK_create()
    elif algo == 6:
        return cv2.KAZE_create()
    elif algo == 7:
        return cv2.SimpleBlobDetector_create()
    else:
        print("[ ERROR ] AlgorithmSelecter : Receive unexpected input")
        exit()

########## Main ###########

parser = argparse.ArgumentParser()
parser.add_argument("-i","--input_dir", help="Directry which your images saved", nargs="?", default="./images/")
parser.add_argument("-img","--save_image", help="Save images with featurepoints", nargs="?", const="./output/", default="./output/")
parser.add_argument("-log","--save_log",  help="Save logs", nargs="?", const="result.csv", default=None)
parser.add_argument("-m","--method",  help="Choice use method to find fp", nargs="?",  default=0, choices=range(0, 8), type=int)
parser.add_argument("-c","--count", help="Change count mode", action="store_true")
args = parser.parse_args()

image_dir = args.input_dir
outfilename = args.save_log
outimage_dir = args.save_image
algonum = args.method
countmode = args.count
algorithm = [
    'ORB',
    'AGAST',
    'FAST',
    'MSER',
    'AKAZE',
    'BRISK',
    'KAZE',
    'BLOB',
]

filenames = [str(filename) for filename in Path(image_dir).glob('*') if filename.suffix in ['.bmp', '.jpg', '.png']] #陷茨ｽ･陷牙ｸｷ蛻､陷剃ｸ�?蜉ｱ?�ｽｮ郢昜ｻ｣縺帷ｹｧ蟶�?讎�?蜉ｱ竊楢ｭｬ�ｽｼ驍�?


if len(filenames) == 0:
    print("[ ERROR ] Can't find any images in {}".format(image_dir))
    exit()
else:
    filenum = len(filenames)
    print("\n[ input {} pictures ]\n".format(filenum))

if countmode:
    Counter = CountFP()
    if not outfilename:
        outfilename = "result.csv"

    f = open(outfilename, mode='w')
    f.write("Name,{}\n".format(",".join(algorithm)))
    for i, imagename in enumerate(filenames, 1):
        img = cv2.imread(imagename)
        basename = os.path.basename(imagename)
        Counter.print_fpnum(f, img, basename)
        print("Progress: [{}/{}] Counted FeaturePoints".format(i, len(filenames)))

    exit()

if args.save_log:
    f = open(outfilename, mode='w')

os.makedirs(outimage_dir, exist_ok="True") 

########## detect and output start ##########

start_time = time.perf_counter() # timer
finder = AlgorithmSelecter(algonum)

for i, imagename in enumerate(filenames, 1):

    img = cv2.imread(imagename)
    kp = finder.detect(img)

    basename = os.path.basename(imagename)
    if args.save_log:
        f.write(basename + ',' + str(len(kp)) + "\n") 
    
    img2 = cv2.drawKeypoints(img,kp,None,color=(0,255,0))
    outpath = outimage_dir + basename
    outpath = os.path.join(outimage_dir, basename)
    cv2.imwrite(outpath, img2)

    print("Progress: [{}/{}] output {}".format(i, filenum, outpath))


print("\n[ Execution time : {:.7f} ]".format(time.perf_counter() - start_time))

if args.save_log:
f.close()