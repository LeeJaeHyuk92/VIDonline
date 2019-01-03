import argparse
import glob
import os
import sys

from helper.videofig import videofig
from matplotlib.pyplot import Rectangle
from scipy.misc import imread


def set_bbox(artist, bbox):
    if len(bbox) == 4:
        artist.set_xy((bbox[0], bbox[1]))
        artist.set_width(bbox[2])
        artist.set_height(bbox[3])
    else:
        artist.set_xy((0., 0.))
        artist.set_width(0.)
        artist.set_height(0.)


def vot_rect(polygon):
    FLOAT_MAX = sys.float_info.max
    FLOAT_MIN = sys.float_info.min

    left_x = FLOAT_MAX
    right_x = FLOAT_MIN
    left_y = FLOAT_MAX
    right_y = FLOAT_MIN
    for j, point in enumerate(polygon.split(',')):
        point = float(point)
        if j % 2 == 0:
            left_x = min(left_x, point)
            right_x = max(right_x, point)
        else:
            left_y = min(left_y, point)
            right_y = max(right_y, point)
    width = right_x - left_x
    height = right_y - left_y
    bbox = [left_x, left_y, width, height]
    return bbox


parser = argparse.ArgumentParser()
parser.add_argument("-s", help="sequence ex) ball", type=str, metavar="SEQUENCE")
parser.add_argument("-r", help="red, result filename ex) GOTURN", type=str, metavar="FOLDER")
parser.add_argument("-g", help="green, result filename ex) GOTURN", type=str, metavar="FOLDER")
parser.add_argument("-b", help="blue, result filename ex) GOTURN", type=str, metavar="FOLDER")
parser.add_argument("-y", help="yellow, result filename ex) GOTURN", type=str, metavar="FOLDER")
args = parser.parse_args()

# import pdb; pdb.set_trace()

results_vot = []
for r in [args.r, args.g, args.b, args.y]:
    if r is not None:
        results_vot.append(r)

seqs = os.listdir('./sequences')
if 'list.txt' in seqs:
    seqs.remove('list.txt')
for seq in seqs:
    if args.s is not None:
        seq = args.s

    # file path
    img_files = sorted(glob.glob(os.path.join('./sequences', seq, 'color', '*.jpg')))
    bbox_seq_dict = {}
    for result in results_vot:
        seq_path = os.path.join('./results_vot',result, 'baseline', seq)
        result_txt = os.path.join(seq_path, "{}_001.txt".format(seq))

        bbox_seq = [x.strip('\n') for x in open(result_txt, 'r').readlines()]
        for idx, bbox in enumerate(bbox_seq):
            bbox_seq[idx] = map(float, bbox.split(',')) 
        bbox_seq_dict.update({result: bbox_seq})

    def redraw_fn(idx, axes):
        idx += 1
        img = imread(img_files[idx])
        if not redraw_fn.initialized:
            redraw_fn.im = axes.imshow(img)
            
            if args.r is not None:
                x, y, w, h = bbox_seq_dict[args.r][idx]
                redraw_fn.bb1 = Rectangle((x, y), w, h, fill=False, edgecolor='red', alpha=0.7, linewidth=3)
                axes.add_patch(redraw_fn.bb1)
            if args.g is not None:
                x, y, w, h = bbox_seq_dict[args.g][idx]
                redraw_fn.bb2 = Rectangle((x, y), w, h, fill=False, edgecolor='green', alpha=0.7, linewidth=3)
                axes.add_patch(redraw_fn.bb2)
            if args.b is not None:
                x, y, w, h = bbox_seq_dict[args.b][idx]
                redraw_fn.bb3 = Rectangle((x, y), w, h, fill=False, edgecolor='blue', alpha=0.7, linewidth=3)
                axes.add_patch(redraw_fn.bb3)
            if args.y is not None:
                x, y, w, h = bbox_seq_dict[args.y][idx]
                redraw_fn.bb4 = Rectangle((x, y), w, h, fill=False, edgecolor='yellow', alpha=0.7, linewidth=3)
                axes.add_patch(redraw_fn.bb4)

            redraw_fn.text1 = axes.text(0.03, 0.97, '{}: {}'.format(seq, idx+1), fontdict={'size':10,},
                    ha='left', va='top', bbox={'facecolor':'yellow', 'alpha':0.7}, transform=axes.transAxes)
            redraw_fn.initialized = True
        else:
            redraw_fn.im.set_array(img)
            if args.r is not None:
                set_bbox(redraw_fn.bb1, bbox_seq_dict[args.r][idx])
            if args.g is not None:
                set_bbox(redraw_fn.bb2, bbox_seq_dict[args.g][idx])
            if args.b is not None:
                set_bbox(redraw_fn.bb3, bbox_seq_dict[args.b][idx])
            if args.y is not None:
                set_bbox(redraw_fn.bb4, bbox_seq_dict[args.y][idx])
            redraw_fn.text1.set_text('{}: {}'.format(seq, idx+1))

    redraw_fn.initialized = False
    videofig(len(img_files) - 1, redraw_fn, play_fps=60)
    if args.s is not None:
        break
