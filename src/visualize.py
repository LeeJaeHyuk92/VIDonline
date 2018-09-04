import argparse
import glob
import os
import sys

from helper.videofig import videofig
from matplotlib.pyplot import Rectangle
from scipy.misc import imread


def set_bbox(artist, bbox):
    artist.set_xy((bbox[0], bbox[1]))
    artist.set_width(bbox[2])
    artist.set_height(bbox[3])


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
args = parser.parse_args()
seqs = os.listdir('./sequences')
if 'list.txt' in seqs:
    seqs.remove('list.txt')
for seq in seqs:
    if args.s is not None:
        seq = args.s
    seq_path = os.path.join('./result', seq)

    # file path
    img_files = sorted(glob.glob(os.path.join('./sequences', seq, 'color', '*.jpg')))
    plot_png = os.path.join('./result', seq, 'plot.png')
    gt_files = os.path.join('./sequences', seq, 'groundtruth.txt')
    overlap_txt = os.path.join(seq_path, 'overlap.txt')
    objectness_txt = os.path.join(seq_path, 'objectness.txt')
    result_txt = os.path.join(seq_path, 'result.txt')

    bboxes_gt = [x.strip('\n') for x in open(gt_files, 'r').readlines()]
    overlap_seq = [x.strip('\n') for x in open(overlap_txt, 'r').readlines()]
    objectness_seq = [x.strip('\n') for x in open(objectness_txt, 'r').readlines()]
    bbox_seq = [x.strip('\n') for x in open(result_txt, 'r').readlines()]

    def redraw_fn(idx, axes):
        idx += 1
        img = imread(img_files[idx])
        x_gt, y_gt, w_gt, h_gt = vot_rect(bboxes_gt[idx])
        b_seq = bbox_seq[idx - 1].split(',')
        b_seq = map(float, b_seq)
        overlap = float(overlap_seq[idx - 1])
        objectness = float(objectness_seq[idx - 1])

        if not redraw_fn.initialized:
            # import pdb; pdb.set_trace()
            axes, axes_p = axes
            redraw_fn.im_p = axes_p.imshow(imread(plot_png))
            redraw_fn.im = axes.imshow(img)
            redraw_fn.bb1 = Rectangle((x_gt, y_gt), w_gt, h_gt, fill=False, edgecolor='red', linewidth=5)
            redraw_fn.bb2 = Rectangle((b_seq[0], b_seq[1]), b_seq[2], b_seq[3], fill=False, edgecolor='blue', linewidth=5)
            axes.add_patch(redraw_fn.bb2)
            axes.add_patch(redraw_fn.bb1)
            redraw_fn.text1 = axes.text(0.03, 0.97, '{}: {}'.format(seq, idx+1), fontdict={'size':10,},
                    ha='left', va='top', bbox={'facecolor':'yellow', 'alpha':0.7}, transform=axes.transAxes)
            redraw_fn.text2 = axes.text(0.03, 0.92, 'olap: {:.2f}'.format(overlap), fontdict={'size':10,},
                    ha='left', va='top', bbox={'facecolor':'red', 'alpha':0.7}, transform=axes.transAxes)
            redraw_fn.text3 = axes.text(0.03, 0.87, 'obj: {:.2f}'.format(objectness), fontdict={'size':10,},
                    ha='left', va='top', bbox={'facecolor':'blue', 'alpha':0.7}, transform=axes.transAxes)
            redraw_fn.text1_p = axes_p.text(0.03, 0.97, '{}: {}'.format(seq, idx+1), fontdict={'size':10,},
                    ha='left', va='top', bbox={'facecolor':'yellow', 'alpha':0.7}, transform=axes_p.transAxes)
            redraw_fn.text2_p = axes_p.text(0.03, 0.92, 'olap: {:.2f}'.format(overlap), fontdict={'size':10,},
                    ha='left', va='top', bbox={'facecolor':'red', 'alpha':0.7}, transform=axes_p.transAxes)
            redraw_fn.text3_p = axes_p.text(0.03, 0.87, 'obj: {:.2f}'.format(objectness), fontdict={'size':10,},
                    ha='left', va='top', bbox={'facecolor':'blue', 'alpha':0.7}, transform=axes_p.transAxes)
            redraw_fn.initialized = True
        else:
            # import pdb; pdb.set_trace()
            redraw_fn.im.set_array(img)
            # redraw_fn.im_p.set_array(imread(plot_png))
            set_bbox(redraw_fn.bb1, vot_rect(bboxes_gt[idx]))
            set_bbox(redraw_fn.bb2, b_seq)
            redraw_fn.text1.set_text('{}: {}'.format(seq, idx+1))
            redraw_fn.text2.set_text('olap: {:.2f}'.format(overlap))
            redraw_fn.text3.set_text('obj: {:.2f}'.format(objectness))
            redraw_fn.text1_p.set_text('{}: {}'.format(seq, idx+1))
            redraw_fn.text2_p.set_text('olap: {:.2f}'.format(overlap))
            redraw_fn.text3_p.set_text('obj: {:.2f}'.format(objectness))

    redraw_fn.initialized = False
    videofig(len(img_files) - 1, redraw_fn, play_fps=60,
            grid_specs={'nrows': 1, 'ncols': 2, 'wspace': 0, 'hspace': 0},
            layout_specs=['[0, 0]', '[0, 1]'])
    if args.s is not None:
        break
