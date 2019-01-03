import argparse
import glob
import os
import sys
import time

import cv2
# jaehyuk, check network file
import goturn_net_coord
import numpy as np
import tensorflow as tf
from helper.BoundingBox import BoundingBox, calculate_box
from helper.config import POLICY
from helper.image_proc import cropPadImage
from logger.logger import setup_logger
from progressbar import ProgressBar, ETA, Percentage

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
FLOAT_MAX = sys.float_info.max
FLOAT_MIN = sys.float_info.min

class bbox_estimator:
    """tracker class"""
    def __init__(self, show_intermediate_output, logger):
        """TODO: to be defined. """
        self.show_intermediate_output = show_intermediate_output
        self.logger = logger

    def init(self, image_curr, init_bbox):
        """ initializing the first frame in the video """
        left = float(init_bbox[0])
        top = float(init_bbox[1])
        right = float(init_bbox[2])
        bottom = float(init_bbox[3])
        bbox_gt = BoundingBox(left, top, right, bottom)
        self.image_prev = image_curr
        self.bbox_prev_tight = bbox_gt
        self.bbox_curr_prior_tight = bbox_gt
        self.DeltaBox = np.array([0., 0.])
        self.lambdaBox = 0.3
        self.prevBoxeffect = 0
        self.occlusion_flag = 0

        target_pad, _, _,  _ = cropPadImage(self.bbox_prev_tight, self.image_prev)

        # image, BGR(training type)
        target_pad_resize = self.preprocess(target_pad)
        
        # jaehyuk, check hanning windows
        hann_1d = np.expand_dims(np.hanning(227), axis=0)
        hann_2d = np.transpose(hann_1d) * hann_1d
        hann_2d = np.expand_dims(hann_2d, axis=2)
        target_pad_resize = target_pad_resize * hann_2d

        target_pad_expdim = np.expand_dims(target_pad_resize, axis=0)
        self.target_pool5 = sess.run([tracknet.target_pool5], feed_dict={tracknet.target: target_pad_expdim})
        self.target_pool5 = np.resize(self.target_pool5, [1,6,6,256])

    def preprocess(self, image):
        """TODO: Docstring for preprocess.

        :arg1: TODO
        :returns: TODO """
        image_out = image
        if image_out.shape != (POLICY['HEIGHT'], POLICY['WIDTH'], POLICY['channels']):
            image_out = cv2.resize(image_out, (POLICY['WIDTH'], POLICY['HEIGHT']), interpolation=cv2.INTER_CUBIC)

        image_out = np.float32(image_out)
        return image_out

    def track(self, image_curr, tracknet, sess):
        """TODO: Docstring for tracker.
        :returns: TODO

        """
        # target_pad, _, _,  _ = cropPadImage(self.bbox_prev_tight, self.image_prev)
        cur_search_region, search_location, edge_spacing_x, edge_spacing_y = cropPadImage(self.bbox_curr_prior_tight, image_curr)

        # image, BGR(training type)
        cur_search_region_resize = self.preprocess(cur_search_region)
        # target_pad_resize = self.preprocess(target_pad)
        
        # jaehyuk, check hanning windows
        # hann_1d = np.expand_dims(np.hanning(227), axis=0)
        # hann_2d = np.transpose(hann_1d) * hann_1d
        # hann_2d = np.expand_dims(hann_2d, axis=2)
        # target_pad_resize = target_pad_resize * hann_2d

        cur_search_region_expdim = np.expand_dims(cur_search_region_resize, axis=0)
        # target_pad_expdim = np.expand_dims(target_pad_resize, axis=0)

        re_fc4_image, fc4_adj = sess.run([tracknet.re_fc4_image, tracknet.fc4_adj],
                                         feed_dict={tracknet.image: cur_search_region_expdim,
                                                    tracknet.target_pool5: self.target_pool5})
        bbox_estimate, object_bool, objectness = calculate_box(re_fc4_image, fc4_adj)
        # print('objectness_s is: ', objectness)

        ########### original method ############
        # this box is NMS result, TODO, all bbox check

        if not len(bbox_estimate) == 0:
            bbox_estimate = BoundingBox(bbox_estimate[0][0], bbox_estimate[0][1], bbox_estimate[0][2], bbox_estimate[0][3])

            # Inplace correction of bounding box
            bbox_estimate.unscale(cur_search_region)
            bbox_estimate.uncenter(image_curr, search_location, edge_spacing_x, edge_spacing_y)

            # jaehyuk, check first vs adj
            # self.image_prev = image_curr
            # self.bbox_prev_tight = bbox_estimate
            self.bbox_curr_prior_tight = bbox_estimate
        else:
            # self.image_prev = self.image_prev
            # self.bbox_prev_tight = self.bbox_prev_tight
            self.bbox_curr_prior_tight =self.bbox_curr_prior_tight
            bbox_estimate = self.bbox_curr_prior_tight

        ########### original method ############


        ############ trick method ############

        # if object_bool:
        # # if not len(bbox_estimate) == 0:
        #     # current_box_wh = np.array([(bbox_estimate.[0][2] - bbox_estimate.[0][0]), (bbox_estimate.[0][3] - bbox_estimate.[0][1])], dtype=np.float32)
        #     # prev_box_wh = np.array([5., 5.], dtype=np.float32)
        #
        #     bbox_estimate = BoundingBox(bbox_estimate[0][0], bbox_estimate[0][1], bbox_estimate[0][2], bbox_estimate[0][3])
        #
        #     # relative distance from center point [5. 5.]
        #     relative_current_box = np.array([(bbox_estimate.x2 + bbox_estimate.x1) / 2,
        #                             (bbox_estimate.y2 + bbox_estimate.y1) / 2],
        #                            dtype=np.float32)
        #     relative_distance = np.linalg.norm(relative_current_box - np.array([5., 5.]))
        #
        #     # Inplace correction of bounding box
        #     bbox_estimate.unscale(cur_search_region)
        #     bbox_estimate.uncenter(image_curr, search_location, edge_spacing_x, edge_spacing_y)
        #
        #     # image's width height , center point
        #     current_box = np.array([(bbox_estimate.x2 + bbox_estimate.x1) / 2, (bbox_estimate.y2 + bbox_estimate.y1) / 2], dtype=np.float32)
        #     prev_box = np.array([(self.bbox_curr_prior_tight.x2 + self.bbox_curr_prior_tight.x1) / 2, (self.bbox_curr_prior_tight.y2 + self.bbox_curr_prior_tight.y1) / 2],
        #                         dtype=np.float32)
        #
        #     if relative_distance < 2:
        #         self.DeltaBox = self.lambdaBox * (current_box - prev_box) + (1 - self.lambdaBox) * self.DeltaBox
        #
        #
        #         self.image_prev = image_curr
        #         self.bbox_prev_tight = bbox_estimate
        #         self.bbox_curr_prior_tight = bbox_estimate
        #         print(self.DeltaBox)
        #     else:
        #         # under prev img, box block is no update
        #         self.image_prev = self.image_prev
        #         self.bbox_prev_tight = self.bbox_prev_tight
        #         # self.bbox_curr_prior_tight = self.bbox_prev_tight
        #         self.bbox_curr_prior_tight = BoundingBox(self.bbox_curr_prior_tight.x1 + self.DeltaBox[0],
        #                                                  self.bbox_curr_prior_tight.y1 + self.DeltaBox[1],
        #                                                  self.bbox_curr_prior_tight.x2 + self.DeltaBox[0],
        #                                                  self.bbox_curr_prior_tight.y2 + self.DeltaBox[1])
        #         bbox_estimate = self.bbox_curr_prior_tight
        #         print('distance is {:>3}'.format(relative_distance))
        #         print(self.DeltaBox)
        # else:
        #     # under prev img, box block is no update
        #     self.image_prev = self.image_prev
        #     self.bbox_prev_tight = self.bbox_prev_tight
        #     # self.bbox_curr_prior_tight = self.bbox_prev_tight
        #     self.bbox_curr_prior_tight = BoundingBox(self.bbox_curr_prior_tight.x1 + self.DeltaBox[0],
        #                                              self.bbox_curr_prior_tight.y1 + self.DeltaBox[1],
        #                                              self.bbox_curr_prior_tight.x2 + self.DeltaBox[0],
        #                                              self.bbox_curr_prior_tight.y2 + self.DeltaBox[1])
        #     bbox_estimate = self.bbox_curr_prior_tight
        #     print('occlusion is detected')
        #     print(self.DeltaBox)
        #
        # ############ trick method ############

        return bbox_estimate, objectness
        # return bbox_estimate


if __name__ == '__main__':
    BATCH_SIZE = 1

    # for progressbar
    logger = setup_logger(logfile=None)
    # ckpt_dir = "/home/jaehyuk/code/github/vot-toolkit/tracker/examples/python/checkpoints_temp"
    ckpt_dir = "./checkpoints"
    seq_dir = "./sequences"

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", dest='vis', help='plot visualization', action='store_true', default=False)
    parser.add_argument("-c", "--ckpt", dest='ckpt', help='upload checkpount', type=str, default=None, metavar="FILE")
    args = parser.parse_args()

    # import pdb; pdb.set_trace() # debug

    visualize = args.vis
    ckpt = args.ckpt
        
    bbox_estim = bbox_estimator(False, logger)
    # jaehyuk, check network file
    tracknet = goturn_net_coord.TRACKNET(BATCH_SIZE, train=False)
    tracknet.build()

    sess = tf.Session()
    init = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()
    sess.run(init)
    sess.run(init_local)

    
    # if ckpt does not exist, pick max num ckpt in ckpt_dir
    if ckpt:
        saver = tf.train.Saver()
        saver.restore(sess, ckpt)
        logger.info(str(ckpt) + " is restored")
        ckpt_num = str(ckpt).split('-')[-1]
    else:
        # jaehyuk, check checkpoint, descend
        all_ckpt_meta = glob.glob(os.path.join(ckpt_dir, '*.meta'))
        num = []
        for ckpt_meta in all_ckpt_meta:
            num.append(int(ckpt_meta.split('-')[2].split('.')[0]))
        max_num = max(num)
        ckpt_in_dir = os.path.join(ckpt_dir, 'checkpoint.ckpt-' + str(max_num))

        saver = tf.train.Saver()
        saver.restore(sess, ckpt_in_dir)
        logger.info("model is restored using " + str(ckpt_in_dir))
        ckpt_num = str(ckpt_in_dir).split('-')[-1]

    seqs = os.listdir(seq_dir)
    if 'list.txt' in seqs:
        seqs.remove('list.txt')

    widgets = ['Sequence: ', Percentage(), " ", ETA()]
    pbar = ProgressBar(widgets=widgets)
    for seq in pbar(seqs):
        seq_path = os.path.join(seq_dir, seq)
        result_seq = []
        objectness_seq = []
        overlap_seq = []

        gt = os.path.join(seq_path, "groundtruth.txt")
        f = open(gt, 'r')
        bboxes_gt = [x.strip('\n') for x in open(gt, 'r').readlines()]
        
        images_path = sorted(glob.glob(os.path.join(seq_path, "color", "*.jpg")))
        time_total = 0.
        for i, image_path in enumerate(images_path):
            image = cv2.imread(image_path)

            # bboxes_gt[i].split(',') = [_, _, xleft, ytop, _, _, xright, ybot]
            left_x_gt = FLOAT_MAX 
            right_x_gt = FLOAT_MIN
            left_y_gt = FLOAT_MAX
            right_y_gt = FLOAT_MIN
            for j, point in enumerate(bboxes_gt[i].split(',')):
                point = float(point)
                if j % 2 == 0:
                    left_x_gt = min(left_x_gt, point) 
                    right_x_gt = max(right_x_gt, point)
                else:
                    left_y_gt = min(left_y_gt, point)
                    right_y_gt = max(right_y_gt, point)
            gt_bbox = [left_x_gt, left_y_gt, right_x_gt, right_y_gt]
            
            if i == 0:
                bbox_estim.init(image, gt_bbox)
            else:
                start_time = time.time()
                bbox_estimate, objectness = bbox_estim.track(image, tracknet,  sess)
                # print('test: time elapsed: %.4fs.'%(time.time()-start_time))
                time_total += time.time()-start_time

                # vot result format
                left_x = bbox_estimate.x1
                left_y = bbox_estimate.y1
                right_x = bbox_estimate.x2
                right_y = bbox_estimate.y2
                width = bbox_estimate.x2 - bbox_estimate.x1
                height = bbox_estimate.y2 - bbox_estimate.y1
                result = [left_x, left_y, width, height]
                result_seq.append(result)

                # compute overlap
                area_pred = width * height
                area_gt = (right_x_gt - left_x_gt) * (right_y_gt - left_y_gt)
                
                inter_left_x = max(left_x, left_x_gt)
                inter_left_y = max(left_y, left_y_gt)
                inter_right_x = min(right_x, right_x_gt)
                inter_right_y = min(right_y, right_y_gt)
                if (inter_right_x - inter_left_x) > 0 and (inter_right_y - inter_left_y) > 0:
                    inter_area = (inter_right_x - inter_left_x) * (inter_right_y - inter_left_y)
                else:
                    inter_area = 0
                union = area_pred + area_gt - inter_area
                overlap = inter_area / union
                overlap_seq.append(overlap)

                # objectness 
                objectness_seq.append(objectness)

                # TODO, output
                if visualize:
                    pass
                    # cv2.rectangle
                    # cv2.putText

        print('test: time elapsed: %.4fs.'%(time_total / len(images_path)))
        # (overlap, objectness)/frame
        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.plot(range(len(overlap_seq)), overlap_seq, 'bo', label='overlap', markersize=3)
        plt.plot(range(len(objectness_seq)), objectness_seq, 'ro', label='objectness', markersize=3)
        plt.ylim(0, 1)
        plt.grid(True)
        plt.title("{} with {}".format(str(seq), str(ckpt_num)))
        plt.xlabel("Frames")
        plt.legend(loc='best')
        # visualization
        if visualize:
            plt.show()

        # save result
        if not os.path.exists('./result'):
            os.mkdir('./result')
        if not os.path.exists('./result/{}'.format(seq)):
            os.mkdir('./result/{}'.format(seq))

        with open(os.path.join('./result/{}'.format(seq), 'result.txt'), 'w') as f:
            for b in result_seq:
                line = "{},{},{},{}\n".format(b[0], b[1], b[2], b[3])
                f.write(line)

        with open(os.path.join('./result/{}'.format(seq), 'overlap.txt'), 'w') as f:
            for o in overlap_seq:
                line = "{}\n".format(o)
                f.write(line)

        with open(os.path.join('./result/{}'.format(seq), 'objectness.txt'), 'w') as f:
            for obj in objectness_seq:
                line = "{}\n".format(obj[0])
                f.write(line)

        plt.savefig(os.path.join('./result/{}'.format(seq), 'plot.png'))
        plt.close()

