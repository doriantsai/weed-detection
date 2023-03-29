#! /usr/bin/env python3

"""
Evaluate model - work in progress, separating model evaluation from model training in WeedModel.py (removed as of March 28th 2023)
# TODO create ModelEvaluator class
! this is depracated code directly cut out of WeedModel.py, this code will not work
- Should import Detector.py to load model
- show model predictions
- determine outcomes, given labelled data
- create PR curves/etc - currently only set for single-class detector
- ModelEvaluator is made redundant since UNE's code should handle all of the model evaluation
"""


def show(self,
             image,
             sample=None,
             predictions=None,
             outcomes=None,
             sample_color=(0, 0, 255),  # RGB
             predictions_color=(255, 0, 0),
             iou_color=(255, 255, 255),
             transpose_image_channels=True,
             transpose_color_channels=False,
             resize_image=False,
             resize_height=(256)):
        """ show image, sample/groundtruth, model predictions, outcomes
        (TP/FP/etc)

        image - input as a tensor
        sample - groundtruth information for given image
        predictions - model inference results
        outcomes - a string defining true/false positive/negative outcome for detection
        sample_color - triplet RGB values for sample markup on input image
        predictions_color - same as above for prediction markup
        iou_color - same as above for iou with nearest bounding box
        transpose_image_channels - boolean to transpose image channels from
        tensor format [c, h, w] to numpy format [h, w, c], where c = channels, h
        = height, w = width
        transpose_color_channels - switch from RGB images (standard image format/order)
        to BGR format (OpenCV)
        resize_image - boolean, if true, resize image output to resize_height to save on image size
        resize_height - int, image save height, aspect ratio is preserved
        """

        # TODO check input

        # set plotting parameters
        gt_box_thick = 12   # groundtruth bounding box
        dt_box_thick = 6    # detection bounding box
        out_box_thick = 3   # outcome bounding box/overlay
        font_scale = 2  # font scale should be function of image size
        font_thick = 2

        if transpose_color_channels:
            # image tensor comes in as [color channels, length, width] format
            print('swap color channels in tensor format')
            image = image[(2, 0, 1), :, :]

        # move to cpu and convert from tensor to numpy array since opencv
        # requires numpy arrays
        image_out = image.cpu().numpy()

        if transpose_image_channels:
            # if we were working with BGR as opposed to RGB
            image_out = np.transpose(image_out, (1, 2, 0))

        # normalize image from 0,1 to 0,255
        image_out = cv.normalize(
            image_out, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

        # ----------------------------------- #
        # first plot groundtruth boxes
        if sample is not None:
            # NOTE we assume sample is also a tensor

            # plot groundtruth bounding box
            boxes_gt = sample['boxes']
            if len(boxes_gt) > 0:
                n_gt, _ = boxes_gt.size()
                for i in range(n_gt):
                    # TODO just specify int8 or imt16?
                    bb = np.array(boxes_gt[i, :].cpu(), dtype=np.float32)
                    # overwrite the original image with groundtruth boxes
                    image_out = cv.rectangle(image_out,
                                             (int(bb[0]), int(bb[1])),
                                             (int(bb[2]), int(bb[3])),
                                             color=sample_color,
                                             thickness=gt_box_thick)

        # ----------------------------------- #
        # second, plot predictions
        if predictions is not None:
            boxes_pd = predictions['boxes']
            scores = predictions['scores']

            if len(boxes_pd) > 0:
                for i in range(len(boxes_pd)):
                    # TODO just specify int8 or imt16?
                    bb = np.array(boxes_pd[i], dtype=np.float32)
                    image_out = cv.rectangle(image_out,
                                             (int(bb[0]), int(bb[1])),
                                             (int(bb[2]), int(bb[3])),
                                             color=predictions_color,
                                             thickness=dt_box_thick)

                    # add text to top left corner of bbox
                    # no decimals, just x100 for percent
                    sc = format(scores[i] * 100.0, '.0f')
                    cv.putText(image_out,
                               '{}: {}'.format(i, sc),
                               # buffer numbers should be function of font scale
                               (int(bb[0] + 10), int(bb[1] + 30)),
                               fontFace=cv.FONT_HERSHEY_COMPLEX,
                               fontScale=font_scale,
                               color=predictions_color,
                               thickness=font_thick)

        # ----------------------------------- #
        # third, add iou info (within the predicitons if statement)
            if outcomes is not None:
                iou = outcomes['dt_iou']

                # iou is a list or array with iou values for each boxes_pd
                if len(iou) > 0 and len(boxes_pd) > 0:
                    for i in range(len(iou)):
                        bb = np.array(boxes_pd[i], dtype=np.float32)
                        # print in top/left corner of bbox underneath bbox # and
                        # score
                        iou_str = format(iou[i], '.2f')  # max 2 decimal places
                        cv.putText(image_out,
                                   'iou: {}'.format(iou_str),
                                   (int(bb[0] + 10), int(bb[1] + 60)),
                                   fontFace=cv.FONT_HERSHEY_COMPLEX,
                                   fontScale=font_scale,
                                   color=iou_color,
                                   thickness=font_thick)

        # ----------------------------------- #
        # fourth, add outcomes
            if (outcomes is not None) and (sample is not None):
                # for each prediction, if there is a sample, then there is a
                # known outcome being an array from 1-4:
                outcome_list = ['TP', 'FP', 'FN', 'TN']
                # choose color scheme default: blue is groundtruth default: red
                # is detection -> red is false negative green is true positive
                # yellow is false positive
                outcome_color = [(0, 255, 0),   # TP - green
                                 (255, 255, 0),  # FP - yellow
                                 (255, 0, 0),   # FN - red
                                 (0, 0, 0)]     # TN - black
                # structure of the outcomes dictionary outcomes = {'dt_outcome':
                # dt_outcome, # detections, integer index for tp/fp/fn
                # 'gt_outcome': gt_outcome, # groundtruths, integer index for fn
                # 'dt_match': dt_match, # detections, boolean matched or not
                # 'gt_match': gt_match, # gt, boolean matched or not 'fn_gt':
                # fn_gt, # boolean for gt false negatives 'fn_dt': fn_dt, #
                # boolean for dt false negatives 'tp': tp, # true positives for
                # detections 'fp': fp, # false positives for detections
                # 'dt_iou': dt_iou} # intesection over union scores for
                # detections
                dt_outcome = outcomes['dt_outcome']
                if len(dt_outcome) > 0 and len(boxes_pd) > 0:
                    for i in range(len(boxes_pd)):
                        # replot detection boxes based on outcome
                        bb = np.array(boxes_pd[i], dtype=np.float32)
                        image_out = cv.rectangle(image_out,
                                                 (int(bb[0]), int(bb[1])),
                                                 (int(bb[2]), int(bb[3])),
                                                 color=outcome_color[dt_outcome[i]],
                                                 thickness=out_box_thick)
                        # add text top/left corner including outcome type prints
                        # over existing text, so needs to be the same starting
                        # string
                        # no decimals, just x100 for percent
                        sc = format(scores[i] * 100.0, '.0f')
                        ot = format(outcome_list[dt_outcome[i]])
                        cv.putText(image_out,
                                   '{}: {}/{}'.format(i, sc, ot),
                                   (int(bb[0] + 10), int(bb[1] + 30)),
                                   fontFace=cv.FONT_HERSHEY_COMPLEX,
                                   fontScale=font_scale,
                                   color=outcome_color[dt_outcome[i]],
                                   thickness=font_thick)

                # handle false negative cases (ie, groundtruth bboxes)
                boxes_gt = sample['boxes']
                fn_gt = outcomes['fn_gt']
                if len(fn_gt) > 0 and len(boxes_gt) > 0:
                    for j in range(len(boxes_gt)):
                        # gt boxes already plotted, so only replot them if false
                        # negatives
                        if fn_gt[j]:  # if True
                            bb = np.array(
                                boxes_gt[j, :].cpu(), dtype=np.float32)
                            image_out = cv.rectangle(image_out,
                                                     (int(bb[0]), int(bb[1])),
                                                     (int(bb[2]), int(bb[3])),
                                                     color=outcome_color[2],
                                                     thickness=out_box_thick)
                            cv.putText(image_out,
                                       '{}: {}'.format(j, outcome_list[2]),
                                       (int(bb[0] + 10), int(bb[1]) + 30),
                                       fontFace=cv.FONT_HERSHEY_COMPLEX,
                                       fontScale=font_scale,
                                       color=outcome_color[2],  # index for FN
                                       thickness=font_thick)

        # resize image to save space
        if resize_image:
            aspect_ratio = self._image_width / self._image_height
            resize_width = int(resize_height * aspect_ratio)
            resize_height = int(resize_height)
            image_out = cv.resize(image_out,
                                  (resize_width, resize_height),
                                  interpolation=cv.INTER_CUBIC)
        return image_out


def compute_iou_bbox(self, boxA, boxB):
        """
        compute intersection over union for bounding boxes box = [xmin, ymin,
        xmax, ymax], tensors
        """
        # determine the coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # compute area of boxA and boxB
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # compute iou
        iou = interArea / (float(boxAArea + boxBArea - interArea))

        return iou

    def compute_match_cost(self, score, iou, weights=None):
        """ compute match cost metric """
        # compute match cost based on score and iou this is the arithmetic mean,
        # consider the geometric mean
        # https://en.wikipedia.org/wiki/Geometric_mean which automatically
        # punishes the 0 IOU case because of the multiplication small numbers
        # penalized harder
        if weights is None:
            weights = np.array([0.5, 0.5])
        # cost = weights[0] * score + weights[1] * iou # classic weighting
        cost = np.sqrt(score * iou)
        return cost

    def compute_outcome_image(self,
                              pred,
                              sample,
                              DECISION_IOU_THRESH,
                              DECISION_CONF_THRESH):
        """ compute outcome (tn/fp/fn/tp) for single image """
        # output: outcome, tn, fp, fn, tp for the image
        dt_bbox = pred['boxes']
        dt_scores = pred['scores']
        gt_bbox = sample['boxes']

        ndt = len(dt_bbox)
        ngt = len(gt_bbox)

        # compute ious for each dt_bbox, compute iou and record confidence score
        gt_iou_all = np.zeros((ndt, ngt))

        # for each detection bounding box, find the best-matching groundtruth
        # bounding box gt/dt_match is False - not matched, True - matched
        dt_match = np.zeros((ndt,), dtype=bool)
        gt_match = np.zeros((ngt,), dtype=bool)

        # gt/dt_match_id is the index of the match vector eg. gt = [1 0 2 -1]
        # specifies that gt_bbox[0] is matched to dt_bbox[1], gt_bbox[1] is
        # matched to dt_bbox[0], gt_bbox[2] is matched to dt_bbox[2], gt_bbox[3]
        # is not matched to any dt_bbox
        dt_match_id = -np.ones((ndt,), dtype=int)
        gt_match_id = -np.ones((ngt,), dtype=int)

        # dt_assignment_idx = -np.ones((len(dt_bbox),), dtype=int)
        dt_iou = -np.ones((ndt,))

        # find the max overlapping detection box with the gt box
        for i in range(ndt):
            gt_iou = []
            if ngt > 0:
                for j in range(ngt):
                    iou = self.compute_iou_bbox(dt_bbox[i], gt_bbox[j])
                    gt_iou.append(iou)
                gt_iou_all[i, :] = np.array(gt_iou)

                # NOTE finding the max may be insufficient dt_score also
                # important consideration find max iou from gt_iou list
                idx_max = gt_iou.index(max(gt_iou))
                iou_max = gt_iou[idx_max]
                dt_iou[i] = iou_max
            else:
                dt_iou[i] = 0

        # calculate cost of each match/assignment
        cost = np.zeros((ndt, ngt))
        for i in range(ndt):
            if ngt > 0:
                for j in range(ngt):
                    cost[i, j] = self.compute_match_cost(
                        dt_scores[i], gt_iou_all[i, j])
            # else:
                # import code code.interact(local=dict(globals(), **locals()))
                # cost[i,j] = 0

        # choose assignment based on max of matrix. If equal cost, we choose the
        # first match higher cost is good - more confidence, more overlap

        # if there are any gt boxes, then we need to choose an assignment. if no
        # gt, then... dt_iou is all zero, as there is nothing to iou with!
        if ngt > 0:
            for j in range(ngt):
                dt_cost = cost[:, j]
                i_dt_cost_srt = np.argsort(dt_cost)[::-1]

                for i in i_dt_cost_srt:
                    if (not dt_match[i]) and \
                        (dt_scores[i] >= DECISION_CONF_THRESH) and \
                            (gt_iou_all[i, j] >= DECISION_IOU_THRESH):
                        # if the detection box in question is not already
                        # matched
                        dt_match[i] = True
                        gt_match[j] = True
                        dt_match_id[i] = j
                        gt_match_id[j] = i
                        dt_iou[i] = gt_iou_all[i, j]
                        break
                        # stop iterating once the highest-scoring detection
                        # satisfies the criteria\

        tp = np.zeros((ndt,), dtype=bool)
        fp = np.zeros((ndt,), dtype=bool)
        # fn = np.zeros((ngt,), dtype=bool) now determine if TP, FP, FN TP: if
        # our dt_bbox is sufficiently within a gt_bbox with a high-enough
        # confidence score we found the right thing FP: if our dt_bbox is a high
        # confidence score, but is not sufficiently within a gt_bbox (iou is
        # low) we found the wrong thing FN: if our dd_bbox is within a gt_bbox,
        # but has low confidence score we didn't find the right thing TN:
        # basically everything else in the scene, not wholely relevant for
        # PR-curves

        # From the slides: TP: we correctly found the weed FP: we think we found
        # the weed, but we did not FN: we missed a weed TN: we correctly ignored
        # everything else we were not interested in (doesn't really show up for
        # binary detectors)
        dt_iou = np.array(dt_iou)
        dt_scores = np.array(dt_scores)

        # TP: if dt_match True, also, scores above certain threshold
        tp = np.logical_and(dt_match, dt_scores >= DECISION_CONF_THRESH)
        fp = np.logical_or(np.logical_not(dt_match),
                           np.logical_and(dt_match, dt_scores < DECISION_CONF_THRESH))
        fn_gt = np.logical_not(gt_match)
        fn_dt = np.logical_and(dt_match, dt_scores < DECISION_CONF_THRESH)

        # define outcome as an array, each element corresponding to tp/fp/fn for
        # 0/1/2, respectively
        dt_outcome = -np.ones((ndt,), dtype=np.int16)
        for i in range(ndt):
            if tp[i]:
                dt_outcome[i] = 0
            elif fp[i]:
                dt_outcome[i] = 1
            elif fn_dt[i]:
                dt_outcome[i] = 2

        gt_outcome = -np.ones((ngt,), dtype=np.int16)
        if ngt > 0:
            for j in range(ngt):
                if fn_gt[j]:
                    gt_outcome = 1

        # package outcomes
        outcomes = {'dt_outcome': dt_outcome,  # detections, integer index for tp/fp/fn
                    'gt_outcome': gt_outcome,  # groundtruths, integer index for fn
                    'dt_match': dt_match,  # detections, boolean matched or not
                    'gt_match': gt_match,  # gt, boolean matched or not
                    'fn_gt': fn_gt,  # boolean for gt false negatives
                    'fn_dt': fn_dt,  # boolean for dt false negatives
                    'tp': tp,  # true positives for detections
                    'fp': fp,  # false positives for detections
                    'dt_iou': dt_iou,  # intesection over union scores for detections
                    'dt_scores': dt_scores,
                    'gt_iou_all': gt_iou_all,
                    'cost': cost}
        return outcomes

    def compute_pr_dataset(self,
                           dataset,
                           predictions,
                           DECISION_CONF_THRESH,
                           DECISION_IOU_THRESH,
                           imsave=False,
                           save_folder=None):
        """ compute single pr pair for entire dataset, given the predictions """
        # note for pr_curve, see get_prcurve

        self._model.eval()
        self._model.to(self._device)

        # annotations from the original dataset object
        dataset_annotations = dataset.annotations

        tp_sum = 0
        fp_sum = 0
        # tn_sum = 0
        fn_sum = 0
        gt_sum = 0
        dt_sum = 0
        idx = 0
        dataset_outcomes = []
        for image, sample in dataset:

            image_id = sample['image_id'].item()

            img_name = dataset_annotations[image_id]['filename'][:-4]
            # else: img_name = 'st' + str(image_id).zfill(3)

            # get predictions
            pred = predictions[idx]
            pred = self.threshold_predictions(
                pred, DECISION_CONF_THRESH, annotation_type='box')

            outcomes = self.compute_outcome_image(pred,
                                                  sample,
                                                  DECISION_IOU_THRESH,
                                                  DECISION_CONF_THRESH)

            # save for given image:
            if imsave:
                # resize image for space-savings!
                image_out = self.show(image,
                                      sample=sample,
                                      predictions=pred,
                                      outcomes=outcomes,
                                      resize_image=True,
                                      resize_height=720)
                imgw = cv.cvtColor(image_out, cv.COLOR_RGB2BGR)

                if save_folder is None:
                    save_folder = os.path.join('output',
                                               self._model_folder,
                                               'outcomes')

                conf_str = format(DECISION_CONF_THRESH, '.2f')
                save_subfolder = 'conf_thresh_' + conf_str
                save_path = os.path.join(save_folder, save_subfolder)
                os.makedirs(save_path, exist_ok=True)

                save_img_name = os.path.join(
                    save_path, img_name + '_outcome.png')
                cv.imwrite(save_img_name, imgw)

            tp = outcomes['tp']
            fp = outcomes['fp']
            fn_gt = outcomes['fn_gt']
            fn_dt = outcomes['fn_dt']
            fn = np.concatenate((fn_gt, fn_dt), axis=0)
            gt = len(outcomes['gt_match'])
            dt = len(outcomes['dt_match'])

            tp_sum += np.sum(tp)
            fp_sum += np.sum(fp)
            fn_sum += np.sum(fn)
            gt_sum += np.sum(gt)
            dt_sum += np.sum(dt)
            # end per image outcome calculations

            dataset_outcomes.append(outcomes)
            idx += 1

        print('tp sum: ', tp_sum)
        print('fp sum: ', fp_sum)
        print('fn sum: ', fn_sum)
        print('dt sum: ', dt_sum)
        print('tp + fp = ', tp_sum + fp_sum)
        print('gt sum: ', gt_sum)
        print('tp + fn = ', tp_sum + fn_sum)

        prec = tp_sum / (tp_sum + fp_sum+1e-12)
        rec = tp_sum / (tp_sum + fn_sum+1e-12)

        print('precision = {}'.format(prec))
        print('recall = {}'.format(rec))

        f1score = self.compute_f1score(prec, rec)

        print('f1 score = {}'.format(f1score))
        # 1 is good, 0 is bad

        return dataset_outcomes, prec, rec, f1score

    def compute_f1score(self, p, r):
        """ compute f1 score """
        return 2 * (p * r) / (p + r + 1e-12)

    def extend_pr(self,
                  prec,
                  rec,
                  conf,
                  n_points=101,
                  PLOT=False,
                  save_name='outcomes'):
        """ extend precision and recall vectors from 0-1 """
        # typically, pr curve will only span a relatively small range, eg from
        # 0.5 to 0.9 for most pr curves, we want to span the range of recall
        # from 0 to 1 therefore, we "smooth" out precision-recall curve by
        # taking max of precision points for when r is very small (takes the top
        # of the stairs ) take max:
        diff_thresh = 0.001
        dif = np.diff(rec)
        prec_new = [prec[0]]
        for i, d in enumerate(dif):
            if d < diff_thresh:
                prec_new.append(prec_new[-1])
            else:
                prec_new.append(prec[i+1])
        prec_new = np.array(prec_new)
        # print(prec_new)

        if PLOT:
            fig, ax = plt.subplots()
            ax.plot(rec, prec, marker='o',
                    linestyle='dashed', label='original')
            ax.plot(rec, prec_new, marker='x', color='red',
                    linestyle='solid', label='max-binned')
            plt.xlabel('recall')
            plt.ylabel('precision')
            plt.title('prec-rec, max-binned')
            ax.legend()

            os.makedirs(os.path.join(
                'output', self._model_folder), exist_ok=True)
            save_plot_name = os.path.join(
                'output', self._model_folder, save_name + '_test_pr_smooth.png')
            plt.savefig(save_plot_name)

        # now, expand prec/rec values to extent of the whole precision/recall
        # space:
        rec_x = np.linspace(0, 1, num=n_points, endpoint=True)

        # create prec_x by concatenating vectors first
        prec_temp = []
        rec_temp = []
        conf_temp = []
        for r in rec_x:
            if r < rec[0]:
                prec_temp.append(prec_new[0])
                rec_temp.append(r)
                conf_temp.append(conf[0])

        for i, r in enumerate(rec):
            prec_temp.append(prec_new[i])
            rec_temp.append(r)
            conf_temp.append(conf[i])

        for r in rec_x:
            if r >= rec[-1]:
                prec_temp.append(0)
                rec_temp.append(r)
                conf_temp.append(conf[-1])

        prec_temp = np.array(prec_temp)
        rec_temp = np.array(rec_temp)
        conf_temp = np.array(conf_temp)

        # now interpolate: prec_x = np.interp(rec_x, rec_temp, prec_temp)
        prec_interpolator = interp1d(rec_temp, prec_temp, kind='linear')
        prec_x = prec_interpolator(rec_x)

        conf_interpolator = interp1d(rec_temp, conf_temp, kind='linear')
        conf_x = conf_interpolator(rec_x)

        if PLOT:
            fig, ax = plt.subplots()
            ax.plot(rec_temp, prec_temp, color='blue',
                    linestyle='dashed', label='combined')
            ax.plot(rec, prec_new, marker='x', color='red',
                    linestyle='solid', label='max-binned')
            ax.plot(rec_x, prec_x, color='green',
                    linestyle='dotted', label='interp')
            plt.xlabel('recall')
            plt.ylabel('precision')
            plt.title('prec-rec, interpolated')
            ax.legend()

            os.makedirs(os.path.join(
                'output', self._model_folder), exist_ok=True)
            save_plot_name = os.path.join(
                'output', self._model_folder, save_name + '_test_pr_interp.png')
            plt.savefig(save_plot_name)
            # plt.show()

        # make the "official" plot:
        p_out = prec_x
        r_out = rec_x
        c_out = conf_x
        return p_out, r_out, c_out

    def compute_ap(self, p, r):
        """ compute ap score """
        n = len(r) - 1
        ap = np.sum((r[1:n] - r[0:n-1]) * p[1:n])
        return ap

    def get_confidence_from_pr(self, p, r, c, f1, pg=None, rg=None):
        """ interpolate confidence threshold from p, r and goal values. If no
        goal values, then provide the "best" conf, corresponding to the pr-pair
        closest to the ideal (1,1)
        """
        # TODO check p, r are valid if Pg is not none, check Pg is valid, if
        # not, set it at appropriate value if Rg is not none, then check if Rg
        # is valid, if not, set appropriate value
        if pg is not None and rg is not None:
            # print('TODO: check if pg, rg are valid') find the nearest pg-pair,
            # find the nearest index to said pair
            prg = np.array([pg, rg])
            pr = np.array([p, r])
            dist = np.sqrt(np.sum((pr - prg)**2, axis=1))
            # take min dist
            idistmin = np.argmin(dist, axis=1)
            pact = p[idistmin]
            ract = r[idistmin]

            # now interpolate for cg
            f = interp1d(r, c)
            cg = f(ract)

        elif pg is None and rg is not None:
            # TODO search for best pg given rg pg = 1 we use rg to interpolate
            # for cg:
            f = interp1d(r, c, bounds_error=False, fill_value=(c[0], c[-1]))
            cg = f(rg)

        elif rg is None and pg is not None:
            # we use pg to interpolate for cg print('using p to interp c')
            f = interp1d(p, c, bounds_error=False, fill_value=(c[-1], c[0]))
            cg = f(pg)

        else:
            # find p,r closest to (1,1) being max of f1 score
            if1max = np.argmax(f1)
            rmax = r[if1max]
            f = interp1d(r, c)
            cg = f(rmax)

        # we use r to interpolate with c
        return cg

    def get_prcurve(self,
                    dataset,
                    confidence_thresh,
                    nms_iou_thresh,
                    decision_iou_thresh,
                    save_folder,
                    imshow=False,
                    imsave=False,
                    PLOT=False,
                    annotation_type='poly'):
        """ get complete/smoothed pr curve for entire dataset """
        # infer on 0-decision threshold iterate over diff thresholds extend_pr
        # compute_ap save output

        # infer on dataset with 0-decision threshold
        predictions = self.infer_dataset(dataset,
                                         conf_thresh=0,
                                         iou_thresh=nms_iou_thresh,
                                         save_folder=save_folder,
                                         save_subfolder=os.path.join(
                                             'prcurve', 'detections'),
                                         imshow=imshow,
                                         imsave=imsave,
                                         image_name_suffix='_prcurve_0',
                                         annotation_type=annotation_type)

        # iterate over different decision thresholds
        prec = []
        rec = []
        f1score = []
        tps = []
        fps = []
        fns = []
        start_time = time.time()
        for c, conf in enumerate(confidence_thresh):
            print('{}: outcome confidence threshold: {}'.format(c, conf))

            data_full, p, r, f1 = self.compute_pr_dataset(dataset,
                                                  predictions,
                                                  conf,
                                                  nms_iou_thresh,
                                                  imsave=imsave,
                                                  save_folder=save_folder)
            

            prec.append(p)
            rec.append(r)
            f1score.append(f1)
            tps.append(sum([sum(data_full[idx]['tp']) for idx in range(len(data_full))]))
            fps.append(sum([sum(data_full[idx]['fp']) for idx in range(len(data_full))]))
            fns.append(sum([sum(data_full[idx]['fn_gt'])+sum(data_full[idx]['fn_dt']) 
                       for idx in range(len(data_full))]))

        end_time = time.time()

        sec = end_time - start_time
        print('prcurve time: {} sec'.format(sec))
        print('prcurve time: {} min'.format(sec / 60.0))
        print('prcurve time: {} hrs'.format(sec / 3600.0))

        rec = np.array(rec)
        prec = np.array(prec)
        f1score = np.array(f1score)
        tps = np.array(tps)
        fps = np.array(fps)
        fns = np.array(fns)

        # Trim rec and prec to remove any moments where rec == 0 (they mean nothing)
        prec = prec[rec > 0]
        f1score = f1score[rec > 0]
        tps = tps[rec > 0]
        fps = fps[rec > 0]
        fns = fns[rec > 0]
        rec = rec[rec > 0]

        # plot raw PR curve
        fig, ax = plt.subplots()
        ax.plot(rec, prec, marker='o', linestyle='dashed')
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.title('precision-recall for varying confidence')
        os.makedirs(save_folder, exist_ok=True)
        save_plot_name = os.path.join(
            save_folder, self._model_name + '_pr_raw.png')
        plt.savefig(save_plot_name)
        if PLOT:
            plt.show()

        # plot F1score
        # fig, ax = plt.subplots() ax.plot(rec, f1score, marker='o',
        # linestyle='dashed') plt.xlabel('recall') plt.ylabel('f1 score')
        # plt.title('f1 score vs recall for varying confidence') save_plot_name
        # = os.path.join('output', save_name, save_name + '_test_f1r.png')
        # plt.savefig(save_plot_name) plt.show()

        # smooth the PR curve: take the max precision values along the recall
        # curve we do this by binning the recall values, and taking the max
        # precision from each bin

        p_final, r_final, c_final, = self.extend_pr(
            prec, rec, confidence_thresh)
        ap = self.compute_ap(p_final, r_final)

        # plot final pr curve
        fig, ax = plt.subplots()
        ax.plot(r_final, p_final)
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.title(
            'prec-rec curve, iou={}, ap = {:.2f}'.format(decision_iou_thresh, ap))
        # ax.legend()
        save_plot_name = os.path.join(
            save_folder, self._model_name + '_pr.png')
        plt.savefig(save_plot_name)
        if PLOT:
            plt.show()

        print('ap score: {:.5f}'.format(ap))
        print('max f1 score: {:.5f}'.format(max(f1score)))
        max_f1_idx = np.argmax(f1score)
        print(f'TP: {tps[max_f1_idx]}; FP: {fps[max_f1_idx]}; FN: {fns[max_f1_idx]}')

        # save ap, f1score, precision, recall, etc
        res = {'precision': p_final,
               'recall': r_final,
               'ap': ap,
               'f1score': f1score,
               'confidence': c_final}
        save_file = os.path.join(
            save_folder, self._model_name + '_prcurve.pkl')
        with open(save_file, 'wb') as f:
            pickle.dump(res, f)

        return res


    def get_p_from_r(self, prec, rec, r):
        """ given precision and recall curve, given r, interpolate to find p"""
        # check prec and rec are same size
        f = interp1d(rec, prec)
        p = f(r)
        return p


    def compare_models(self,
                    models, # dictionary: model_type, model_name, model_folder
                    datasets, # corresponding length of models
                    confidence_thresh=None,
                    decision_iou_thresh=0.5,
                    nms_iou_thresh=0.5,
                    rgoal=0.7,
                    save_prcurve_dir=None,
                    save_plot_name=None,
                    load_prcurve=False,
                    pr_files=None,
                    save_prcurve_images=False,
                    show_fig=False):
        """ compute pr curves of each model in models (a list of models)
            then plot prcurves together and save """

        # TODO: update for multiple classes? currently only built for single class detection

        # model['model_type'] = 'box' or 'poly'
        # model['model_name'] = name of the model/pointer to model folder (normally the same)
        # model['model_folder'] = location of model folder/.pth file
        # model['model_description'] = descriptor for the legend (eg 'MaskRCNN_Horehound')
        # model['epochs'] = list of epochs that correspond to which snapshot to select

        model_names = models['name']
        model_folders = model_names
        model_descriptions = models['description']
        model_types = models['type']
        model_epochs = models['epoch']
        # datasets = full path to .pkl files

        # TODO ensure proper input in
        # setup default values
        if confidence_thresh is None:
            confidence_thresh = np.linspace(0.99, 0.01, num=25, endpoint=True)
            confidence_thresh = np.array(confidence_thresh, ndmin=1)

        # for each model
        #   build model
        #   get prcurve (option to load already done pkl file)
        prds = []
        for i, name in enumerate(model_names):

            # get pr curve
            if load_prcurve:
                # see replot_prcurves.py
                if pr_files is None:
                    pr_pkl = glob.glob(str('output/' + name + '/prcurve/*_prcurve.pkl'))
                    print(pr_pkl)
                    if len(pr_pkl) <= 0:
                        print(f'ERROR: prcurve pickle file not found: {pr_pkl}')
                    elif len(pr_pkl) > 1:
                        print('ERROR: multiple pickle files found. Printing:')
                        for p in pr_pkl:
                            print(p)
                    else:
                        # len(pr_pkl) == 1
                        if os.path.isfile(pr_pkl[0]):
                            with open(pr_pkl[0], 'rb') as f:
                                prd = pickle.load(f)
                else:
                    if os.path.isfile(pr_files[i]):
                        with open(pr_files[i], 'rb') as f:
                            prd = pickle.load(f)
            else:
                # load dataset object to evaluate
                dso = self.load_dataset_objects(datasets[i])

                # set model
                model_path = os.path.join('output', model_folders[i], model_folders[i] + '.pth')
                self.load_model(model_path, annotation_type=model_types[i])
                self.set_model_name = name
                self.set_model_path(model_path)
                self.set_snapshot(model_epochs[i])

                if save_prcurve_dir is None:
                    save_prcurve_dir = os.path.join(model_folders[i], 'prcurve')

                # get pr curve
                prd = self.get_prcurve(dso['ds_test'],
                                        confidence_thresh=confidence_thresh,
                                        nms_iou_thresh=nms_iou_thresh,
                                        decision_iou_thresh=decision_iou_thresh,
                                        save_folder=save_prcurve_dir,
                                        imsave=save_prcurve_images,
                                        annotation_type=model_types[i])
            prds.append(prd)

        # do plot (should be its own method)
        fig, ax = plt.subplots()
        ap_list = []
        for i, prd in enumerate(prds):
            self.plot_prcurve(prd['precision'],
                                prd['recall'],
                                prd['ap'],
                                model_descriptions[i],
                                ax=ax)
            ap_list.append(prd['ap'])

        ax.legend()
        plt.title('model comparison: PR Curve')
        mdl_names_str = "".join(model_descriptions)
        save_plot_name = os.path.join('output', 'model_compare_' +  mdl_names_str + '.png')
        plt.savefig(save_plot_name)
        if show_fig:
            plt.show()

        pgoals, cgoals = self.get_pc_from_rgoal(prds, rgoal)
        print('model comparison complete')
        for i, m in enumerate(model_names):
            print(str(i) + ' model: ' + m)
            print(str(i) + ' name: ' + model_descriptions[i])
            print(f'{str(i)}: rgoal = {rgoal}')
            print(f'{str(i)}: pgoal = {pgoals[i]}')
            print(f'{str(i)}: conf = {cgoals[i]}')

        return True


    def get_pc_from_rgoal(self, prdicts, rgoal=0.7):
        """ get precision, confidence values from recall goal using pr dictionaries """

        pgoal = []
        cgoal = []
        for i, pr in enumerate(prdicts):
            p = pr['precision']
            r = pr['recall']
            c = pr['confidence']
            f1 = pr['f1score']
            cg = self.get_confidence_from_pr(p, r, c, f1, rg=rgoal)
            pg = self.get_p_from_r(p, r, rgoal)

            pgoal.append(pg)
            cgoal.append(cg)

        return pgoal, cgoal



    def plot_prcurve(self, p, r, ap, name, title_str=None, ax=None):
        """ plot pr curve on given ax """
        if ax is None:
            fig, ax = plt.subplots()
        m_str = 'm={}, ap={:.2f}'.format(name, ap)
        ax.plot(r, p, label=m_str)
        plt.xlabel('recall')
        plt.ylabel('precision')
        if title_str is None:
            title_str = str('pr curve: ' + name)
        plt.title(title_str)
        plt.grid(True)
        return ax