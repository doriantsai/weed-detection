import math
import sys
import time
import torch
import torchvision.models.detection.mask_rcnn

import weed_detection.utils as utils

# from weed_detection.coco_utils import get_coco_api_from_dataset
# from weed_detection.coco_eval import CocoEvaluator

# from get_prediction import get_prediction_image


def train_one_epoch(model, optimizer, data_loader_train, data_loader_val, device, epoch, val_epoch, print_freq):
    """
    train_one_epoch trains a single epoch for the model
    - data_loader_train - data loader for the training set
    - data_loader_val - data loader for the validation set, which evaluates at a frequency defined by int "val_epoch"
    """

    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    # added for validation
    metric_logger_val = utils.MetricLogger(delimiter="  ")
    metric_logger_val.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader_train) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader_train, print_freq, header):
        images = list(image.to(device) for image in images)

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # import code
        # code.interact(local=dict(globals(), **locals()))

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # perform validation on the validation dataset every val_epoch epochs
    if (epoch % val_epoch) == (val_epoch - 1):

        for images, targets in metric_logger.log_every(data_loader_val, 10, header):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_outputs = model(images, targets)
            loss_output_reduced = utils.reduce_dict(loss_outputs)
            losses_reduced = sum(loss for loss in loss_output_reduced.values())
            metric_logger_val.update(loss=losses_reduced, **loss_output_reduced)

    return metric_logger, metric_logger_val


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


# @torch.no_grad()
# def evaluate(model, data_loader, device, conf, iou, class_names):
#     n_threads = torch.get_num_threads()
#     # FIXME remove this and make paste_masks_in_image run on the GPU
#     torch.set_num_threads(1)
#     cpu_device = torch.device("cpu")

#     model.to(device)
#     model.eval()
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     header = 'Test:'

#     coco = get_coco_api_from_dataset(data_loader.dataset)
#     iou_types = _get_iou_types(model)
#     coco_evaluator = CocoEvaluator(coco, iou_types)

#     for images, targets in metric_logger.log_every(data_loader, 10, header):

#         # if not isinstance(images, list) and (len(images) == 1):
#         #     images = [images]
#         images = list(img.to(device) for img in images)

#         # if not isinstance(targets, list) and (len(targets) == 1):
#             # targets = [targets]
#         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]  # added

#         torch.cuda.synchronize()

#         model_time = time.time()

#         # import code
#         # code.interact(local=dict(globals(), **locals()))

#         # if we have a tuple of images:
#         # imgs = list(img for img in images)
#         # if we have a single image:
#         # output_raw = model(images) # no nms, conf thresh

#         # outputs = get_prediction_batch(model, images, conf, iou, device, class_names)

#         # raw_outputs = model(images)

#         # do non-maxima suppression for evaluation
#         outputs = []
#         for i in range(len(images)):
#             # print(i)
#             out, _ = get_prediction_image(model,
#                                             images[i],
#                                             conf,
#                                             iou,
#                                             device,
#                                             class_names)
#             # need to convert out values into tensors, so we can send this to the device (GPU)
#             # if no boxes, just make out empty TODO trying to deal w batch cases
#             if len(out['boxes']) == 0:
#                 # print('warning: len boxes 0')
#                 # print('output through get_prediction_image')
#                 # print(out)
#                 # print('output through model')
#                 # output_raw = model([images[i]])
#                 # print(output_raw)
#                 # import code
#                 # code.interact(local=dict(globals(), **locals()))

#                 continue
#             #     out = []


#             outputs.append(out)

#         # outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
#         outputs = [{k: torch.tensor(v).to(device) for k, v in t.items()} for t in outputs]
#         model_time = time.time() - model_time

#         res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}

#         evaluator_time = time.time()
#         # this is where the magic happens
#         coco_evaluator.update(res)
#         evaluator_time = time.time() - evaluator_time
#         metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
#     coco_evaluator.synchronize_between_processes()

#     # accumulate predictions from all images
#     coco_evaluator.accumulate() # should have PR matrix
#     #  coco_evaluator.eval['precision']

#     # import code
#     # code.interact(local=dict(globals(), **locals()))
#     # print('calling get_eval')
#     ccres = coco_evaluator.get_eval()
#     # print(type(ccres))
#     # ccres['precision'].shape

#     # import code
#     # code.interact(local=dict(globals(), **locals()))

#     coco_evaluator.summarize()
#     torch.set_num_threads(n_threads)
#     return metric_logger, ccres
