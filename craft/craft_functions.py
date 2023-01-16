import torch
import os
import time
import cv2
from craft.test import test_net
from craft.craft import CRAFT
from craft.test import copyStateDict
import craft.imgproc


def runCRAFTSingleImage(image_path, result_folder):
    net = CRAFT()

    weights = "craft/weights/craft_mlt_25k.pth"
    net.load_state_dict(copyStateDict(torch.load(weights, map_location='cpu')))

    net.eval()

    t = time.time()

    # load data
    image = craft.imgproc.loadImage(image_path)
    text_threshold = 0.7
    link_threshold = 0.4
    low_text = 0.4
    cuda = False
    poly = False
    refine_net = None

    bboxes, polys, score_text = test_net(net, image, text_threshold, link_threshold, low_text,
                                         cuda, poly, refine_net)

    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)

    # save score text
    filename, file_ext = os.path.splitext(os.path.basename(image_path))
    mask_file = result_folder + "/res_" + filename + '_mask.jpg'
    cv2.imwrite(mask_file, score_text)

    craft.file_utils.saveResult(image_path, image[:, :, ::-1], polys, dirname=result_folder)

    print("elapsed time : {}s".format(time.time() - t))

    error = 0
    # if result files don't exist, we had a problem
    # TODO: Could maybe be optimized to read results from a function call instead of checking if files exist

    return error
