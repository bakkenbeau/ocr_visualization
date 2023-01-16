import torch
import os
import time
import cv2
from CRAFT_pytorch.test import test_net
from CRAFT_pytorch.craft import CRAFT
from CRAFT_pytorch.test import copyStateDict
import CRAFT_pytorch.imgproc

import CRAFT_pytorch.craft_functions

# def runCRAFTSingleImage(image_path):
#     net = CRAFT()
#
#     weights = "CRAFT_pytorch/weights/craft_mlt_25k.pth"
#     net.load_state_dict(copyStateDict(torch.load(weights, map_location='cpu')))
#
#     print("Made it here")
#
#     net.eval()
#
#     t = time.time()
#
#     # load data
#     image = CRAFT_pytorch.imgproc.loadImage(image_path)
#     text_threshold = 0.7
#     link_threshold = 0.4
#     low_text = 0.4
#     cuda = False
#     poly = False
#     refine_net = None
#
#     bboxes, polys, score_text = test_net(net, image, text_threshold, link_threshold, low_text,
#                                          cuda, poly, refine_net)
#
#     result_folder = './result/'
#     if not os.path.isdir(result_folder):
#         os.mkdir(result_folder)
#
#     # save score text
#     filename, file_ext = os.path.splitext(os.path.basename(image_path))
#     mask_file = result_folder + "/res_" + filename + '_mask.jpg'
#     cv2.imwrite(mask_file, score_text)
#
#     CRAFT_pytorch.file_utils.saveResult(image_path, image[:, :, ::-1], polys, dirname=result_folder)
#
#     print("elapsed time : {}s".format(time.time() - t))


if __name__ == '__main__':
    print("running function")

    image_option_1 = "C:\\Users\\BeauBakken\\OneDrive - CASPIATECHNOLOGIES\\Desktop\\Personal" \
                     "\\Coding\\NJ\\result\\res_ic1.jpg"
    image_option_2 = r'C:\Users\BeauBakken\PycharmProjects\NathanAssignment1\CRAFT_pytorch\test_images\STOP_sign.jpg'

    CRAFT_pytorch.craft_functions.runCRAFTSingleImage(image_option_2)
