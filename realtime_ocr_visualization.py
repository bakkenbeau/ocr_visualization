"""
Author: Beau Bakken
Date: 1//23
Description: Application to display text detection and text recognition results for an uploaded image,
             with features to filter results and in-paint bounding boxes to remove text from image.
             Also was an exercise to explore PyQT, pyqtgraph, and qtextras libraries.
             Currently uses CRAFT and EasyOCR.
             Application will save detected text regions to result directory specified in variable
             'ocr_result_dir' located in 'main'.
Resources:
https://pyqtgraph.readthedocs.io/en/latest/
https://pypi.org/project/qtextras/
https://github.com/clovaai/CRAFT-pytorch
https://github.com/JaidedAI/EasyOCR
https://stackoverflow.com/questions/69043530/attributeerror-module-numpy-core-has-no-attribute-numerictypes
     (fixed debugger issues)
"""

# custom-made library from downloaded CRAFT repository, functions located in "craft\craft_functions.py"
# runCRAFTSingleImage is currently the only supported function
from craft.craft_functions import runCRAFTSingleImage

from qtpy import QtWidgets, QtGui
import os
import pyqtgraph as pg
from pyqtgraph.parametertree import RunOptions
import numpy as np
from PIL import Image
import easyocr
from qtextras import MaskCompositor, bindInteractorOptions as bind
# To stay updated on this custom library:
# https://stackoverflow.com/questions/19943022/import-a-python-library-from-github
# pip install git+https://gitlab.com/s3a/qtextras.git

image_option_1 = r'C:\Users\BeauBakken\PycharmProjects\ocr_visualization\test_images\STOP3_sign.jpg'

pg.setConfigOption("imageAxisOrder", "row-major")


class TextDisplayBox(QtWidgets.QGraphicsTextItem):
    def paint(self, painter, option, widget):
        # good common practice to save and revert paint state when modifying
        painter.save()
        painter.setBrush(pg.mkBrush("#ffffff88"))
        painter.drawRect(self.boundingRect())
        painter.restore()
        super().paint(painter, option, widget)


class TextRecItem(QtWidgets.QGraphicsRectItem):
    """
    Bounding box graphic to display text recognition results
    Subclassing QGraphicsRectItem is needed to change hover behavior and show predicted text and confidence values
    https://stackoverflow.com/questions/47145117/subclassing-qgraphicsrectitem-in-pyqt
    """
    def __init__(self, text: str, confidence: float, *args, **kwargs):
        QtWidgets.QGraphicsRectItem.__init__(self, *args, **kwargs)
        self.setAcceptHoverEvents(True)
        self.text = text
        self.confidence = confidence

        self.display_text = TextDisplayBox(text)
        self.display_text.setParentItem(self)
        self.display_text.hide()
        self.display_text.setPos(self.rect().topLeft())

        # self.display_text = QtWidgets.QGraphicsTextItem()
        # self.display_text.setPlainText("Prediction: " + text + '\n' + "Confidence: " + str(confidence))
        # self.display_text.setParentItem(self)
        # self.display_text.hide()

        #self.display_text.setPos(self.rect().topLeft())
        viewer.addItem(self.display_text)

    def hoverEnterEvent(self, event):
        print("hovered enter")
        self.display_text.show()

    def hoverLeaveEvent(self, event):
        print("hovered exit")
        self.display_text.hide()


class TextRecControl:
    """
    Control class for text recognition functions
    Currently the primary function is to run EasyOCR text recognition, store all text recognition results as
    TextRecItems and draw them to the scene (a.k.a the "viewer" global variable)
    """
    def __init__(self):
        self.predict_items = []

    def predict_and_draw(self, imgs, coords, img_file_path, res_dir):
        # this needs to run only once to load the model into memory
        reader = easyocr.Reader(['en'], gpu=False)

        count = 0
        for img, coord in zip(imgs, coords):
            # save detected text region(s)
            text_img = Image.fromarray(img)
            text_img.save(self.get_text_only_image_path_wo_ext(img_file_path, res_dir) + '_word_' + str(count) + '.jpg')
            count += 1

            result = reader.readtext(img)
            if not result:
                # if text recognition comes back empty, could not detect anything
                # common when the text detected was a logo graphic
                # create bounding box over detected text with custom QGraphicsRectItem
                detected_text_box = TextRecItem("N/A", 0.0, float(coord[0]), float(coord[1]),
                                                float(coord[2]), float(coord[3]))

                # change bounding box color and size
                detected_text_box.setPen(QtGui.QPen(QtGui.QColor("black"), 0.1))

                viewer.addItem(detected_text_box)
                self.predict_items.append(detected_text_box)
            else:
                text = result[0][1]
                confidence = result[0][2]

                # create bounding box over detected text with custom QGraphicsRectItem
                detected_text_box = TextRecItem(text, confidence, float(coord[0]), float(coord[1]),
                                                                            float(coord[2]), float(coord[3]))

                # change bounding box color and size
                detected_text_box.setPen(QtGui.QPen(QtGui.QColor("black"), 0.1))

                viewer.addItem(detected_text_box)
                self.predict_items.append(detected_text_box)
        return

    def get_text_only_image_path_wo_ext(self, img_file_path, res_dir):
        base_img_path_words = img_file_path.split('\\')
        base_img_name_w_ext = base_img_path_words[-1]
        base_img_name_wo_ext = base_img_name_w_ext.split('.')[0]
        text_only_image_path = os.getcwd() + r'\result\res_' + base_img_name_wo_ext + '_TEXT_ONLY'
        return text_only_image_path


class TextDetectItem(QtWidgets.QGraphicsRectItem):
    """
    Bounding box graphic to display text detection overlay
    """
    def __init__(self, raw_coordinates, *args, **kwargs):
        self.raw_coordinates = raw_coordinates
        # initializing a QGraphicsRectItem requires the (x,y) coordinate of top left corner, and width and height.
        # the results from text detection algorithm give the coordinates for all four corners of the predicted bounding
        # box. Therefore, we need to extract the width and height.
        # this is not currently optimized for angled text, as we would need to change the angle of the TextDetectItem to
        # match the predicted bounding box. A task for the future.
        self.x, self.y, self.w, self.h = self.get_straight_rectangle_coordinates()
        QtWidgets.QGraphicsRectItem.__init__(self, float(self.x), float(self.y), float(self.w), float(self.h))

    def get_straight_rectangle_coordinates(self):
        w = self.raw_coordinates[1][0] - self.raw_coordinates[0][0]
        h = self.raw_coordinates[2][1] - self.raw_coordinates[1][1]
        return self.raw_coordinates[0][0], self.raw_coordinates[0][1], w, h


class TextDetectControl:
    """
    Control class for text detection functions
    Currently the primary function is to run CRAFT text detection, store all text detection results as
    TextDetectItems and draw them to the scene (a.k.a the "viewer" global variable)
    """
    def __init__(self, res_dir: str):
        self.detect_items = []
        # directory to store CRAFT output results
        self.res_dir = res_dir
        self.res_mask_txt_path = ""

    def detect_and_draw(self, img_file_path):
        # CRAFT produces an image with red bounding box, a heatmap image, and text file of coordinates
        # of resulting mask in the provided directory. The number of lines in the mask text folder correspond
        # to the number of detected words (empty line between bounding box coordinates)
        # TODO Optimize to run concurrently to not stall the application and error checking
        error = runCRAFTSingleImage(img_file_path, self.res_dir)

        # file path to mask text file
        self.get_res_mask_txt_file_path(img_file_path)

        # open mask text file and read each line
        with open(self.res_mask_txt_path) as f:
            raw_mask_txt_lines = f.readlines()

        # strip the newline characters
        count = 0
        mask_txt_lines = []
        for line in raw_mask_txt_lines:
            if line == '\n':
                continue
            count += 1
            mask_txt_lines.append(line.strip('\n'))

        # create TextDetectItems for each detected word
        items = []
        for line in mask_txt_lines:
            x1 = line.split(',')[0]
            y1 = line.split(',')[1]
            x2 = line.split(',')[2]
            y2 = line.split(',')[3]
            x3 = line.split(',')[4]
            y3 = line.split(',')[5]
            x4 = line.split(',')[6]
            y4 = line.split(',')[7]

            rect = TextDetectItem([[int(x1), int(y1)], [int(x2), int(y2)], [int(x3), int(y3)], [int(x4), int(y4)]])
            rect.setPen(QtGui.QPen(QtGui.QColor("red"), 2.0))
            items.append(rect)
            viewer.addItem(rect)

        self.detect_items = items
        return

    def get_res_mask_txt_file_path(self, img_file_path):
        base_img_file_path_words = img_file_path.split('\\')
        base_img_name_w_ext = base_img_file_path_words[-1]
        base_img_name_wo_ext = base_img_name_w_ext.split('.')[0]
        self.res_mask_txt_path = os.getcwd() + r'\result\res_' + base_img_name_wo_ext + '.txt'
        return self.res_mask_txt_path

    def redraw_item_at_indx(self, indx, color):
        item = self.detect_items[indx]
        viewer.removeItem(item)
        item.setPen(QtGui.QPen(QtGui.QColor(color), 2.0))
        viewer.addItem(item)


class OCRSceneContainer:
    """
    Container class to control all OCR results and corresponding graphic items in a scene
    TODO currently the scene is stored as 'viewer' global variable. Modify classes to take a given scene as input.
    """
    def __init__(self, ocr_res_dir):
        self.ocr_res_dir = ocr_res_dir
        self.text_detection_control = TextDetectControl(ocr_res_dir)
        self.text_rec_control = TextRecControl()
        self.base_image_file_path = ""
        self.base_image_np = []
        self.img = None
        self.detected_text = False
        self.res_image_path = ""
        self.res_mask_path = ""

        # roi is initialized and added to the viewer right away, but shrunken and enlarged when disabled
        # and enabled respectively
        self.roi = pg.RectROI([0, 0], [0, 0], pen=(0, 9))
        self.roi_brush_selector = False
        self.roi.sigRegionChanged.connect(self.copy_roi_texture)
        viewer.addItem(self.roi)

    # select an image from a directory
    @bind(base_image_file_path=dict(type="file", nameFilter="*.jpg"))
    def select_img(self, base_image_file_path=image_option_1):
        # delete any old images if a new base image has been selected
        viewer.clearOverlays()

        # TODO check image quality. Currently the application is optimized for
        #  smaller (>500x500px) images to see the results. In the future, should display
        #  results such as predicted text as a percentage of the overall image size so the text is not tiny

        # load selected image as main backdrop
        self.base_image_file_path = base_image_file_path
        base_image = Image.open(self.base_image_file_path)
        base_image_np = np.asarray(base_image)
        self.base_image_np = base_image_np
        self.img = pg.ImageItem(self.base_image_np)

        viewer.addItem(self.img)

    def detect_and_draw(self):
        self.text_detection_control.detect_and_draw(self.base_image_file_path)
        self.detected_text = True

    def predict_and_draw(self):
        if self.detected_text:
            imgs = []
            coords = []

            # grab each detected text region and place in list
            text_items = self.text_detection_control.detect_items
            for item in text_items:
                # TODO how to get PyCharm to recognize this as a TextDectItem? Might not be possible
                x, y, w, h = item.get_straight_rectangle_coordinates()
                coords.append([x, y, w, h])
                text_region_np = self.base_image_np[y:y + h, x:x + w, :]
                imgs.append(text_region_np)
            self.text_rec_control.predict_and_draw(imgs, coords, self.base_image_file_path, self.ocr_res_dir)
        else:
            # no text has been detected, therefore run text recognition on the entire image
            # TextRecControl expects list of images
            img = [self.base_image_np]

    @bind(confidence_threshold=dict(step=0.1, limits=[0, 1]))
    def filter_text_rec_results(self, confidence_threshold=0.9):
        # TODO determine if text rec results exist
        count = 0
        for item in self.text_rec_control.predict_items:
            if item.confidence > confidence_threshold:
                # turn rect border green by replacing item
                self.text_detection_control.redraw_item_at_indx(count, "green")
            else:
                # turn rect border red by replacing it
                self.text_detection_control.redraw_item_at_indx(count, "red")
            count += 1

    def toggle_roi_brush_selector(self):
        self.roi_brush_selector = not self.roi_brush_selector
        if self.roi_brush_selector:
            self.roi.setSize([20, 20])
        else:
            self.roi.setSize([0, 0])

    def copy_roi_texture(self):
        texture = self.roi.getArrayRegion(self.base_image_np, self.img)

        # convert extracted texture image to QPixMap in order to create a new QBrush
        # https://stackoverflow.com/questions/61910137/convert-python-numpy-array-to-pyqt-qpixmap-image-result-in-noise-image
        texture_np = np.asarray(texture, dtype='uint8')
        h, w, rgb = texture_np.shape
        bytesPerLine = 3 * w
        qImg = QtGui.QImage(texture_np.data, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)
        intermediate_pixmap = QtGui.QPixmap(qImg)
        texture_pixmap_image = QtGui.QPixmap(intermediate_pixmap)
        texture_brush = QtGui.QBrush(texture_pixmap_image)

        # in-paint TextRecItems with roi texture
        for item in self.text_rec_control.predict_items:
            # remove item from scene
            viewer.removeItem(item)
            # redraw with new texture brush
            item.setBrush(texture_brush)
            viewer.addItem(item)


if __name__ == '__main__':

    application = pg.mkQApp()

    # MaskCompositor is a subclass of an ImageViewer and allows you to easily overlay images from
    # Nathan Jessurun's custom PyQt library: https://pypi.org/project/qtextras/
    viewer = MaskCompositor()

    # OCRSceneContainer is custom class to manipulate a base image and easily place graphic items from
    # OCR machine learning/image processing algorithms
    # some OCR functions output files, such as heatmap images. ocr_result_dir specifies where to save them.
    ocr_result_dir = './result/'
    ocr_scene_container = OCRSceneContainer(ocr_result_dir)

    # function to load a base image
    viewer.toolsEditor.registerFunction(
        ocr_scene_container.select_img, name="Select Image", runOptions=[RunOptions.ON_CHANGED, RunOptions.ON_ACTION,
                                                                         RunOptions.ON_CHANGING]
    )

    # function to run text detection on base image
    viewer.toolsEditor.registerFunction(
        ocr_scene_container.detect_and_draw, name="Detect Text", runOptions=[RunOptions.ON_ACTION]
    )

    # function to run text recognition on text detection results
    viewer.toolsEditor.registerFunction(
        ocr_scene_container.predict_and_draw, name="Predict Text", runOptions=[RunOptions.ON_ACTION]
    )

    # function to filter text recognition results
    viewer.toolsEditor.registerFunction(
        ocr_scene_container.filter_text_rec_results, name="Filter Threshold", runOptions=[RunOptions.ON_CHANGED]
    )

    # button to toggle ROI selector and in-painting predicted text bounding boxes
    viewer.toolsEditor.registerFunction(
        ocr_scene_container.toggle_roi_brush_selector, name="Toggle Threshold", runOptions=[RunOptions.ON_ACTION]
    )

    window = viewer.widgetContainer()
    window.show()

    pg.exec()
