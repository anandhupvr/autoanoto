import tensorflow as tf
import numpy as np
from PIL import Image
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.patches as patches
# import cv2

graph = tf.Graph()



class fg:

    def __init__(self):
        self.INPUT_TENSOR_NAME = 'ImageTensor:0'
        self.OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
        self.INPUT_SIZE = 513

        self.graph = tf.Graph()
        with tf.gfile.GFile('custom/frozen_inference_graph.pb', 'rb') as f:
            self.graph_def = tf.GraphDef()
            self.graph_def.ParseFromString(f.read())

        with self.graph.as_default():
            tf.import_graph_def(self.graph_def, name='')

        self.sess = tf.Session(graph=self.graph)


    def save_contours(self, mask, c_size, r_size):
        polygon_ = []
        height, width = mask.shape[0], mask.shape[1]
        padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        # mask_ = np.array(mask).transpose([1,0])
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            verts = np.fliplr(verts) - 1
            verts[:][:, 0:1] *= c_size[0] / r_size[0]
            verts[:][:, 1:2] *= c_size[1] / r_size[1]
            polygon_.append(verts.tolist())
            # p = Polygon(verts, facecolor="none", edgecolor='red')
        return polygon_

    def mod(self, image_path, *args):

        image = Image.open(image_path)
        poly = []
        width, height = image.size
        crop_im = image
        c_width, c_height = crop_im.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(c_width, c_height)
        target_size = (int(resize_ratio * c_width), int(resize_ratio * c_height))
        resized_image = crop_im.convert('RGB').resize(target_size, Image.ANTIALIAS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]
        np.unique(seg_map)
        img = plt.imshow(seg_map)
        plt.show()
        poly = self.save_contours(seg_map, crop_im.size, resized_image.size)

        return poly