import tensorflow as tf
import numpy as np
from PIL import Image
# import matplotlib.pyplot as plt
from skimage.measure import find_contours


graph = tf.Graph()



class fg:
    def __init__(self):
        self.INPUT_TENSOR_NAME = 'ImageTensor:0'
        self.OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
        self.INPUT_SIZE = 513

        self.graph = tf.Graph()
        with tf.gfile.GFile('/run/media/user1/disk2/agrima/git_repos/object-localization/model_zoo/tf_deeplab/deeplabv3_mnv2_pascal_trainval/frozen_inference_graph.pb', 'rb') as f:
            self.graph_def = tf.GraphDef()
            self.graph_def.ParseFromString(f.read())

        with self.graph.as_default():
            tf.import_graph_def(self.graph_def, name='')

        self.sess = tf.Session(graph=self.graph)


    def save_contours(self, mask):
        polygon_ = []
        height, width = mask.shape[0], mask.shape[1]
        padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        # mask_ = np.array(mask).transpose([1,0])
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)

        for verts in contours:
            verts = np.fliplr(verts) - 1
            polygon_.append(verts)
            # p = Polygon(verts, facecolor="none", edgecolor='red')
            # print ("hello")
        return polygon_

    def mod(self, image_path, crop_size):
        import pdb; pdb.set_trace()
        image = Image.open(image_path)
        width, height = image.size
        crop_im = image.crop(crop_size)
        c_width, c_height = crop_im.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(c_width, c_height)
        target_size = (int(resize_ratio * c_width), int(resize_ratio * c_height))
        resized_image = crop_im.convert('RGB').resize(target_size, Image.ANTIALIAS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]
        import pdb; pdb.set_trace()
        poly = self.save_contours(seg_map)
        poly_ = np.array(poly)
        poly_[0][:,0:1] *= crop_im.size[0] / resized_image.size[0]
        poly_[0][:, 1:2] *= crop_im.size[1] / resized_image.size[1]
        poly_[0][:,0:1] += crop_size[0]
        poly_[0][:, 1:2] += crop_size[1]
        # p = Polygon(poly_[0], facecolor="none", edgecolor='red')
        # fig, ax = plt.subplots(1)
        # ax.imshow(image)
        # ax.add_patch(p)
        # # plt.savefig(image_path.split('/')[-1])
        # plt.show()
        return poly_
 
# image_path = '/home/user1/Downloads/cab.jpg'
