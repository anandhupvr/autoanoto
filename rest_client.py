import requests
import matplotlib.pyplot as plt
import numpy as np
import sys
from skimage.measure import find_contours



pic = sys.argv[1]

img = plt.imread(pic).tolist()




def save_contours(mask):
    polygon_ = []
    padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
    # mask_ = np.array(mask).transpose([1,0])
    padded_mask[1:-1, 1:-1] = mask
    contours = find_contours(padded_mask, 0.5)
    for verts in contours:
        polygon_.append(verts.tolist())
        # p = Polygon(verts, facecolor="none", edgecolor='red')
    return polygon_





def main():
  # we have used simple tensorflow servings to serve the model  
  endpoint = "http://localhost:port"
  json_data = {"model_name": "default", "data": {"images": [img]} }
  result = requests.post(endpoint, json=json_data)
  out_img = result.json()
  pred = out_img['out']
  pred = np.array(pred)
  polygon = save_contours(pred[0])
  # mask = pred[0]
  # for i in range(mask.shape[0]):
  # 	for j in range(mask.shape[1]):
  # 		if mask[i,j] != 2:
  # 			mask[i, j] = 0
  # import pdb; pdb.set_trace()
  plt.imshow(pred[0])
  plt.show()

if __name__ == "__main__":
  main()
  
