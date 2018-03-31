import sys

sys.path.append('./')

from yolo.net.yolo_tiny_net import YoloTinyNet 
import tensorflow as tf 
# import cv2
import numpy as np
import os
import PIL.Image as Image

classes_name =  ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]


def process_predicts(predicts):
  p_classes = predicts[0, :, :, 0:20]
  C = predicts[0, :, :, 20:22]
  coordinate = predicts[0, :, :, 22:]

  p_classes = np.reshape(p_classes, (7, 7, 1, 20))
  C = np.reshape(C, (7, 7, 2, 1))

  P = C * p_classes

  #print P[5,1, 0, :]

  index = np.argmax(P)


  index = np.unravel_index(index, P.shape)
  prob = P[index]
  class_num = index[3]

  coordinate = np.reshape(coordinate, (7, 7, 2, 4))

  max_coordinate = coordinate[index[0], index[1], index[2], :]

  xcenter = max_coordinate[0]
  ycenter = max_coordinate[1]
  w = max_coordinate[2]
  h = max_coordinate[3]

  xcenter = (index[1] + xcenter) * (448/7.0)
  ycenter = (index[0] + ycenter) * (448/7.0)

  w = w * 448
  h = h * 448

  xmin = xcenter - w/2.0
  ymin = ycenter - h/2.0

  xmax = xmin + w
  ymax = ymin + h

  return xmin, ymin, xmax, ymax, class_num, prob

common_params = {'image_size': 448, 'num_classes': 20, 
                'batch_size':1}
net_params = {'cell_size': 7, 'boxes_per_cell':2, 'weight_decay': 0.0005}

net = YoloTinyNet(common_params, net_params, test=True)

image = tf.placeholder(tf.float32, (1, 448, 448, 3))
predicts = net.inference(image)
sess = tf.Session()
saver = tf.train.Saver(net.trainable_collection)
saver.restore(sess, 'models/pretrain/yolo_tiny.ckpt')

count=0
recongnized_cnt=0
for (root, dirs, files) in os.walk(r"D:\Programming\fsai_z\data\warmup\Images\skirt_length_labels"):
#for (root, dirs, files) in os.walk(os.path.join(os.path.split(__file__)[0],"imgs")):
  for file in files:
    with Image.open(os.path.join(root,file)) as img:
      original_shape=img.size
      resized_img=img.resize((448, 448), Image.ANTIALIAS)
      np_img=np.array(resized_img)
      np_img = np_img / 255.0 * 2 - 1
      np_img = np.reshape(np_img, (1, 448, 448, 3))
      np_predict = sess.run(predicts, feed_dict={image: np_img})

      xmin, ymin, xmax, ymax, class_num,prob= process_predicts(np_predict)
      xmin=max(xmin,0)
      ymin=max(ymin,0)
      xmax=max(xmax,0)
      ymax=max(ymax,0)

      xmin=min(xmin,448)
      ymin=min(ymin,448)
      xmax=min(xmax,448)
      ymax=min(ymax,448)

      box = (int(xmin), int(ymin), int(xmax), int(ymax))
      class_name = classes_name[class_num]
      #print(class_name,prob)
      if class_name=="person" and prob>0.1:
        img2=resized_img.crop(box).resize(original_shape, Image.ANTIALIAS)
        img2.save(os.path.join("img_out",file))
        recongnized_cnt+=1
      else:
        img.save(os.path.join("img_out",file))
      count+=1
      if count%100==0:
        print(count,"processed, with",recongnized_cnt,"augmented")



sess.close()
# exit(0)
# for p in pic:
#   np_img = cv2.imread(p+".jpg")
#   resized_img = cv2.resize(np_img, (448, 448))
#   np_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
#
#
#   np_img = np_img.astype(np.float32)
#
#   np_img = np_img / 255.0 * 2 - 1
#   np_img = np.reshape(np_img, (1, 448, 448, 3))
#
#   saver = tf.train.Saver(net.trainable_collection)
#
#   saver.restore(sess, 'models/pretrain/yolo_tiny.ckpt')
#
#   np_predict = sess.run(predicts, feed_dict={image: np_img})
#
#   xmin, ymin, xmax, ymax, class_num = process_predicts(np_predict)
#   print(xmin, ymin, xmax, ymax, class_num)
#   class_name = classes_name[class_num]
#   cv2.rectangle(resized_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255))
#   cv2.putText(resized_img, class_name, (int(xmin), int(ymin)), 2, 1.5, (0, 0, 255))
#   cv2.imwrite(p+"_out.jpg", resized_img)
# sess.close()
