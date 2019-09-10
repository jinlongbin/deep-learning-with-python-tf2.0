import tensorflow as tf   
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.7)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
tf.disable_v2_behavior()

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import imutils
import cv2

K.clear_session()
model = VGG16(weights='imagenet')

camera = cv2.VideoCapture(0)
while True:
    # === preprocess ===
    frame = camera.read()[1]
    #frame = imutils.resize(frame,width=500)
    frameClone = frame.copy()
    frame1 = cv2.resize(frame, (224,224))
    img = np.expand_dims(frame1, axis=0)
    img = preprocess_input(img)

    # === predict ===
    preds = model.predict(img)

    # === Grad-CAM ===
    dog_output = model.output[:,np.argmax(preds[0])]
    last_conv_layer = model.get_layer('block5_conv3')
    grads = K.gradients(dog_output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0,1,2))
    iterate = K.function([model.input],[pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([img])
    for i in range(512):
        conv_layer_output_value[:,:,i] *= pooled_grads_value[i]
    
    # === heatmap ===
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    frame = np.expand_dims(frame, axis=0)
    superimposed_img = heatmap * 0.4 + frame
    output = superimposed_img[0]/255.
    # print(output.shape)

    #print('Predicted: ', decode_predictions(preds, top=3)[0])
    label = [decode_predictions(preds, top=3)[0][i][1] for i in range(3)]
    cv2.putText(output, label[0], (1, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(output, label[1], (1, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(output, label[2], (1, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('heatmap', output)
    # cv2.imshow('your_face', frameClone)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
camera.release()
cv2.destroyAllWindows()