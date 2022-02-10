import tensorflow as tf
from nets import nets_factory
from preprocessing import preprocessing_factory
import numpy as np
import json
import cv2
from flask import Flask,request
app = Flask(__name__)

# 标签字典
labels_map = {'0': '8', '1': '3', '2': '2', '3': '0', '4': '7', '5': '4', '6': '5', '7': '6', '8': '1', '9': 'NC', '10': '9'}
json_labels = json.dumps(labels_map)
labels = json.loads(json_labels)

# 图像输入大小
test_image_size = 224

# 网络配置
network_fn = nets_factory.get_network_fn(
    "resnet_v1_50",
    num_classes=11,
    is_training=False)

# 图像预处理配置
image_preprocessing_fn = preprocessing_factory.get_preprocessing(
    "resnet_v1_50",
    is_training=False)

# 权重路径
checkpoint_path = 'resnet_v1_0210/model.ckpt-200000'

# 初始化网络
tensor_input = tf.placeholder(tf.float32, [None, test_image_size, test_image_size, 3])
logits, _ = network_fn(tensor_input)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

def image_index(image):

    with graph.as_default():
        images = list()
        image = tf.image.decode_jpeg(image, channels=3)
        processed_image = image_preprocessing_fn(image, test_image_size, test_image_size)
        processed_image = sess.run(processed_image)
        images.append(processed_image)
        images = np.array(images)
        predictions = sess.run(logits, feed_dict={tensor_input: images})
        max_index = np.argmax(predictions)
        return labels[str(max_index)]

@app.route('/recognize',methods=['POST'])
def recognize_img():
     # 获取到post请求传来的file里的image文件
    image = request.files['image'].read()

    if image is None:
        return "nothing found"

    npimg = np.fromstring(image, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    tmp_jpg_bin = np.array(cv2.imencode('.jpg', img)[1]).tobytes()
    return str(image_index(tmp_jpg_bin))


if __name__ == '__main__':
    # 创建全局会话
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_path)
    graph = tf.get_default_graph()
    # 运行flask
    app.run(host='0.0.0.0', port=5000, debug=True)