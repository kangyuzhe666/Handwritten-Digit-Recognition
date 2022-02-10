# Handwritten-Digit-Recognition
大分辨率彩色手写数字识别，采用flask进行部署

## 环境
必要环境 tensoeflow==1.14.0
	opencv
	numpy
	json
	flask
	request
	opencv

## 部署

运行 recognize_img.py 会在本机的5000端口提供一个post服务具体url：ip地址:5000/recognize，向这个url发送照片的post请求就会返回照片识别的结果。
python发送post请求例程见send_post.py