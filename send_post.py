
import requests

# 请求地址
url = 'http://127.0.0.1:5000/recognize'
# 向files里传入图片
files={'image':open("001P1.png","rb")}
# 发送post请求
r=requests.post(url,files=files)
print(r.text)
