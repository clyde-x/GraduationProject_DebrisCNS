'''
creat: 20250316
update: 20250316
author: 黄乙笑
'''

#导入必要的包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from dataclasses import dataclass

current_dir = 'f:/buaa/python/final_proj' 
os.chdir(current_dir) #设置工作目录
# 各个文件路径
sensor_file = 'data/mysenior_sensor_data_output.csv' # 星敏数据，包含时间、惯性系位置、速度、姿态四元数、角速度等
debris_file = 'data/0_ALEXIS_22638_debris_new.csv' # 碎片数据，包含时间、惯性系中位置等
star_file = 'data/Star-100027_star.csv' # 恒星数据。包含恒星的赤经赤纬以及在惯性系中位置
star_magnitude_file = 'data/Star-100027_magnitude.csv' # 恒星的星等数据
access_file_csv = 'data/Satellite-observe-Sensor-mysenior-To-Satellite-0_ALEXIS_22638_Access.csv' # stk导出的access信息，包含星敏能观测到的恒星和碎片的时间段（没有这个也可以自己算）
access_file_txt = 'data/Satellite-observe-Sensor-mysenior-To-Satellite-0_ALEXIS_22638_Access.txt'# 同上


f = 0.01413 # 焦距
dh = 0.000006 #水平单位像素的实际长度 0.000006m/pixel
dv = 0.000006 #同上
H = 700 #alpha = 16.05deg = 2arctan(H/2f), -->H=662，这里取700是简便

class SpaceObject():# 创建空间目标父类
  def __init__(self, category, name='unknow'):
    self.name = name
    self.category = category # 'sensor', 'debris', 'star'

class Sensor(SpaceObject): # 继承自SpaceObject，将每个时刻的星敏数据记录到每个Sensor对象中
  def __init__(self, name, time, x, y, z, vx, vy, vz, q1, q2, q3, q4, wx, wy, wz, category='sensor'):
    super().__init__(category, name)
    self.time = pd.to_datetime(time)
    self.position = np.array([x, y, z])
    self.velocity = np.array([vx, vy, vz])
    self.quaternions = np.array([q1, q2, q3, q4])
    self.angular_velocity = np.array([wx, wy, wz])

class Star(SpaceObject): # 继承自SpaceObject，将每个时刻的恒星数据记录到每个Star对象中
  def __init__(self, name, magnitude, x, y, z, right_ascension, declination, category='star'):
    super().__init__(category, name)
    self.name = name
    self.magnitude = float(magnitude)
    self.position = np.array([x, y, z]).astype(float)
    self.as_dec = np.array([right_ascension, declination]).astype(float)

@dataclass() # 用dataclass装饰器定义碎片类，记录每个时刻碎片的名字、时间、大小、位置
class Debris:
  name: str
  time: pd.Timestamp
  magnitude: float
  position: np.ndarray

def read_sensor_data(sensor_file): #读取星敏数据，返回Sensor对象列表（每个元素是一个Sensor对象）和时间列表（每个元素是一个时间戳）
  sensor_data = pd.read_csv(sensor_file)
  time_list = pd.to_datetime(sensor_data['Time (UTCG)']).to_list()
  sensor=[]
  name = 'mysensor'
  for sensor_ in sensor_data.iterrows():
    sensor_temp = Sensor(name, sensor_[1]['Time (UTCG)'],sensor_[1]['x (km)'],sensor_[1]['y (km)'],sensor_[1]['z (km)'],sensor_[1]['Velocity x (km/sec)'],sensor_[1]['Velocity y (km/sec)'],sensor_[1]['Velocity z (km/sec)'],sensor_[1]['q1'],sensor_[1]['q2'],sensor_[1]['q3'],sensor_[1]['q4'],sensor_[1]['wx (deg/sec)'],sensor_[1]['wy (deg/sec)'],sensor_[1]['wz (deg/sec)'])
    sensor.append(sensor_temp)
  return sensor, time_list

def read_debris_data(debris_file): #读取碎片数据，返回Debris对象二维列表（行是每个碎片，列是每个时间）和碎片名字列表
  debris_data = pd.read_csv(debris_file,header=None)
  is_header = debris_data.iloc[:,0].str.startswith('Time (UTCG)')
  group_ids = is_header.cumsum()-1
  group_count = group_ids.max() + 1
  magnitudes = np.random.normal(4, 0.5, group_count)
  headers = debris_data[is_header].iloc[:, 2]
  names = headers.str.rsplit('-', n=1, expand=True)[0].str.replace(' ', '')
  debris_name_list = names.unique().tolist()
  debris = []
  for grp_id, sub_df in debris_data.groupby(group_ids):
    print(grp_id,names.iloc[grp_id])
    coords = sub_df.iloc[1:, 1:4].to_numpy(dtype=np.float64)
    time = pd.to_datetime(sub_df.iloc[1:, 0])
    group_debris = [Debris(
        name = names.iloc[grp_id],
        time = time.iloc[i],
        magnitude = magnitudes[grp_id],
        position = coords[i]
      )for i in range(len(time))
    ]
    debris.append(group_debris)
  return debris, debris_name_list

def read_star_data(star_file,star_magnitude_file): #读取恒星数据，返回Star对象列表（每个元素是一个Star对象）和恒星名字列表
  star_data = pd.read_csv(star_file,header=None)
  magnitude_data  = pd.read_csv(star_magnitude_file,header=None)
  magnitude = [eval(magnitude_data.iloc[i][0]) for i in range(1,len(magnitude_data),2)]
  star = []
  star_name_list = []
  for _ in star_data.iterrows():
    if _[0]%3 == 0:
      name = _[1][1].rsplit('-',maxsplit=1)[0].replace(' ','')
      star_name_list.append(name)
    if _[0]%3 == 1:
      star_temp = Star(name, magnitude[_[0]//3], _[1][1], _[1][2], _[1][3], _[1][4], _[1][5])
      star.append(star_temp)
  return star, star_name_list

def read_access_data(access_file_csv,access_file_txt,skiprows=29): #读取access数据，返回access数据的DataFrame，行是碎片，列是能观测到的碎片的开始和截止时间戳（由于仿真时间段，所以一般只能观测到一次）；skiprows是txt文件的前几行是无用的
  access = pd.read_csv(access_file_csv)
  access_df = pd.read_csv(access_file_txt,skiprows=skiprows,header=None)
  access_ls = []
  access = access.drop(list(range(1,len(access),2)))
  access.reset_index(drop=True,inplace=True)

  for i in range(len(access_df)):
    if access_df.iloc[i,0].replace(' ','').startswith('Access'):
      access_ls.append(access_df.iloc[i-2,0].split('-',maxsplit=2)[-1])
  access['name']=access_ls
  access.set_index('name',inplace=True)
  access['Start Time (UTCG)'] = pd.to_datetime(access['Start Time (UTCG)'])
  access['Stop Time (UTCG)'] = pd.to_datetime(access['Stop Time (UTCG)'])
  return access

sensor, time_list = read_sensor_data(sensor_file)
debris, debris_name_list = read_debris_data(debris_file)
star, star_name_list = read_star_data(star_file,star_magnitude_file)
access = read_access_data(access_file_csv,access_file_txt,29)


class DebrisInImage():#记录在图像中的碎片和星，包含坐标星等和灰度
  def __init__(self,name,time,u,v,magnitude,category):
    self.name = name
    self.time = time
    self.u = u
    self.v = v
    self.magnitude = magnitude
    self.category = category
    # 关于星等和灰度转换，缪按有这几个流行的公式，测试出来第二个公式最好，满足条件1最大的星等最暗，灰度值最低 2星等与灰度是指数关系
    max_magnitude_threshold = 6 #最大星等，也就是最暗的
    #self.gray = 255*2.512**(1-self.magnitude) #0黑 255白  ？
    self.gray = 255-255*2.512**(self.magnitude-6) 
    #self.gray = 50+10*(6-self.magnitude)  #？
    self.gray = int(self.gray)
    self.gray = 255 if self.gray > 255 else self.gray
    self.gray = 0 if self.gray < 0 else self.gray


class SpaceImage():#绘制每个时刻的图像
  def __init__(self,time,f,dh,dv,H):
    self.time = time
    self.f = f
    self.dh = dh
    self.dv = dv
    self.H = H
    self.visible_debris=[] #当前时刻中能观测到的碎片
    self.visible_star = [] #当前时刻中能观测到的恒星

  def rotate(self,sensor):# 从当前星敏数据中获取四元数，计算旋转矩阵（从惯性系到相机系）
    self.sensor = sensor
    q = sensor.quaternions
    if q[3] < 0:
      q = -q
    q0, q1, q2, q3 = q
    print(q0, q1, q2, q3)
    Msi = np.array([[q3**2+q0**2-q1**2-q2**2, 2*q3*q2+2*q0*q1,     -2*q3*q1+2*q0*q2],
                    [-2*q3*q2+2*q0*q1,    q3**2-q0**2+q1**2-q2**2, 2*q3*q0+2*q1*q2],
                    [2*q3*q1+2*q0*q2,     -2*q3*q0+2*q1*q2,    q3**2-q0**2-q1**2+q2**2]])
    self.Msi = Msi
    return self.Msi

  def log_debris(self,debris0,debris_name):
    self.debris = debris0
    sensor_vec = self.sensor.position #相机在惯性系位置
    debris_vec = self.debris.position #碎片在惯性系位置
    vec = sensor_vec-debris_vec #相机到碎片的矢量
    distance = np.linalg.norm(vec) #相机到碎片的距离
    vec = vec/np.linalg.norm(vec) #单位化矢量
    A = np.dot(self.Msi,vec) # 星敏成像的几何方程。将方向矢量转为星敏坐标
    self.u = self.f*A[0]/(self.dh*A[2]) #星敏坐标转为像素坐标
    self.v = self.f*A[1]/(self.dv*A[2]) #同上
    self.u_int, self.v_int = int(self.u), int(self.v) #取整
    # 计算星等，先初始化星等在4附近（距离1000km处得出的统计规律），再根据实际距离计算星敏观测的视星等
    magnitude = self.debris.magnitude-6 #其实这里只应该减3，但是这样的话星等太大了，成像点很暗。所以减6
    magnitude = magnitude -30 + 10*np.log10(distance)
    # print('u:',self.u," v:",self.v, 'magnitude:',magnitude)
    if self.u > self.H or self.u < -self.H or self.v > self.H or self.v < -self.H: #判断成像点是否超出成像范围，虽然一般不会超出
      print(self.u,self.v)
    debris_temp = DebrisInImage(debris_name,self.time,self.u,self.v,magnitude,'debris')
    self.visible_debris.append(debris_temp)

  def log_star(self,star,star_name):#同上
    # 将位置矢量近似为方向矢量
    vec = star.position
    A = np.dot(self.Msi,vec)
    self.u = self.f*A[0]/(self.dh*A[2])
    self.v = self.f*A[1]/(self.dv*A[2])
    self.u_int, self.v_int = int(self.u), int(self.v)
    self.u, self.v = int(self.u), int(self.v)
    if self.u > self.H or self.u < -self.H or self.v > self.H or self.v < -self.H:
      print(self.u,self.v)
    star_temp = DebrisInImage(star_name,self.time,self.u,self.v, star.magnitude,'star')
    self.visible_star.append(star_temp)

  def diffuse_img(self, img, u, v, gray): # 恒星或者碎片的点扩散模型，得到一个9*9的二维高斯分布矩阵，然后叠加到图像上
    #为了实现亚像素级别的模拟，根据uv小数点的部分确定他们再27*27中的位置，再将27*27的缩小为9*9
    diffuse = np.zeros((27,27),dtype=np.uint8)
    for i in range(27):
      for j in range(27):
        diffuse[i,j] = int(gray*np.exp(-0.5*((i-13+u*3-int(u)*3)**2+(j-13+v*3-int(v)*3)**2)/2))
    diffuse = np.clip(diffuse, 0, 255).astype(np.uint8)
    diffuse = cv2.resize(diffuse, (9, 9), interpolation=cv2.INTER_AREA)
    img_h, img_w = img.shape
    radius = 4  # 9x9矩阵的半径
    u = int(u)
    v = int(v)
    start_u = max(0, u - radius)
    end_u = min(img_h, u + radius + 1)
    start_v = max(0, v - radius)
    end_v = min(img_w, v + radius + 1)

    d_start_u = radius - (u - start_u) if u < radius else 0
    d_end_u = d_start_u + (end_u - start_u)
    d_start_v = radius - (v - start_v) if v < radius else 0
    d_end_v = d_start_v + (end_v - start_v)

    img[start_u:end_u, start_v:end_v] = diffuse[d_start_u:d_end_u, d_start_v:d_end_v]#叠加
    return img

  def add_noise(self, img):
    # 添加高斯噪声
    noise = np.random.normal(10,10,(self.H*2,self.H*2))
    img = img + noise
    img_clipped = np.clip(img, 0, 255).astype(np.uint8)
    return img_clipped

  def plot_image(self,ifplot=False, ifsave=False, iflabel=True):# 三个参数分别是：是否在程序运行时显示图像，是否保存图像，是否显示标签
    #生成图像
    visible_debris = self.visible_debris
    visible_star = self.visible_star
    img = np.zeros((self.H*2,self.H*2),dtype=np.uint8)
    for debris_ in visible_debris:
      img = SpaceImage.diffuse_img(self, img,debris_.u+self.H,debris_.v+self.H,debris_.gray)
      if iflabel:
        cv2.putText(img,debris_.name,(debris_.v+self.H+10,debris_.u+self.H+10),cv2.FONT_HERSHEY_SIMPLEX,0.2,(255,255,255),1)
    for star_ in visible_star:
      img = SpaceImage.diffuse_img(self, img,star_.u+self.H,star_.v+self.H,star_.gray)
      if iflabel:
        cv2.putText(img,star_.name,(star_.v+self.H+10,star_.u+self.H+10),cv2.FONT_HERSHEY_SIMPLEX,0.2,(255,255,255),1)
    img = SpaceImage.add_noise(self,img)
    if ifplot:
      cv2.imshow('image',img)
      cv2.waitKey(0)
    if ifsave:
      cv2.imwrite('image/'+"{:04d}".format(time_list.index(self.time))+'.png',img)
    return img
  
def images_sequence(star_index,end_index):
  # 在开始到结束的两个时间内，生成图像列表，images列表内每个元素是一个SpaceImage对象
  images = []
  for curr_time in time_list[star_index:end_index]:
    time_index = time_list.index(curr_time)
    visible_star = []
    visible_debris = []
    #统计可见的星和碎片
    for i in range(len(access)):
      if access.iloc[i]['Start Time (UTCG)'] <= curr_time and access.iloc[i]['Stop Time (UTCG)'] >= curr_time:
        if access.index[i] in star_name_list:
          visible_star.append(access.index[i])
        if access.index[i] in debris_name_list:
          visible_debris.append(access.index[i])

    space_image = SpaceImage(curr_time, f, dh, dv,H)
    Msi = space_image.rotate(sensor[time_index])

    if len(visible_debris) != 0:
      for debris_name in visible_debris:
        debris_index = debris_name_list.index(debris_name)
        space_image.log_debris(debris[debris_index][time_index],debris_name)
    if len(visible_star) != 0:
      for star_name in visible_star:
        star_index = star_name_list.index(star_name)
        space_image.log_star(star[star_index],star_name)
    images.append(space_image)
  return images

images = images_sequence(1,1000)#从第一帧到第1000帧生成图像
log_data = pd.DataFrame(columns=['time','debris_num','star_num','target','u','v'])
temp_log=[]
for image in images:
  image.plot_image(iflabel=False, ifsave=True)
  visible_debris = [_.name for _ in image.visible_debris]
  visible_star = [_.name for _ in image.visible_star]
  for _ in image.visible_debris:
    ss_temp = pd.Series({'time':image.time,'debris_num':len(image.visible_debris),'star_num':len(image.visible_star),'target':_.name,'u':_.u,'v':_.v})
    pd.concat([log_data,ss_temp],ignore_index=True)
  for _ in image.visible_star:
    ss_temp = pd.Series({'time':image.time,'debris_num':len(image.visible_debris),'star_num':len(image.visible_star),'target':_.name,'u':_.u,'v':_.v})
    pd.concat([log_data,ss_temp],ignore_index=True)
cv2.destroyAllWindows()
log_data.to_csv('log_data.csv',index=False)#这个csv记录了每个时刻碎片、恒星在星敏感器中的坐标和星等

