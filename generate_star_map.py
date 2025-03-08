import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from dataclasses import dataclass

current_dir = 'f:/buaa/python/final_proj'
os.chdir(current_dir)
sensor_file = 'data/mysenior_sensor_data_output.csv'
debris_file = 'data/0_ALEXIS_22638_debris_new.csv'
star_file = 'data/Star-100027_star.csv'
star_magnitude_file = 'data/Star-100027_magnitude.csv'
access_file_csv = 'data/Satellite-observe-Sensor-mysenior-To-Satellite-0_ALEXIS_22638_Access.csv'
access_file_txt = 'data/Satellite-observe-Sensor-mysenior-To-Satellite-0_ALEXIS_22638_Access.txt'
f = 0.01413
dh = 0.000006
dv = 0.000006
H = 700

class SpaceObject():
  def __init__(self, category, name='unknow'):
    self.name = name
    self.category = category # 'sensor', 'debris', 'star'

class Sensor(SpaceObject):
  def __init__(self, name, time, x, y, z, vx, vy, vz, q1, q2, q3, q4, wx, wy, wz, category='sensor'):
    super().__init__(category, name)
    self.time = pd.to_datetime(time)
    self.position = np.array([x, y, z])
    self.velocity = np.array([vx, vy, vz])
    self.quaternions = np.array([q1, q2, q3, q4])
    self.angular_velocity = np.array([wx, wy, wz])

class Star(SpaceObject):
  def __init__(self, name, magnitude, x, y, z, right_ascension, declination, category='star'):
    super().__init__(category, name)
    self.name = name
    self.magnitude = float(magnitude)
    self.position = np.array([x, y, z]).astype(float)
    self.as_dec = np.array([right_ascension, declination]).astype(float)

@dataclass()
class Debris:
  name: str
  time: pd.Timestamp
  magnitude: float
  position: np.ndarray

def read_sensor_data(sensor_file):
  sensor_data = pd.read_csv(sensor_file)
  time_list = pd.to_datetime(sensor_data['Time (UTCG)']).to_list()
  sensor=[]
  for i in range(len(sensor_data)):
    name = 'mysensor'
  for sensor_ in sensor_data.iterrows():
    sensor_temp = Sensor(name, sensor_[1]['Time (UTCG)'],sensor_[1]['x (km)'],sensor_[1]['y (km)'],sensor_[1]['z (km)'],sensor_[1]['Velocity x (km/sec)'],sensor_[1]['Velocity y (km/sec)'],sensor_[1]['Velocity z (km/sec)'],sensor_[1]['q1'],sensor_[1]['q2'],sensor_[1]['q3'],sensor_[1]['q4'],sensor_[1]['wx (deg/sec)'],sensor_[1]['wy (deg/sec)'],sensor_[1]['wz (deg/sec)'])
    sensor.append(sensor_temp)
  return sensor, time_list

def read_debris_data(debris_file):
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

def read_star_data(star_file,star_magnitude_file):
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

def read_access_data(access_file_csv,access_file_txt,skiprows=29):
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
    max_magnitude_threshold = 6 #最大星等，也就是最暗的
    #self.gray = 255*2.512**(1-self.magnitude) #0黑 255白  ？
    self.gray = 255-255*2.512**(self.magnitude-6) #？
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
    self.visible_debris=[]
    self.visible_star = []

  def rotate(self,sensor):
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
    sensor_vec = self.sensor.position
    debris_vec = self.debris.position
    vec = sensor_vec-debris_vec
    distance = np.linalg.norm(vec)
    vec = vec/np.linalg.norm(vec)
    A = np.dot(self.Msi,vec)
    self.u = self.f*A[0]/(self.dh*A[2])
    self.v = self.f*A[1]/(self.dv*A[2])
    self.u, self.v = int(self.u), int(self.v)
    magnitude = self.debris.magnitude-6
    magnitude = magnitude -30 + 10*np.log10(distance)
    # print('u:',self.u," v:",self.v, 'magnitude:',magnitude)
    if self.u > self.H or self.u < -self.H or self.v > self.H or self.v < -self.H:
      print(self.u,self.v)
    debris_temp = DebrisInImage(debris_name,self.time,self.u,self.v,magnitude,'debris')
    self.visible_debris.append(debris_temp)

  def log_star(self,star,star_name):
    # 将位置矢量近似为方向矢量
    vec = star.position
    A = np.dot(self.Msi,vec)
    self.u = self.f*A[0]/(self.dh*A[2])
    self.v = self.f*A[1]/(self.dv*A[2])
    self.u, self.v = int(self.u), int(self.v)
    if self.u > self.H or self.u < -self.H or self.v > self.H or self.v < -self.H:
      print(self.u,self.v)
    star_temp = DebrisInImage(star_name,self.time,self.u,self.v, star.magnitude,'star')
    self.visible_star.append(star_temp)

  def diffuse_img(self, img, u, v, gray):
    diffuse = np.zeros((9,9),dtype=np.uint8)
    for i in range(9):
      for j in range(9):
        diffuse[i,j] = int(gray*np.exp(-0.5*((i-4)**2+(j-4)**2)/2))
    img_h, img_w = img.shape
    radius = 4  # 9x9矩阵的半径
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
    noise = np.random.normal(10,10,(self.H*2,self.H*2))
    img = img + noise
    img_clipped = np.clip(img, 0, 255).astype(np.uint8)
    return img_clipped

  def plot_image(self,ifplot=False, ifsave=False, iflabel=True):
    # 662*662 焦距f=0.01413m  0.000006m/pixel, alpha = 16.05deg = 2arctan(H/2f), -->H=662
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

images = images_sequence(1,1000)
log_data = pd.DataFrame(columns=['time','debris_num','star_num','target','u','v'])
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
log_data.to_csv('log_data.csv',index=False)

