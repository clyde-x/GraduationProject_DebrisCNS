import numpy as np 
import pandas as pd
import cv2
from scipy.spatial.transform import Rotation as R

class DebrisInImage():#记录在图像中的碎片和星，包含坐标星等和灰度
  def __init__(self,name,time,u,v,x,y,z,magnitude,category):
    self.name = name
    self.time = time
    self.u = u
    self.v = v
    self.x = x
    self.y = y
    self.z = z
    self.magnitude = magnitude
    self.category = category
    max_magnitude_threshold = 7 #最大星等，也就是最暗的
    #self.gray = 255*2.512**(1-self.magnitude) #0黑 255白  ？
    self.gray = 255-255*2.512**(self.magnitude-7) #？
    #self.gray = 50+10*(6-self.magnitude)  #？
    self.gray = int(self.gray)
    self.gray = 255 if self.gray > 255 else self.gray
    self.gray = 0 if self.gray < 0 else self.gray


class SpaceImage():#绘制每个时刻的图像
  def __init__(self,time,sensor,f,dh,dv,H):
    self.time = time
    self.f = f
    self.dh = dh
    self.dv = dv
    self.H = H
    self.visible_debris=[]
    self.visible_star = []
    self.sensor = sensor
    self.Msi = self.rotate() #旋转矩阵
    

  def rotate(self):
    sensor=self.sensor
    Csb = sensor.Csb
    q = sensor.quaternions
    if q[3] < 0:
      q = -q
    q0, q1, q2, q3 = q
    Msi = np.array([[q3**2+q0**2-q1**2-q2**2, 2*q3*q2+2*q0*q1,     -2*q3*q1+2*q0*q2],
                    [-2*q3*q2+2*q0*q1,    q3**2-q0**2+q1**2-q2**2, 2*q3*q0+2*q1*q2],
                    [2*q3*q1+2*q0*q2,     -2*q3*q0+2*q1*q2,    q3**2-q0**2-q1**2+q2**2]])
    return Csb@Msi

  def log_debris(self,debris0,debris_name):
    #记录视场中的碎片坐标
    sensor_vec = self.sensor.position
    debris_vec = debris0.position
    [x,y,z] = debris0.position
    vec = sensor_vec-debris_vec
    distance = np.linalg.norm(vec)
    vec = vec/np.linalg.norm(vec) #单位矢量
    magnitude = debris0.magnitude-3 # 做了点弊，把所有碎片的星等-3，否则啥也看不到
    magnitude = magnitude -30 + 10*np.log10(distance)#碎片星等变化
    if magnitude > 7: #观测不到
      return
    Msi = self.Msi
    deg = 2/3600
    error_rotation = R.from_euler('xyz', [deg, deg, deg], degrees=True) # 这里第二个参数应该是np.ramdom.randn(3)*deg，而不是一个固定值
    Msi = Msi @ error_rotation.as_matrix() # 星敏感器的误差，2角秒，体现在从惯性系到本体系/星敏感器坐标系的坐标变化矩阵的不准确
    A = np.dot(Msi,vec)
    u = self.f*A[0]/(self.dh*A[2])
    v = self.f*A[1]/(self.dv*A[2])
    # print('u:',self.u," v:",self.v, 'magnitude:',magnitude)
    if u > self.H or u < -self.H or v > self.H or v < -self.H: # 超出视场范围了，正常不会出现这样的情况
      print('error',self.time)
      # print(self.u,self.v)
    debris_temp = DebrisInImage(debris_name,self.time,u,v,x,y,z,magnitude,'debris') 
    self.visible_debris.append(debris_temp)

  def log_star(self,star,star_name):#同上
    # 将位置矢量近似为方向矢量，
    vec = star.position
    [x,y,z] = vec
    Msi = self.Msi
    deg = 2/3600
    error_rotation = R.from_euler('xyz', [deg, deg, deg], degrees=True)
    Msi = Msi @ error_rotation.as_matrix()
    A = np.dot(Msi,vec)
    u = self.f*A[0]/(self.dh*A[2])
    v = self.f*A[1]/(self.dv*A[2])

    if u > self.H or u < -self.H or v > self.H or v < -self.H:
      print('error',self.time)
      # print(self.u,self.v)
    star_temp = DebrisInImage(star_name,self.time,u,v,x,y,z, star.magnitude,'star')
    self.visible_star.append(star_temp)

  def diffuse_img(self, img, u, v, gray):
    sigma = 0.5  # 高斯核标准差
    radius = 2    # 对应5x5矩阵半径
    
    # 计算亚像素偏移后的中心坐标（相对于5x5矩阵）
    u_center = (u - int(u)) + radius
    v_center = (v - int(v)) + radius

    # 向量化生成高斯分布
    i, j = np.indices((5, 5))
    exponent = -((i - u_center)**2 + (j - v_center)**2) / (2 * sigma**2)
    diffuse = np.clip(gray * np.exp(exponent), 0, 255).astype(np.uint8)

    # 计算图像有效区域
    u_int, v_int = int(u), int(v)
    start_u, end_u = max(0, u_int - radius), min(img.shape[0], u_int + radius + 1)
    start_v, end_v = max(0, v_int - radius), min(img.shape[1], v_int + radius + 1)

    # 计算扩散矩阵的对应区域（自动处理边界裁剪）
    d_start_u = max(0, start_u - (u_int - radius))
    d_end_u = min(5, d_start_u + (end_u - start_u))
    d_start_v = max(0, start_v - (v_int - radius))
    d_end_v = min(5, d_start_v + (end_v - start_v))

    # 仅在有效区域进行赋值
    if d_start_u < d_end_u and d_start_v < d_end_v:
        img[start_u:end_u, start_v:end_v] = diffuse[d_start_u:d_end_u, d_start_v:d_end_v]
    
    return img

  def add_noise(self, img): #对星图添加噪声
    noise = np.random.normal(10,10,(self.H*2,self.H*2))
    img = img + noise
    img_clipped = np.clip(img, 0, 255).astype(np.uint8)
    return img_clipped

  def plot_image(self,time_list, ifplot=False, ifsave=False, iflabel=False):
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


def images_sequence(start_index,end_index,time_list, sensor, debris, star, access, debris_name_list, star_name_list,f, dh, dv,H):# 生成一系列的图像，放在images中
  images = []
  for curr_time in time_list[start_index:end_index]:
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

    space_image = SpaceImage(curr_time, sensor[time_index], f, dh, dv,H)

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

def log_img_data(images,img_log_file, iflabel, ifsave, ifplot):
  log_data = pd.DataFrame(columns=['time','debris_num','star_num','target','x','y','z','u','v','magnitude','category'])
  temp_log=[]
  for image in images:
    print(image.time)
    # image.plot_image(time_list, iflabel=iflabel, ifsave=ifsave, ifplot=ifplot)
    visible_debris = [_.name for _ in image.visible_debris]
    visible_star = [_.name for _ in image.visible_star]
    for _ in image.visible_debris:
      ss_temp = {'time':image.time,'debris_num':len(image.visible_debris),'star_num':len(image.visible_star),'target':_.name,'x':_.x,'y':_.y,'z':_.z,'u':_.u,'v':_.v,'magnitude':_.magnitude,'category':_.category}
      temp_log.append(ss_temp)
    log_data = pd.concat([log_data,pd.DataFrame(temp_log)],ignore_index=True)
    temp_log = []
    for _ in image.visible_star:
      ss_temp = {'time':image.time,'debris_num':len(image.visible_debris),'star_num':len(image.visible_star),'target':_.name,'x':_.x,'y':_.y,'z':_.z,'u':_.u,'v':_.v,'magnitude':_.magnitude,'category':_.category}
      temp_log.append(ss_temp)
    log_data = pd.concat([log_data,pd.DataFrame(temp_log)],ignore_index=True)
  log_data.to_csv(img_log_file,index=False)