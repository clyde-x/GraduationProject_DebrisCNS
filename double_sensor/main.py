import numpy as np 
import pandas as pd 
import os 
import data_processing
import star_map_sim
import multi_cns


#CONST 
f = 0.01413
dh = 0.000006
dv = 0.000006
H = 700

# 设置文件根路径
origin_data_path = 'f:\\buaa\\python\\final_proj\\double_sensor\\new_data' 

# debis data: 从STK satellite导出的，包括
# time, x, y, z. (xyz 是ICRF系下的笛卡尔坐标)
debris_data_path = os.path.join(origin_data_path, 'debris_new.csv')

# sensor data: 从STK sensor导出，包括
# time, x, y, z, vx, vy, vz, q1, q2, q3, q4, wx, wy, wz
# sensor access data: 从 STK access导出, 包括:
# csv和txt两种格式的。需要看txt文件中有用的部分从那一行开始，并在'data_processing.read_reduced_data'这个函数里修改参数（写这个main的时候忘了传参了，懒得改了）
sensor1_data_path = os.path.join(origin_data_path, 'sensor1_data_output.csv')
sensor2_data_path = os.path.join(origin_data_path, 'sensor2_data_output.csv')
sensor1_access_txt = os.path.join(origin_data_path, 'sensor1_access.txt')
sensor2_access_txt = os.path.join(origin_data_path, 'sensor2_access.txt')
sensor1_access_csv = os.path.join(origin_data_path, 'sensor1_access.csv')
sensor2_access_csv = os.path.join(origin_data_path, 'sensor2_access.csv')

# star data: 从STK star导出, 包括:
# time, x, y, z, right_ascension, declination
# magnitude data: 从STK star导出, 包括:
# star, magnitude
star_data_path = os.path.join(origin_data_path, 'Star.csv')
magnitude_data_path = os.path.join(origin_data_path, 'star_magnitude.csv')

# 以下文件路径用于储存data_processing.data_washing进行简化之后的数据，以减少文件体积和运行时间 
reduced_sensor1_debris_path = os.path.join(origin_data_path, 'reduced_sensor1_debris.csv')
reduced_sensor2_debris_path = os.path.join(origin_data_path, 'reduced_sensor2_debris.csv')
reduced_sensor1_star_path = os.path.join(origin_data_path, 'reduced_sensor1_star.csv')
reduced_sensor2_star_path = os.path.join(origin_data_path, 'reduced_sensor2_star.csv')
reduced_sensor1_magnitude_path = os.path.join(origin_data_path, 'reduced_sensor1_magnitude.csv')
reduced_sensor2_magnitude_path = os.path.join(origin_data_path, 'reduced_sensor2_magnitude.csv')

# image log data: 储存每一帧的像点坐标和对应星/碎片的空间向量/坐标
img1_log_path = os.path.join(origin_data_path, 'img_log1.csv')
img2_log_path = os.path.join(origin_data_path, 'img_log2.csv')

# 天文定位结果，矩阵和四元数涉及两个星敏的安装矢量（其实是我凑出来的，我也不知道怎么算，坐标系有点混乱）
cns_Q = os.path.join(origin_data_path, 'cns_Q.csv')
cns_pos = os.path.join(origin_data_path, 'cns_pos.csv')
Cs1b = np.eye(3)
Cs2b = np.array([[0,1,0],[-1,0,0],[0,0,1]])
deltaq1 = np.array([0,0.701,0,0.701])
deltaq2 = np.array([-0.5,0.5,0.5,0.5])

if __name__ == "__main__":

  # data_processing，简化碎片编目库，保留能观测到的碎片，只运行一次。
  data_processing.data_washing(sensor1_access_txt, sensor1_access_csv, sensor1_data_path, debris_data_path, star_data_path, magnitude_data_path, reduced_sensor1_debris_path, reduced_sensor1_star_path, reduced_sensor1_magnitude_path)
  data_processing.data_washing(sensor2_access_txt, sensor2_access_csv, sensor2_data_path, debris_data_path, star_data_path, magnitude_data_path, reduced_sensor2_debris_path, reduced_sensor2_star_path, reduced_sensor2_magnitude_path)
  print('sensor data_washing done, ')

  # 读取简化的编目库
  sensor1, time_list1, debris1, debris_name_list1, star1, star_name_list1, access1 = data_processing.read_reduced_data(sensor1_data_path, Cs1b, reduced_sensor1_debris_path, reduced_sensor1_star_path, reduced_sensor1_magnitude_path, sensor1_access_csv, sensor1_access_txt)
  sensor2, time_list2, debris2, debris_name_list2, star2, star_name_list2, access2 = data_processing.read_reduced_data(sensor2_data_path, Cs2b, reduced_sensor2_debris_path, reduced_sensor2_star_path, reduced_sensor2_magnitude_path, sensor2_access_csv, sensor2_access_txt)
  print('sensor data read done, ')

  # 生成图像数据，images1是一个列表，元素是自定义的SpaceImage类的实例
  images1 = star_map_sim.images_sequence(0,len(time_list1)-1, time_list1, sensor1, debris1, star1, access1, debris_name_list1, star_name_list1, f, dh, dv, H)
  images2 = star_map_sim.images_sequence(0,len(time_list2)-1, time_list2, sensor2, debris2, star2, access2, debris_name_list2, star_name_list2, f, dh, dv, H)
  print('images sequence done, ')
  
  # 根据数据生成图像，每个三个if——参数分别是是否给每个像点打上标签（用于展示/验证），是否保存图像（确保硬盘有足够空间），是否在运行时展示图像（用于调试）
  star_map_sim.log_img_data(images1, img1_log_path, iflabel=False, ifsave=True, ifplot=False)
  star_map_sim.log_img_data(images2, img2_log_path, iflabel=False, ifsave=True, ifplot=False)
  print('image done')


  # 按理说这里还要有图像处理、碎片辨别、匹配等内容，但这个程序中我没写，可以参考另一个程序
  # 为什么不写呢，因为高斯法质心提取很慢，精度也不是很满意；而且星图匹配程序是有点问题的
  # 但是要注意给img_log_path储存的数据里面的u，v添加一定的噪声，（mu=0，sigma=0.15）

  # 天文定位
  Q, pos = multi_cns.cns_out([img1_log_path, img2_log_path], [Cs1b, Cs2b], [deltaq1,deltaq2],start=0,end=-1)
  Q.to_csv(cns_Q, index=False)
  pos.to_csv(cns_pos, index=False)
  print('cns done')

  # 事后补救的统计碎片数量的，可以修改上一步的函数，一起做了
  n_debris = multi_cns.cns_count_debris([img1_log_path, img2_log_path],start=0,end=-1)
  np.save(os.path.join(origin_data_path, 'cns_n_debris.npy'), n_debris)