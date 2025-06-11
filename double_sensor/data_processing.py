import numpy as np 
import pandas as pd
import os
from dataclasses import dataclass

# this python file is to washing the data(only run for 1 time)
# and defines a Class for spaceObjects, including sensor, debris and star

class SpaceObject():
  def __init__(self, category, name='unknow'):
    self.name = name
    self.category = category # 'sensor', 'debris', 'star'

class Sensor(SpaceObject):
  def __init__(self, name, time, x, y, z, vx, vy, vz, q1, q2, q3, q4, wx, wy, wz, Csb, category='sensor'):
    super().__init__(category, name)
    self.time = pd.to_datetime(time)
    self.position = np.array([x, y, z])
    self.velocity = np.array([vx, vy, vz])
    self.quaternions = np.array([q1, q2, q3, q4])
    self.angular_velocity = np.array([wx, wy, wz])
    self.Csb = Csb

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
def read_sensor_data(sensor_file, Csb=np.eye(3)):
  sensor_data = pd.read_csv(sensor_file)
  time_list = pd.to_datetime(sensor_data['Time (UTCG)']).to_list()
  sensor=[]
  for i in range(len(sensor_data)):
    name = 'mysensor'
  for sensor_ in sensor_data.iterrows():
    sensor_temp = Sensor(name, sensor_[1]['Time (UTCG)'],sensor_[1]['x (km)'],sensor_[1]['y (km)'],sensor_[1]['z (km)'],sensor_[1]['Velocity x (km/sec)'],sensor_[1]['Velocity y (km/sec)'],sensor_[1]['Velocity z (km/sec)'],sensor_[1]['q1'],sensor_[1]['q2'],sensor_[1]['q3'],sensor_[1]['q4'],sensor_[1]['wx (deg/sec)'],sensor_[1]['wy (deg/sec)'],sensor_[1]['wz (deg/sec)'],Csb)
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
    coords = sub_df.iloc[1:, 1:4].to_numpy(dtype=np.float64)
    time = pd.to_datetime(sub_df.iloc[1:, 0])
    name0 = names.iloc[grp_id]
    if name0.endswith('1') and (name0[:-1] in debris_name_list):
      debris_name_list.remove(name0[:-1])
      continue
    # print(grp_id,names.iloc[grp_id])
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

def read_access_data(access_file_csv,access_file_txt,skiprows=23):
  access = pd.read_csv(access_file_csv)
  access_df = pd.read_csv(access_file_txt,skiprows=skiprows,header=None)
  access_ls = []
  access = access.drop(access[access['Access']=='Access'].index)
  access.reset_index(drop=True,inplace=True)

  for i in range(len(access_df)):
    if access_df.iloc[i,0].replace(' ','').startswith('Access'):
      access_ls.append(access_df.iloc[i-2,0].split('-',maxsplit=2)[-1])
  access['name']=access_ls
  access.set_index('name',inplace=True)
  access['Start Time (UTCG)'] = pd.to_datetime(access['Start Time (UTCG)'])
  access['Stop Time (UTCG)'] = pd.to_datetime(access['Stop Time (UTCG)'])
  return access

def data_washing(access_txt, access_csv, sensor_file, debris_file, star_file, magnitude_file, reduced_debris_file, reduced_star_file, reduced_magnitude_file):
  access = read_access_data(access_csv, access_txt)
  access_names = access.index.tolist()
  # print(access_names)
  sensor, time_list = read_sensor_data(sensor_file)

  debris_data = pd.read_csv(debris_file,header=None)
  is_header = debris_data.iloc[:,0].str.startswith('Time (UTCG)')
  group_ids = is_header.cumsum()-1
  group_count = group_ids.max() + 1
  magnitudes = np.random.normal(4, 0.5, group_count)
  headers = debris_data[is_header].iloc[:, 2]
  names = headers.str.rsplit('-', n=1, expand=True)[0].str.replace(' ', '')
  debris_name_list = names.unique().tolist()

  sensor_pos=[]
  for i in range(6):
    sensor_pos.append(sensor[i*1000].position)
  sensor_pos = np.array(sensor_pos)
  washlist = []
  for i in range(len(debris_name_list)):
    line = i*6002
    flag = 0
    count=0
    for j in range(6):
      debris_vec=debris_data.iloc[line+1+j*1000,-3:].to_numpy().astype(float)
      if np.linalg.norm(sensor_pos[j]-debris_vec) > 4000:
        count+=1
    if count == 6:
        flag = 1
    if debris_name_list[i] not in access_names:
      flag = 1
    if debris_name_list[i].endswith('1') and debris_name_list[i][:-1] in debris_name_list:
      flag = 1
    if flag == 1:
      washlist.append(list(range(line,line+6002)))
    else:
      print(debris_name_list[i])
  washlist = np.array(washlist)
  washlist = washlist.flatten().tolist()
  reduced_debris_data = debris_data.drop(washlist)
  reduced_debris_data.reset_index(drop=True,inplace=True)

  star = pd.read_csv(star_file,header=None)
  star_magnitude = pd.read_csv(magnitude_file,header=None)
  star_washlist = []
  magnitude_wash_list=[]
  for _ in star.iterrows():
    if _[0]%3 == 0:
      name = _[1][1].rsplit('-',maxsplit=1)[0].replace(' ','')
      if name not in access_names:
        star_washlist.append(list(range(_[0],_[0]+3)))
        magnitude_wash_list.append([int(_[0]/3*2),int(_[0]/3*2)+1])
  star_washlist = np.array(star_washlist)
  star_washlist = star_washlist.flatten().tolist()
  magnitude_wash_list = np.array(magnitude_wash_list)
  magnitude_wash_list = magnitude_wash_list.flatten().tolist()
  # print(magnitude_wash_list)
  # print(len(star_washlist)/3)
  # print(len(star))
  # print(len(magnitude_wash_list)/2)
  reduced_star = star.drop(star_washlist)
  reduced_star.reset_index(drop=True,inplace=True)
  reduced_star_magnitude = star_magnitude.drop(magnitude_wash_list)
  reduced_star_magnitude.reset_index(drop=True,inplace=True)

  reduced_debris_data.to_csv(reduced_debris_file,index=False,header=None)
  reduced_star.to_csv(reduced_star_file,index=False,header=None)
  reduced_star_magnitude.to_csv(reduced_magnitude_file,index=False,header=None)

def read_reduced_data(sensor_file, Csb, reduced_debris_file, reduced_star_file, reduced_magnitude_file, access_file_csv, access_file_txt):
  sensor, time_list = read_sensor_data(sensor_file,Csb)
  debris, debris_name_list = read_debris_data(reduced_debris_file)
  star, star_name_list = read_star_data(reduced_star_file, reduced_magnitude_file)
  access = read_access_data(access_file_csv, access_file_txt, 22)
  
  return sensor, time_list, debris, debris_name_list, star, star_name_list, access