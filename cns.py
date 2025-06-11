import numpy as np 
import pandas as pd 
import torch
from my_navigation import Navigation
import matplotlib.pyplot as plt

def quaternion_multiply(q, p):# 四元数乘法
    x1, y1, z1, w1 = q[0], q[1], q[2], q[3]
    x2, y2, z2, w2 = p[0], p[1], p[2], p[3]
    
    # 计算新四元数的各分量
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    
    return np.array([x, y, z, w])

def quaternion_conjugate(q):
    """
    四元数共轭，将虚部取反，输入/输出格式 [x, y, z, w]
    """
    return np.array([-q[0], -q[1], -q[2], q[3]])

def quaternion_inverse(q):
    """
    四元数逆，单位四元数的逆等于其共轭，输入/输出格式 [x, y, z, w]
    """
    norm_sq = q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2
    if norm_sq < 1e-10:
        raise ValueError("四元数模长接近零，无法求逆。")
    inv_norm = 1.0 / norm_sq
    return quaternion_conjugate(q) * inv_norm

def quaternion_divide(q, p):
    """
    四元数除法 q / p = q * p^{-1}，输入/输出格式 [x, y, z, w]
    """
    p_inv = quaternion_inverse(p)
    return quaternion_multiply(q, p_inv)

def get_C_from_q(q):  # 从四元数到坐标变换矩阵
  if q[3] < 0:
    q = -q
  q0, q1, q2, q3 = q
  Msi = np.array([[q3**2+q0**2-q1**2-q2**2, 2*q3*q2+2*q0*q1,     -2*q3*q1+2*q0*q2],
                  [-2*q3*q2+2*q0*q1,    q3**2-q0**2+q1**2-q2**2, 2*q3*q0+2*q1*q2],
                  [2*q3*q1+2*q0*q2,     -2*q3*q0+2*q1*q2,    q3**2-q0**2-q1**2+q2**2]])
  return Msi

def get_q_from_C_bi(C_bi):  #从坐标变化矩阵到四元数
  q = np.zeros(4)
  q[0] = -np.sqrt(1+C_bi[0,0]-C_bi[1,1]-C_bi[2,2])/2
  q[1] = -np.sqrt(1-C_bi[0,0]+C_bi[1,1]-C_bi[2,2])/2
  q[2] = -np.sqrt(1-C_bi[0,0]-C_bi[1,1]+C_bi[2,2])/2
  q[3] = np.sqrt(1+C_bi[0,0]+C_bi[1,1]+C_bi[2,2])/2
  return q


class CNS(Navigation):
  @staticmethod
  def get_attitude_angle_from_q(q): #从四元数到姿态角
    q0, q1, q2, q3 = q
    phi = np.arcsin(2*(q0*q1+q2*q3))
    psi = -np.arctan2(2*(q1*q3-q0*q2), q0**2-q1**2-q2**2+q3**2)
    gamma = -np.arctan2(2*(-q0*q3+q1*q2), q0**2-q1**2+q2**2-q3**2)
    return np.array([phi,psi,gamma])
  
  @staticmethod
  def quest(b,r):
    '''基于星敏感器矢量观测的微小卫星姿态确定算法研究 南理工硕士
    观测矢量b，惯性系下r，有b=C_ib*r
    \sigma=\sum a_ib_i^Tr_i，a_i为加权系数，和为1
    B = \sum a_ib_ir_i^T
    S = B+B^T
    z=[B_23-B_32, B_31-B_13, B_12-B_21]^T
    g=[(1+sigma)I-S]^{-1}z
    '''
    sigma = 0
    B = np.zeros((3,3))
    n = b.shape[0]
    for i in range(n):
      bi = b[i].reshape(3,1)
      ri = r[i].reshape(3,1)
      sigma += (bi.T @ ri /n)[0][0]
      B += bi @ (ri.T) /n
    S = B + B.T
    z = np.array([[B[1,2]-B[2,1]], [B[2,0]-B[0,2]], [B[0,1]-B[1,0]]])
    g = np.eye(3)*(1+sigma) - S
    g = np.linalg.inv(g) @ z
    g_ = np.ones((4,1))
    # g_[1:] = g  #??
    g_[:-1] = g
    q = g_ / np.linalg.norm(g_)
    q = q.flatten()
    # return np.array([q[2],-q[1],q[0],-q[3]])
    return q
  
  def attitude_measure(self): # 使用UQEST算法计算姿态
    b = np.array(self.star_table.loc[:, ['u', 'v']].values)
    r = np.array(self.star_table.loc[:, ['x', 'y', 'z']].values)
    b = np.append((b)*self.dh, np.ones((b.shape[0], 1))*self.f, axis=1)
    b = b/np.linalg.norm(b, axis=1).reshape(-1, 1)
    self.q = CNS.quest(b, r)
    self.attitude_angle = CNS.get_attitude_angle_from_q(self.q)
  
  def get_q_attitude_angle(self):
    self.attitude_measure()
    return self.q, self.attitude_angle
  
  @staticmethod
  def least_squares( A, b):
    # 使用SVD分解求解最小二乘问题
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    L_flat = Vt.T @ np.diag(1/s) @ U.T @ b
    return L_flat
  
  def direction_vector(self):# 获取各个碎片的方向矢量，储存在L中，R是对应的各个碎片发坐标
    debris = self.debris_table
    star = self.star_table
    n_debris = debris.shape[0]
    n_star = star.shape[0]
    L = []
    R = []
    for i in range(n_debris):
        S = []
        cos = []
        debris_u, debris_v = debris.iloc[i][['u','v']].astype(float).values
        debris_uv = np.array([(debris_u)*0.000006,(debris_v)*0.000006,0.01413])
        debris_vec = debris.iloc[i][['x','y','z']].astype(float).values
        for j in range(n_star):
            star_vec = star.iloc[j][['x','y','z']].astype(float).values
            star_vec = np.array(star_vec)
            star_u, star_v = star.iloc[j][['u','v']].astype(float).values
            star_uv = np.array([(star_u)*0.000006,(star_v)*0.000006,0.01413])
            cos_angle = np.dot(star_uv,debris_uv)/(np.linalg.norm(star_uv)*np.linalg.norm(debris_uv))
            # print(star_uv,debris_uv,cos_angle)
            S.append(star_vec)
            cos.append(cos_angle)
        L0 = self.least_squares(np.array(S),np.array(cos))
        # print(S)
        L.append(L0)
        R.append(debris_vec)
    self.L = np.array(L)
    self.R = np.array(R)
    self.n = self.L.shape[0]


  def get_Ac(self): #构建那个很复杂的矩阵Ac
    L = self.L
    n = self.n
    matrix = np.zeros((int(3*n*(n-1)/2),n))
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            matrix[3*count:3*count+3,i] = list(L[i])
            matrix[3*count:3*count+3,j] = [-x for x in L[j]]
            count += 1
    return matrix

  def get_delta_R(self): #Ac*rho=delta_R
    R = self.R
    n = self.n
    matrix = np.zeros((int(3*n*(n-1)/2)))
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            matrix[3*count:3*count+3] = R[j]-R[i]
            count += 1
    return matrix

  def get_position(self):# 计算位置，取平均值
    R = self.R
    L = self.L
    rho = self.rho
    position = []
    for i in range(len(R)):
        position.append(R[i]+rho[i]*L[i])
    self.pos = np.array(position)
    return self.pos
  
  def cns_measure2(self): #主程序，有多个碎片时的天文导航定位
    self.direction_vector()
    Ac = self.get_Ac()
    delta_R = self.get_delta_R()
    self.rho = CNS.least_squares(Ac, delta_R)
    position = self.get_position()
    return position.mean(axis=0)


def cns_measure1_toech(ini_pos, ini_v, L0, R0, delta_H=0, delta_E=0):
  # ！！！没用，试着写的，doesn't work
  # 有轨道机动的时候，delta_H和delta_E都不为0
  # 角动量守恒np.cross(ini_pos, ini_v) = L, 能量守恒0.5*np.norm(ini_v)**2 - mu/np.linalg.norm(ini_pos) = E
  pos = torch.tensor(ini_pos, dtype=torch.float32, requires_grad=True)
  v = torch.tensor(ini_v, dtype=torch.float32, requires_grad=True)
  mu = 398600.4418
  L = torch.tensor(L0, dtype=torch.float32)
  R = torch.tensor(R0, dtype=torch.float32)
  
  H = torch.cross(pos, v)
  H = torch.nn.functional.normalize(H, dim=0)  # 归一化
  E = 0.5*torch.norm(v)**2 - mu/torch.norm(pos)
  E = torch.nn.functional.normalize(E, dim=0)  # 归一化

  optimizer = torch.optim.Adam([pos, v], lr=1e-3)
  loss_fn = torch.nn.MSELoss()
  for i in range(10000):
    optimizer.zero_grad()
    H_ = torch.cross(pos, v)  # 归一化
    H_ = torch.nn.functional.normalize(H_, dim=0)  # 归一化
    E_ = 0.5*torch.norm(v)**2 - mu  # 归一化
    E_ = torch.nn.functional.normalize(E_, dim=0)  # 归一化
    delta_angle = torch.dot(L, pos-R) / (torch.norm(L) * torch.norm(pos-R))
    print(L, pos-R)
    # loss = loss_fn(H_, H) + loss_fn(E_, E) + loss_fn(torch.abs(delta_angle),torch.tensor(1.0))
    loss1 = torch.nn.functional.mse_loss(H_, H)
    loss2 = torch.nn.functional.mse_loss(E_, E)
    loss3 = torch.nn.functional.mse_loss(torch.abs(delta_angle), torch.tensor(1.0))
    loss = loss1 + loss2 + loss3
    print(H, H_)
    print(E, E_)
    print(delta_angle)
    loss.backward()
    optimizer.step()
    if i % 1000 == 0:
      print(f"Iteration {i}, Loss: {loss.item()}")
  pos = pos.detach().numpy()
  v = v.detach().numpy()
  return pos, v


def cns_main(star, debris,time_list,Csb=np.eye(3),dq = np.array([0,0.701,0,0.701])):
  ## Csb 是坐标转换矩阵；dq是从星敏到卫星的四元数，由stk导出
  qC = get_q_from_C_bi(Csb)
  df = []
  columns = ['time', 'x', 'y', 'z', 'q0', 'q1', 'q2', 'q3', 'phi', 'psi', 'gamma', 'n_star', 'n_debris']

  for time0 in time_list:
    series_ = {'time': time0}
    star_table = star[star['time'] == time0]
    debris_table = debris[debris['time'] == time0]
    n_star = star_table.shape[0]
    n_debris = debris_table.shape[0]
    series_.update({'n_star': n_star, 'n_debris': n_debris})
    cns = CNS(star_table,debris_table, Csb=Csb)
    q, attitude = cns.get_q_attitude_angle()
    q = quaternion_multiply(q, qC)
    q = quaternion_divide(q, dq) #这样解算的是STK导出Satellite-Attitude Qanternions
    if q[3] < 0:
      q = -q
    series_.update({'q0': q[0], 'q1': q[1], 'q2': q[2], 'q3': q[3]})
    series_.update({'phi': attitude[0], 'psi': attitude[1], 'gamma': attitude[2]})
    if n_debris < 2:
      series_.update({'x': np.nan, 'y': np.nan, 'z': np.nan})
      print(time0)
      df.append(series_)
      continue
    pos_ = cns.cns_measure2()
    series_.update({'x': pos_[0], 'y': pos_[1], 'z': pos_[2]})
    df.append(series_)
  df = pd.DataFrame(df, columns=columns)
  df['time'] = pd.to_datetime(df['time'])
  return df