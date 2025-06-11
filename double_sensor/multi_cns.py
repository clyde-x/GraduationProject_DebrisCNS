'''
这是多个星敏感器时的天文定位
'''

import numpy as np
import pandas as pd
import sys
sys.path.append('f:\\buaa\\python\\final_proj') #cns.py的路径
import cns




def cns_out(log_list, C_list, dq_list, start=0,end=-1):
  if len(log_list) != len(C_list):
    raise ValueError("log_list and C_list must have the same length")
  Q_df_list = []
  L_dic = {}
  R_dic = {}
  n_debris_dic = {}
  for i in range(len(log_list)):
    log_file = log_list[i]
    C = C_list[i]
    dq = dq_list[i]

    visible_star = pd.read_csv(log_file)
    visible_star['time'] = pd.to_datetime(visible_star['time'])
    visible_star = visible_star[visible_star['category'] == 'star']
    star = visible_star.reset_index(drop=True)
    star['u'] += np.random.normal(0, 0.12, len(star))
    star['v'] += np.random.normal(0, 0.12, len(star))
    visible_debris = pd.read_csv(log_file)
    visible_debris['time'] = pd.to_datetime(visible_debris['time'])
    visible_debris = visible_debris[visible_debris['category'] == 'debris']
    debris = visible_debris.reset_index(drop=True)
    debris['u'] += np.random.normal(0, 0.12, len(debris))
    debris['v'] += np.random.normal(0, 0.12, len(debris))

    time_list = visible_star['time'].unique()[start:end]
    df = cns.cns_main(star, pd.DataFrame(columns=star.columns), time_list, Csb=C, dq=dq)
    Q_df_list.append(df[['time', 'q0', 'q1', 'q2', 'q3']].reset_index(drop=True))

    for time0 in time_list:
      star_table = star[star['time'] == time0]
      debris_table = debris[debris['time'] == time0]
      n_star = star_table.shape[0]
      n_debris = debris_table.shape[0]

      cns0 = cns.CNS(star_table,debris_table, Csb=C)
      cns0.direction_vector()
      if time0 not in L_dic:
        L_dic[time0] = [cns0.L]
      else:
        L_dic[time0].append(cns0.L)
      if time0 not in R_dic:
        R_dic[time0] = [cns0.R]
      else:
        R_dic[time0].append(cns0.R)
      if time0 not in n_debris_dic:
        n_debris_dic[time0] = [n_debris]
      else:
        n_debris_dic[time0].append(n_debris)

  Q_df = np.array(Q_df_list[0].iloc[:][[ 'q0', 'q1', 'q2', 'q3']].values)
  for i in range(1, len(Q_df_list)):
    Q_df = Q_df + np.array(Q_df_list[i].iloc[:][[ 'q0', 'q1', 'q2', 'q3']].values)
  Q_df = Q_df / len(Q_df_list)
  Q_df = pd.DataFrame(Q_df, columns=['q0', 'q1', 'q2', 'q3'])
  Q_df['time'] = pd.to_datetime(time_list)
  Q_df = Q_df[['time', 'q0', 'q1', 'q2', 'q3']]

  position_df = []
  for time0 in time_list:
    L = np.vstack([L_dic[time0][i] for i in range(len(L_dic[time0]))])
    R = np.vstack([R_dic[time0][i] for i in range(len(R_dic[time0]))])
    n_debris = np.array(n_debris_dic[time0]).sum()
    cns0 = cns.CNS(None, None)
    cns0.L = L
    cns0.R = R
    cns0.n = n_debris
    Ac = cns0.get_Ac()
    delta_R = cns0.get_delta_R()
    cns0.rho = cns0.least_squares(Ac, delta_R)
    position = cns0.get_position()
    position = position.mean(axis=0)
    position_df.append({'time': time0,'x': position[0],'y': position[1],'z': position[2],})
    print(time0, position)
  position_df = pd.DataFrame(position_df)
  position_df['time'] = pd.to_datetime(position_df['time'])
  return Q_df, position_df

def cns_count_debris(log_list, start=0,end=-1):
  n_debris_dic = {}
  for i in range(len(log_list)):
    log_file = log_list[i]

    visible_star = pd.read_csv(log_file)
    visible_star['time'] = pd.to_datetime(visible_star['time'])
    visible_star = visible_star[visible_star['category'] == 'star']
    star = visible_star.reset_index(drop=True)
    visible_debris = pd.read_csv(log_file)
    visible_debris['time'] = pd.to_datetime(visible_debris['time'])
    visible_debris = visible_debris[visible_debris['category'] == 'debris']
    debris = visible_debris.reset_index(drop=True)

    time_list = visible_star['time'].unique()[start:end]

    for time0 in time_list:
      debris_table = debris[debris['time'] == time0]
      n_debris = debris_table.shape[0]
      if time0 not in n_debris_dic:
        n_debris_dic[time0] = [n_debris]
      else:
        n_debris_dic[time0].append(n_debris)
  n_debris_df = np.array(list(n_debris_dic.values()))
  return n_debris_df