import numpy as np 
import pandas as pd 

class Navigation():
  def __init__(self, star_table, debris_table, Csb = np.eye(3)):
    self.star_table = star_table
    self.debris_table = debris_table
    self.f = 0.01413
    self.dh = 0.000006
    self.dv = 0.000006
    self.H = 700
    self.pos = np.nan 
    self.attitude_angle = np.nan 
    self.q = np.nan
    self.Csb = Csb