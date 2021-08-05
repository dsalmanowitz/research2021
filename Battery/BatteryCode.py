# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 12:00:51 2021

@author: David Romens
"""


import pandas as pd
import matplotlib.pyplot as plt
import math 


P_rated = 3.3   #value in kW

E_rated = 7        #value in kWH

Bat_Efficiency = .85 

E_bat = .5 * E_rated #initial amount of energy stored in battery on 1st day

delta_t = 1/15

P_bat = 0

P_bat_list = []       #empty list for graphing energy values later

Tx_power = pd.read_csv("C:\\Users\\djrom\\SYS800 code\\Single Day October 31.csv")

print(Tx_power.head())
    
H = Tx_power["dataid"]                  #unique id for household data
d = Tx_power["day number"]              #what day of the year
m = Tx_power["time quarter number"]     #what minute of the day 
    
    
t = 0  #for loop counter

for i in Tx_power["time quarter number"]:
    P_use = Tx_power["grid1"]         #energy used calculated in excel as sum of all eguage values except grid and solar
    P_gen = Tx_power["solar1"]
    P_target = P_use[t]-P_gen[t]
    
    if P_target > P_rated:
        P_target = P_rated
    
    if P_target > 0 and E_bat >= 0:
       P_bat = min(P_target, P_rated)
       
    elif P_target < 0 and E_bat <= E_rated:
        P_bat = max(P_target, -P_rated)
        
    else : 
        P_bat = 0
        
    if P_bat > 0:
        k = 1/math.sqrt(Bat_Efficiency)     #discharging
    else : 
        k = math.sqrt(Bat_Efficiency)       #charging
        
    E_bat = E_bat - P_bat * k * delta_t
    
    list.append(P_bat_list, P_bat)
  #  print(t , E_bat)
    t = t + 1 
    if t > 210220:
        break
    
Tx_power["Battery Power"] = P_bat_list    #add battery power list to dataframe
print(Tx_power.head())
Tx_power["date"] = Tx_power["date"].astype(str)


plt.plot("time", "Battery Power", data=Tx_power, color = 'green', marker = '')
plt.plot("time", "solar1", data=Tx_power, color = 'blue', marker = '')
plt.plot("time", "grid1", data=Tx_power, color = 'red', marker = '')
plt.title('Battery Power over time', fontsize = 14)
plt.xlabel("time", fontsize = 14)
plt.ylabel("Power (kW)", fontsize = 14)
plt.legend()
plt.show()
