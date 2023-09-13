import math
import numpy as np
SR=np.array([ 112 , 140 , 110 , 90 , 90 ])
ER=120
DR=6
TI=ER+DR*4
print(ER,DR,TI)
for i in range(5):
    tempER=ER*7/8+SR[i]/8
    tempDR=0.75*DR+0.25*math.fabs(SR[i]-ER)
    tempTI=ER+DR*4
    ER=tempER
    DR=tempDR
    TI=tempTI
    print(ER, DR, TI)
