import picopy
import picopy.pico_status as status

import copy
import math
import time

pico = picopy.Pico3k()

#pico.configure_channel('A', enable=True, voltage_range='20mV',channel_type='DC', offset=0.0)
#pico.configure_channel('B', enable=True, voltage_range='20mV',channel_type='DC', offset=0.0)

n_captures = 100
pico.setChannel("A", coupling="DC", VRange='20mV')
pico.setChannel("B", coupling="DC", VRange='500mV')
(sampleInterval, noSamples, maxSamples) = pico.setSamplingInterval(0.00001,0.0035,
	number_of_frames=n_captures, downsample=1, downsample_mode='NONE')

#trigger = picopy.EdgeTrigger(channel='B', threshold=-0.35, direction='FALLING')
#pico.set_trigger(trigger)
pico.setSimpleTrigger(trigSrc="B", threshold_V=-0.350, direction='FALLING',
						 timeout_ms=10, enabled=True,delay=0)

r = pico.capture_prep_block(return_scaled_array=1)

from pylab import *

dataA = r[0]['A']
dataB = r[0]['B']
t = r[1]
t_ = np.linspace(0,(t[-1]-t[0])/len(dataA),len(dataA[0]))

#for i in dataA:
plot(t_,dataA.mean(axis=0),'r')

#for i in dataB:
plot(t_,dataB.mean(axis=0),'g')
show()

del pico
