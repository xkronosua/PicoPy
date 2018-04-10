import picopy
import picopy.pico_status as status

import copy
import math
import time

pico = picopy.Pico3k()

#pico.configure_channel('A', enable=True, voltage_range='20mV',channel_type='DC', offset=0.0)
#pico.configure_channel('B', enable=True, voltage_range='20mV',channel_type='DC', offset=0.0)

n_captures = 100
pico.setChannel("A", coupling="DC", VRange='500mV')
pico.setChannel("B", coupling="DC", VRange='500mV')
(sampleInterval, noSamples, maxSamples) = pico.setSamplingInterval(0.00000005,0.0035)

#trigger = picopy.EdgeTrigger(channel='B', threshold=-0.35, direction='FALLING')
#pico.set_trigger(trigger)
pico.setSimpleTrigger(trigSrc="B", threshold_V=-0.350, direction='FALLING',
						 timeout_ms=100, enabled=True,delay=0)

r = pico.capture_prep_block( number_of_frames=n_captures, downsample=0, downsample_mode='NONE',
        return_scaled_array=1)

from pylab import *

dataA = r[0]['A']
dataB = r[0]['B']

#for i in dataA:
plot(dataA.mean(axis=0),'r')

#for i in dataB:
plot(dataB.mean(axis=0),'g')
show()

del pico
