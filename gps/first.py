
import time
from openpyxl import Workbook
import numpy as np, csv
import pandas as pd
import nidaqmx as daq, pprint
import matplotlib.pyplot as plt
from nidaqmx.stream_readers import AnalogSingleChannelReader, AnalogMultiChannelReader
import sys

pp = pprint.pprint

#fig, axs = plt.subplots()
#fig.canvas.manager.show()
#plt.ion()

sampling_rate = 2500

task = daq.Task()
task.ai_channels.add_ai_voltage_chan("cDAQ9188XT-1ADE9F6Mod3/ai1")
task.timing.cfg_samp_clk_timing(rate = sampling_rate, sample_mode = daq.constants.AcquisitionType.CONTINUOUS)
task.in_stream.input_buf_size = (10**7)
reader = AnalogMultiChannelReader(task.in_stream)

sample_array = np.zeros([1, sampling_rate], dtype = np.float64)
task.start()
no_of_speeds= int(sys.argv[1])
dur = 5
running_time = dur * no_of_speeds
num = 0


num_of_files = int(running_time/dur)


t_0 = time.time() - dur - 50

for i in range(num_of_files):
    
    t = np.empty(shape = (0, 0), dtype = np.float64)
    v_y = np.empty(shape = (0, 0), dtype = np.float64)
    
    if (time.time() - t_0) < (dur):
        time.sleep((dur) - time.time() + t_0)
    
    t_0 = time.time()
    t_1 = 0
    
    for num_sec in range(dur*i, dur*(i+1)):
        t_2 = time.time()
        csvobj = open("RMS_MOTOR\\speed_" + str(num_sec + 1) + ".csv", 'w', newline = '')
        csvw = csv.writer(csvobj)
        csvw.writerow(['v_y'])
        plt.xlim([0, num_sec + 5])
        
        reader.read_many_sample(data = sample_array, number_of_samples_per_channel = sampling_rate)
        
        
        t = np.linspace(num_sec, num_sec + 1, sampling_rate)
        v_y = sample_array[0, :]
    
     
        t_1 = t_1 + (time.time() - t_2)
        for row_num in range(0, sampling_rate):
            csvw.writerow([v_y[row_num].tolist()])

        csvobj.close()
     
        #axs.plot(t, v_y, 'k-')
        #axs.set_title('Y-axis vibration')
     
 
        #fig.canvas.draw()
        #fig.canvas.flush_events()
        #plt.pause(10**(-5))
 
    pp("-----Time taken in acquiring:" + str(t_1))
    pp("-----Total time taken: " + str(time.time() - t_0))
    #fig.savefig("RMS MOTOR\\speed_" + str(i + 1) + ".png")
    #plt.show()

task.stop()
task.close()

wb = Workbook()
ws = wb.active
ws.title = "rms values"

for i in range(num_of_files):
    df = pd.read_csv("RMS_MOTOR\\speed_" + str(i + 1) + ".csv")
    df['c'] = df['v_y']**2
    sum = df['c'].sum(axis = 0)
    rms = np.sqrt(sum/float(sampling_rate*dur))
    # #plot this rms vs time after each dur
    # plt.plot(running_time, rms)
    # plt.xlabel('Time')
    # plt.ylabel('RMS')
    # plt.title("RMS")
    # plt.show()
    df['rms'] = df['v_y']*0
    df.at[i + 1, 'rms'] = rms
    del df['c']
    ws.cell(column = 1, row = i+1).value = rms
 
    
    
wb.save("RMS_MOTOR\\rms_speed.xlsx")