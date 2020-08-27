import subprocess
import time
from openpyxl import Workbook
import numpy as np, csv
import pandas as pd
import nidaqmx as daq, pprint
import matplotlib.pyplot as plt
from nidaqmx.stream_readers import AnalogSingleChannelReader, AnalogMultiChannelReader
import sys
import serial as ser
import struct


ser = ser.Serial('COM3',9600)
analog_write_near_value = (1,7,10,14,37,37,53,59,64,67,67,71,154,160,163)
subprocess.Popen(["python", "RAW_DATAF.py",str(len(analog_write_near_value))], shell=True)
time.sleep(1.5)
for a1 in analog_write_near_value:
    ser.write(struct.pack('i',a1))
    time.sleep(3)

ser.close()