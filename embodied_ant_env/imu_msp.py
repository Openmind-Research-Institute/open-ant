import serial
import sys
import struct
import time

# based on https://github.com/tonbut/rpi-python-drone/blob/master/air/pyMultiWii/pyMultiwii.py
# TODO for scaling check this:
# https://github.com/betaflight/betaflight-configurator/blob/aeda56ba407ba54068bad90d7cc069b67d2cd8e4/src/js/msp/MSPHelper.js#L116-L131

class IMU_MSP:
    RAW_IMU = 102
    ATTITUDE = 108

    def __init__(self, port: str, baudrate: int = 115200):
        self.ser = serial.Serial(port, baudrate)

    def sendCMD(self, data_length, code, data):
        checksum = 0
        total_data = [b'$', b'M', b'<', data_length, code] + data
        for i in struct.pack('<2B%dH' % len(data), *total_data[3:len(total_data)]):
            checksum = checksum ^ i
        total_data.append(checksum)
        try:
            b = None
            b = self.ser.write(struct.pack(f'<3c2B{len(data)}HB', *total_data))
        except Exception as error:
            print("\n\nError in sendCMD.")
            print("("+str(error)+")\n\n")
            pass
        return b


    def getData(self, cmd):
        start = time.time()
        # self.ser.flushInput()
        # self.ser.flushOutput()
        self.sendCMD(0,cmd,[])
        while True:
            header = self.ser.read()
            if header == b'$':
                header = header+self.ser.read(2)
                break
        datalength = struct.unpack('<b', self.ser.read())[0]
        # print(datalength)
        code_ = struct.unpack('<b', self.ser.read())
        data = self.ser.read(datalength)
        temp = struct.unpack(f'<{datalength//2}h',data)
        self.ser.flushInput()
        self.ser.flushOutput()
        elapsed = time.time() - start
        # print(temp)
        if cmd == self.ATTITUDE:
            attitude = {}
            attitude['angx']=float(temp[0]/10.0)
            attitude['angy']=float(temp[1]/10.0)
            attitude['heading']=float(temp[2])
            attitude['elapsed']=round(elapsed,3)
            attitude['timestamp']="%0.2f" % (time.time(),) 
            return attitude
        if cmd == self.RAW_IMU:
            rawIMU = {}
            rawIMU['ax']=float(temp[0])
            rawIMU['ay']=float(temp[1])
            rawIMU['az']=float(temp[2])
            rawIMU['gx']=float(temp[3])
            rawIMU['gy']=float(temp[4])
            rawIMU['gz']=float(temp[5])
            rawIMU['mx']=float(temp[6])
            rawIMU['my']=float(temp[7])
            rawIMU['mz']=float(temp[8])
            rawIMU['elapsed']=round(elapsed,5)
            rawIMU['timestamp']="%0.3f" % (time.time(),)
            return rawIMU


if __name__ == "__main__":
    imu = IMU_MSP(sys.argv[1], 1000000)
    prev_time = time.time()
    delta_avg = 0
    while True:
        print(imu.getData(imu.RAW_IMU))
        print(imu.getData(imu.ATTITUDE))
        now = time.time()
        delta = now - prev_time
        prev_time = now
        delta_avg = (delta_avg * 0.9) + (delta * 0.1)
        print(f"update rate: {1/delta_avg}")