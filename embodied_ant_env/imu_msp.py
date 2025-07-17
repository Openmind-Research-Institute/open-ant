import serial
import sys
import struct
import time
import math

class IMU_MSP:
    RAW_IMU = 102
    ATTITUDE = 108

    """
    MSP Packet Format (MSPv1)

    Request (host → flight controller):
        '$' 'M' '<' [payload_size:1B] [command:1B] [payload:N bytes] [checksum:1B]

    Response (flight controller → host):
        '$' 'M' '>' [payload_size:1B] [command:1B] [payload:N bytes] [checksum:1B]

    - Header:      3 bytes ('$', 'M', '<' or '>')
    - Payload Size: number of bytes in payload (0–255)
    - Command:     8-bit command ID (e.g., 108 = MSP_ATTITUDE)
    - Payload:     binary data specific to the command
    - Checksum:    XOR of [payload_size, command, payload...]

    All integers are little-endian. Payload interpretation depends on the command.
    """


    def __init__(self, port: str, baudrate: int = 115200):
        self.device = serial.Serial(port, baudrate)

    @staticmethod
    def checksum(data):
        checksum = 0
        for i in data:
            checksum = checksum ^ i
        return checksum

    def send_cmd(self, cmd, data=[]):
        cmd_bytes = struct.pack('<B', cmd)
        data_bytes = struct.pack(f'<{len(data)}H', *data)
        data_length_code = struct.pack('<B', len(data_bytes))
        header = b'$M<'
        checksum = self.checksum(data_length_code + cmd_bytes + data_bytes)
        msg = header + data_length_code + cmd_bytes + data_bytes + struct.pack('<B', checksum)
        self.device.write(msg)

    def read_cmd(self, cmd):
        start_time = time.time()
        self.send_cmd(cmd,[])
        while True:
            header = self.device.read(1)
            if header == b'$':
                header = header+self.device.read(2)
                break
        if header != b'$M>':
            print(f"unexpected header: {header} != $M<")
            return None
        header_args = self.device.read(2)
        data_length, code = struct.unpack('<BB', header_args)
        if code != cmd:
            print(f"received code: {code} != cmd: {cmd}")
            return None
        data_bytes = self.device.read(data_length)
        data = struct.unpack(f'<{data_length//2}h',data_bytes)
        checksum = struct.unpack('<B', self.device.read(1))[0]
        if checksum != self.checksum(header_args + data_bytes):
            print(f"checksum: {checksum} != {self.checksum(data_bytes)}")
            return None
        self.device.flushInput()
        self.device.flushOutput()
        if cmd == self.ATTITUDE:
            attitude = {}
            attitude['roll_deg']=data[0] * 0.1
            attitude['pitch_deg']=data[1] * 0.1
            attitude['yaw_deg']=data[2] * 0.1
            attitude['timestamp']=start_time
            return attitude
        elif cmd == self.RAW_IMU:
            rawIMU = {}
            # https://github.com/betaflight/betaflight-configurator/blob/aeda56ba407ba54068bad90d7cc069b67d2cd8e4/src/js/msp/MSPHelper.js#L116-L131
            rawIMU['ax']=data[0] / 512.0 * 9.81
            rawIMU['ay']=data[1] / 512.0 * 9.81
            rawIMU['az']=data[2] / 512.0 * 9.81
            rawIMU['wx']=data[3] * (4 / 16.4) * math.pi / 180.0
            rawIMU['wy']=data[4] * (4 / 16.4) * math.pi / 180.0
            rawIMU['wz']=data[5] * (4 / 16.4) * math.pi / 180.0
            rawIMU['mx']=data[6] / 1090.0
            rawIMU['my']=data[7] / 1090.0
            rawIMU['mz']=data[8] / 1090.0
            rawIMU['timestamp']=start_time
            return rawIMU
        else:
            print(f"unknown command: {cmd}")
            return None

    def get_data(self):
        attitude = self.read_cmd(self.ATTITUDE)
        raw_imu = self.read_cmd(self.RAW_IMU)
        return {**attitude, **raw_imu}


if __name__ == "__main__":
    imu = IMU_MSP(sys.argv[1], 1000000)
    prev_time = time.time()
    delta_avg = 0
    while True:
        print(imu.read_cmd(imu.ATTITUDE))
        print(imu.read_cmd(imu.RAW_IMU))
        now = time.time()
        delta = now - prev_time
        prev_time = now
        delta_avg = (delta_avg * 0.9) + (delta * 0.1)
        print(f"update rate: {1/delta_avg}")
