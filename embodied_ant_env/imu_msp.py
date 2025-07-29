import serial
import sys
import struct
import time
import math

class IMU_MSP:
    RAW_IMU = 102
    ATTITUDE = 108
    FC_VARIANT = 2
    BOARD_INFO = 4

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
        self.board_info = None

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
            rawIMU['ax']=data[0] / 512.0 * 9.81 # seems to be accurate
            rawIMU['ay']=data[1] / 512.0 * 9.81
            rawIMU['az']=data[2] / 512.0 * 9.81
            rawIMU['wx']=data[3] / 16.4 * math.pi / 180.0 # seems to be accurate
            rawIMU['wy']=data[4] / 16.4 * math.pi / 180.0
            rawIMU['wz']=data[5] / 16.4 * math.pi / 180.0
            rawIMU['mx']=data[6] / 1090.0 # TODO untested
            rawIMU['my']=data[7] / 1090.0
            rawIMU['mz']=data[8] / 1090.0
            rawIMU['ax_raw']=data[0]
            rawIMU['ay_raw']=data[1]
            rawIMU['az_raw']=data[2]
            rawIMU['wx_raw']=data[3]
            rawIMU['wy_raw']=data[4]
            rawIMU['wz_raw']=data[5]
            rawIMU['mx_raw']=data[6]
            rawIMU['my_raw']=data[7]
            rawIMU['mz_raw']=data[8]
            rawIMU['timestamp']=start_time
            return rawIMU
        elif cmd == self.FC_VARIANT:
            fc_variant = {}
            fc_variant['fc_variant'] = data_bytes.decode('utf-8')
            fc_variant['timestamp'] = start_time
            return fc_variant
        elif cmd == self.BOARD_INFO:
            board_info = {}
            # https://github.com/betaflight/betaflight/blob/ddb0c667faa50a914a50170d35856f9a1286ddf5/src/main/msp/msp.c#L654
            board_info['board'] = data_bytes[:4].decode('utf-8')
            board_info['hw_revision'] = struct.unpack('<H', data_bytes[4:6])[0]
            board_info['USE_MAX7456'] = data_bytes[6]
            board_info['target_capabilities'] = data_bytes[7]
            def decode_pstring(data):
                len = struct.unpack('<B', data[0:1])[0]
                return data[1:len+1].decode('utf-8'), data[len+1:]
            board_info['target_name'], rem = decode_pstring(data_bytes[8:])
            board_info['board_name'], rem = decode_pstring(rem)
            board_info['manufacturer'], rem = decode_pstring(rem)
            board_info['signature'] = struct.unpack('<32B', rem[:32])
            board_info['mcu_type'] = struct.unpack('<B', rem[32:33])[0]
            board_info['config_state'] = struct.unpack('<B', rem[33:34])[0]
            board_info['gyro_sample_rate'] = struct.unpack('<H', rem[34:36])[0]
            board_info['config_problems'] = struct.unpack('<I', rem[36:40])[0]
            board_info['spi_dev_count'] = struct.unpack('<B', rem[40:41])[0]
            board_info['i2c_dev_count'] = struct.unpack('<B', rem[41:42])[0]
            board_info['timestamp'] = start_time
            return board_info
        else:
            print(f"unknown command: {cmd}")
            return None

    def get_data(self):
        attitude = self.read_cmd(self.ATTITUDE)
        raw_imu = self.read_cmd(self.RAW_IMU)
        imu = self.imu_raw_to_imu(raw_imu)
        return {**attitude, **imu}

    def imu_raw_to_imu(self, raw_imu):
        """ different boards seem to have different scale factors for the imu data """
        if self.board_info is None:
            self.board_info = self.read_cmd(self.BOARD_INFO)
        # https://github.com/betaflight/betaflight-configurator/blob/aeda56ba407ba54068bad90d7cc069b67d2cd8e4/src/js/msp/MSPHelper.js#L116-L131
        if self.board_info['board_name'] == 'KAKUTEH7MINI':
            imu = {}
            imu['ax'] = raw_imu['ax_raw'] / 512.0 * 9.81 # seems to be accurate
            imu['ay'] = raw_imu['ay_raw'] / 512.0 * 9.81
            imu['az'] = raw_imu['az_raw'] / 512.0 * 9.81
            imu['wx'] = raw_imu['wx_raw'] / 16.4 * math.pi / 180.0 # seems to be accurate
            imu['wy'] = raw_imu['wy_raw'] / 16.4 * math.pi / 180.0
            imu['wz'] = raw_imu['wz_raw'] / 16.4 * math.pi / 180.0
            imu['mx'] = raw_imu['mx_raw'] # not available on this board
            imu['my'] = raw_imu['my_raw']
            imu['mz'] = raw_imu['mz_raw']
        elif self.board_info['board_name'] == 'TBS_LUCID_FC': # TODO
            imu = {}
            imu['ax'] = raw_imu['ax_raw'] / 2048.0 * 9.81 # seems to be accurate
            imu['ay'] = raw_imu['ay_raw'] / 2048.0 * 9.81
            imu['az'] = raw_imu['az_raw'] / 2048.0 * 9.81
            imu['wx'] = raw_imu['wx_raw'] / 16.4 * math.pi / 180.0 # seems to be accurate
            imu['wy'] = raw_imu['wy_raw'] / 16.4 * math.pi / 180.0
            imu['wz'] = raw_imu['wz_raw'] / 16.4 * math.pi / 180.0
            imu['mx'] = raw_imu['mx_raw'] # not available on this board
            imu['my'] = raw_imu['my_raw']
            imu['mz'] = raw_imu['mz_raw']
        else:
            raise ValueError(f"unknown board: {self.board_info['board_name']}")
        return imu


if __name__ == "__main__":
    imu = IMU_MSP(sys.argv[1], 1000000)
    print(imu.read_cmd(imu.BOARD_INFO))
    print(imu.read_cmd(imu.FC_VARIANT))
    prev_time = time.time()
    delta_avg = 0
    ang_z = 0
    while True:
        imu_data = imu.get_data()
        print(imu_data)
        # print(imu.read_cmd(imu.RAW_IMU))
        # print(imu.read_cmd(imu.ATTITUDE))

        now = time.time()
        delta = now - prev_time
        prev_time = now
        delta_avg = (delta_avg * 0.9) + (delta * 0.1)

        # use this to check gyro scale factor
        ang_z += imu_data['wz'] * delta
        print(f"ang_z: {ang_z}")

        print(f"update rate: {1/delta_avg}")
