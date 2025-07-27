import dynamixel_sdk
import numpy as np

class MotorController:
    ADDR_TORQUE_ENABLE = 64
    ADDR_GOAL_POSITION = 116
    ADDR_PRESENT_LOAD = 126
    ADDR_PRESENT_VELOCITY = 128
    ADDR_PRESENT_POSITION = 132
    ADDR_OPERATING_MODE = 11

    def __init__(self, port, motor_list, baudrate=1000000):
        self.port = dynamixel_sdk.PortHandler(port)
        self.packet = dynamixel_sdk.PacketHandler(2.0)
        if not self.port.openPort():
            raise Exception(f"Failed to open port {port}")
        if not self.port.setBaudRate(baudrate):
            raise Exception(f"Failed to set baudrate {baudrate} on port {port}")
        self.motor_list = motor_list
        self.find_offset()

    def __del__(self):
        self.disable()

    def find_offset(self):
        self.offset = [0] * len(self.motor_list)
        initial_positions = self.get_feedback()[0]
        offset = []
        for motor, pos in zip(self.motor_list, initial_positions):
            center = (motor['min_position'] + motor['max_position']) / 2 + motor['offset']
            delta = pos - center
            offset.append(np.round(delta / (2 * np.pi)) * 2 * np.pi + motor['offset'])
        self.offset = offset

    def enable(self):
        for motor in self.motor_list:
            res, err = self.packet.write1ByteTxRx(self.port, motor['id'], self.ADDR_OPERATING_MODE, 4) # multi-turn mode
            if res != dynamixel_sdk.COMM_SUCCESS:
                raise Exception(f"Failed to set operating mode: {self.packet.getTxRxResult(res)}")
            res, err = self.packet.write1ByteTxRx(self.port, motor['id'], self.ADDR_TORQUE_ENABLE, 1)
            if res != dynamixel_sdk.COMM_SUCCESS:
                raise Exception(f"Failed to enable torque: {self.packet.getTxRxResult(res)}")
        self.find_offset()

    def disable(self):
        for motor in self.motor_list:
            res, err = self.packet.write1ByteTxRx(self.port, motor['id'], self.ADDR_TORQUE_ENABLE, 0)
            if res != dynamixel_sdk.COMM_SUCCESS:
                raise Exception(f"Failed to disable torque: {self.packet.getTxRxResult(res)}")

    def pos_to_dxl_units(self, pos):
        return int((pos) * 4095 / (2 * np.pi))

    def dxl_units_to_pos(self, dxl_units):
        return (dxl_units / 4095 * 2 * np.pi)

    def dxl_units_to_vel(self, dxl_units):
        return dxl_units * 2 * np.pi * 0.229 / 60

    def interpret_int_as_signed(self, value, num_bits):
        if value & (1 << (num_bits - 1)):
            return value - (1 << num_bits)
        return value

    def set_positions(self, positions):
        sync_write = dynamixel_sdk.GroupSyncWrite(self.port, self.packet, self.ADDR_GOAL_POSITION, 4)
        for pos, motor, offset in zip(positions, self.motor_list, self.offset):
            data = [0] * 4
            pos = np.clip(pos, motor['min_position'], motor['max_position'])
            pos_dxl_units = self.pos_to_dxl_units(pos + offset)
            print(pos_dxl_units)
            data[0] = pos_dxl_units & 0xFF
            data[1] = (pos_dxl_units >> 8) & 0xFF
            data[2] = (pos_dxl_units >> 16) & 0xFF
            data[3] = (pos_dxl_units >> 24) & 0xFF
            sync_write.addParam(motor['id'], data)
        dxl_comm_result = sync_write.txPacket()
        if dxl_comm_result != dynamixel_sdk.COMM_SUCCESS:
            raise Exception(f"Failed to set positions: {self.packet.getTxRxResult(dxl_comm_result)}")
        sync_write.clearParam()

    def get_feedback_raw(self):
        sync_read = dynamixel_sdk.GroupSyncRead(self.port, self.packet, self.ADDR_PRESENT_LOAD, 2 + 4 + 4)
        for motor in self.motor_list:
            sync_read.addParam(motor['id'])
        dxl_comm_result = sync_read.txRxPacket()
        if dxl_comm_result != dynamixel_sdk.COMM_SUCCESS:
            raise Exception(f"Failed to get feedback: {self.packet.getTxRxResult(dxl_comm_result)}")
        positions = []
        velocities = []
        loads = []
        for motor in self.motor_list:
            if sync_read.isAvailable(motor['id'], self.ADDR_PRESENT_POSITION, 4):
                data = sync_read.getData(motor['id'], self.ADDR_PRESENT_POSITION, 4)
                positions.append(self.interpret_int_as_signed(data, 32))
            else:
                raise Exception(f"Motor {motor['id']} not found in sync read")
            if sync_read.isAvailable(motor['id'], self.ADDR_PRESENT_VELOCITY, 4):
                data = sync_read.getData(motor['id'], self.ADDR_PRESENT_VELOCITY, 4)
                velocities.append(self.interpret_int_as_signed(data, 32))
            else:
                raise Exception(f"Motor {motor['id']} not found in sync read")
            if sync_read.isAvailable(motor['id'], self.ADDR_PRESENT_LOAD, 2):
                data = sync_read.getData(motor['id'], self.ADDR_PRESENT_LOAD, 2)
                loads.append(self.interpret_int_as_signed(data, 16))
            else:
                raise Exception(f"Motor {motor['id']} not found in sync read")
        sync_read.clearParam()
        print('read', positions)
        return positions, velocities, loads

    def get_feedback(self):
        positions_raw, velocities_raw, loads_raw = self.get_feedback_raw()
        positions = [self.dxl_units_to_pos(pos) for pos in positions_raw]
        positions = [pos - offset for pos, offset in zip(positions, self.offset)]
        velocities = [self.dxl_units_to_vel(vel) for vel in velocities_raw]
        loads = [load/1000 for load in loads_raw]
        return positions, velocities, loads



if __name__ == "__main__":
    import sys
    import time
    drv = MotorController(port=sys.argv[1], motor_list=[
        {'id': 10, 'min_position': -0.79, 'max_position': 0.79, 'offset': 0.79},
        {'id': 11, 'min_position': -0.79, 'max_position': 0.79, 'offset': 0.79},
        {'id': 20, 'min_position': -0.79, 'max_position': 0.79, 'offset': -0.79},
        {'id': 21, 'min_position': -0.79, 'max_position': 0.79, 'offset': 0.79},
        {'id': 30, 'min_position': -0.79, 'max_position': 0.79, 'offset': 0.79},
        {'id': 31, 'min_position': -0.79, 'max_position': 0.79, 'offset': 0.79},
        {'id': 40, 'min_position': -0.79, 'max_position': 0.79, 'offset': -0.79},
        {'id': 41, 'min_position': -0.79, 'max_position': 0.79, 'offset': -0.79},
    ])
    drv.disable()
    drv.enable()
    while True:
        t_start = time.time()
        pos, vel, load = drv.get_feedback()
        print(time.time() - t_start)
        # print(pos)
        # time.sleep(0.01)
        drv.set_positions([np.sin(time.time())*0.8]*len(drv.motor_list))
        t_end = time.time()
        print(f"Time taken: {t_end - t_start:.3f}s")
        time.sleep(0.001)
