import dynamixel_sdk
import numpy as np
import time
import logging
import serial


class MotorControllerError(Exception):
    pass


logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.FileHandler('motor_controller.log'))
    logger.addHandler(logging.StreamHandler())


class MotorController:
    ADDR_TORQUE_ENABLE = 64
    ADDR_GOAL_POSITION = 116
    ADDR_PRESENT_LOAD = 126
    ADDR_PRESENT_VELOCITY = 128
    ADDR_PRESENT_POSITION = 132
    ADDR_PRESENT_TEMPERATURE = 146
    ADDR_OPERATING_MODE = 11
    ADDR_HARDWARE_ERROR_STATUS = 70
    ADDR_PWM_LIMIT = 36
    ADDR_SHUTDOWN = 63

    def __init__(self, port, motor_list, baudrate=1000000):
        self.port = dynamixel_sdk.PortHandler(port)
        self.packet = dynamixel_sdk.PacketHandler(2.0)
        self.logger = logger

        if not self.port.openPort():
            self.logger.error("Failed to open port %s", port)
            raise MotorControllerError(f"Failed to open port {port}")
        if not self.port.setBaudRate(baudrate):
            self.logger.error("Failed to set baudrate %s on port %s", baudrate, port)
            raise MotorControllerError(f"Failed to set baudrate {baudrate} on port {port}")
        self.motor_list = motor_list
        self.find_offset()

    def __del__(self):
        try:
            self.disable()
        except Exception as e:
            self.logger.error(f"Error disabling motors in __del__: {e}")

    def _reopen_port(self):
        # Close and re-open the port to clear the buffer. Tolerant of a missing
        # device (e.g. unplugged USB): a failed re-open raises MotorControllerError
        # so callers only ever handle one exception type.
        try:
            self.port.closePort()
        except (OSError, serial.SerialException):
            pass
        try:
            if not self.port.openPort():
                raise MotorControllerError("Failed to re-open port")
        except (OSError, serial.SerialException) as e:
            raise MotorControllerError(f"Failed to re-open port: {e}")

    def sync_read_rxtx_retry(self, sync_read_obj, num_retries=3):
        try:
            for attempt in range(num_retries):
                dxl_comm_result = sync_read_obj.txRxPacket()
                if dxl_comm_result == dynamixel_sdk.COMM_SUCCESS:
                    return dxl_comm_result
                self.logger.warning("sync_read attempt %d/%d failed: %s; retrying...",
                            attempt, num_retries, dxl_comm_result)
                self._reopen_port()
            self.logger.error("sync_read failed after %d attempts: %s",
                        num_retries, dxl_comm_result)
            raise MotorControllerError(f"Failed to perform sync read: {dxl_comm_result}")
        except (OSError, serial.SerialException) as e:
            self.logger.error(f"Serial exception: {e}")
            raise MotorControllerError(f"Serial exception: {e}")

    def _check_result(self, res, err, motor_id, what):
        # res is the comm result; err is the servo status/hardware error byte.
        if res != dynamixel_sdk.COMM_SUCCESS:
            msg = f"Failed to {what} on motor {motor_id}: {self.packet.getTxRxResult(res)}"
            self.logger.error(msg)
            raise MotorControllerError(msg)
        if err != 0:
            msg = (f"Failed to {what} on motor {motor_id}: "
                   f"0x{err:02X} {self.packet.getRxPacketError(err)}")
            self.logger.error(msg)
            raise MotorControllerError(msg)

    def find_offset(self):
        self.offset = [0] * len(self.motor_list)
        initial_positions = self.get_feedback()[0]  # read positions with offset 0
        offset = []
        for motor, pos in zip(self.motor_list, initial_positions):
            center = (motor['min_position'] + motor['max_position']) / 2 + motor['offset']
            delta = pos - center
            offset.append(np.round(delta / (2 * np.pi)) * 2 * np.pi + motor['offset'])
        self.offset = offset

    def enable(self):
        for motor in self.motor_list:
            res, err = self.packet.write1ByteTxRx(self.port, motor['id'], self.ADDR_TORQUE_ENABLE, 0)
            self._check_result(res, err, motor['id'], "disable torque")
            res, err = self.packet.write1ByteTxRx(self.port, motor['id'], self.ADDR_OPERATING_MODE, 4) # multi-turn mode
            self._check_result(res, err, motor['id'], "set operating mode")
            res, err = self.packet.write2ByteTxRx(self.port, motor['id'], self.ADDR_PWM_LIMIT, int(50/0.113)) # set PWM limit
            self._check_result(res, err, motor['id'], "set PWM limit")
            val = (1 << 0) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5) # faults: voltage, overheat, encoder, electrical, overload
            res, err = self.packet.write1ByteTxRx(self.port, motor['id'], self.ADDR_SHUTDOWN, val) # set fault shutdown
            self._check_result(res, err, motor['id'], "set fault shutdown")
            # all settings must be changed before enabling torque
            res, err = self.packet.write1ByteTxRx(self.port, motor['id'], self.ADDR_TORQUE_ENABLE, 1)
            self._check_result(res, err, motor['id'], "enable torque")
        self.find_offset()

    def disable(self):
        for motor in self.motor_list:
            res, err = self.packet.write1ByteTxRx(self.port, motor['id'], self.ADDR_TORQUE_ENABLE, 0)
            self._check_result(res, err, motor['id'], "disable torque")

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
        try:
            sync_write = dynamixel_sdk.GroupSyncWrite(self.port, self.packet, self.ADDR_GOAL_POSITION, 4)
            for pos, motor, offset in zip(positions, self.motor_list, self.offset):
                data = [0] * 4
                pos = np.clip(pos, motor['min_position'], motor['max_position'])
                pos_dxl_units = self.pos_to_dxl_units(pos + offset)
                data[0] = pos_dxl_units & 0xFF
                data[1] = (pos_dxl_units >> 8) & 0xFF
                data[2] = (pos_dxl_units >> 16) & 0xFF
                data[3] = (pos_dxl_units >> 24) & 0xFF
                sync_write.addParam(motor['id'], data)
            dxl_comm_result = sync_write.txPacket()
            if dxl_comm_result != dynamixel_sdk.COMM_SUCCESS:
                self.logger.error(f"Failed to set positions: {self.packet.getTxRxResult(dxl_comm_result)}")
                raise MotorControllerError(f"Failed to set positions: {self.packet.getTxRxResult(dxl_comm_result)}")
            sync_write.clearParam()
        except (OSError, serial.SerialException) as e:
            self.logger.error(f"Serial exception: {e}")
            raise MotorControllerError(f"Serial exception: {e}")

    def get_feedback_raw(self):
        sync_read = dynamixel_sdk.GroupSyncRead(self.port, self.packet, self.ADDR_PRESENT_LOAD, 2 + 4 + 4)
        for motor in self.motor_list:
            sync_read.addParam(motor['id'])
        self.sync_read_rxtx_retry(sync_read)
        positions = []
        velocities = []
        loads = []
        for motor in self.motor_list:
            if sync_read.isAvailable(motor['id'], self.ADDR_PRESENT_POSITION, 4):
                data = sync_read.getData(motor['id'], self.ADDR_PRESENT_POSITION, 4)
                positions.append(self.interpret_int_as_signed(data, 32))
            else:
                self.logger.error(f"Motor {motor['id']} not found in sync read")
                raise MotorControllerError(f"Motor {motor['id']} not found in sync read")
            if sync_read.isAvailable(motor['id'], self.ADDR_PRESENT_VELOCITY, 4):
                data = sync_read.getData(motor['id'], self.ADDR_PRESENT_VELOCITY, 4)
                velocities.append(self.interpret_int_as_signed(data, 32))
            else:
                self.logger.error(f"Motor {motor['id']} not found in sync read")
                raise MotorControllerError(f"Motor {motor['id']} not found in sync read")
            if sync_read.isAvailable(motor['id'], self.ADDR_PRESENT_LOAD, 2):
                data = sync_read.getData(motor['id'], self.ADDR_PRESENT_LOAD, 2)
                loads.append(self.interpret_int_as_signed(data, 16))
            else:
                self.logger.error(f"Motor {motor['id']} not found in sync read")
                raise MotorControllerError(f"Motor {motor['id']} not found in sync read")
        sync_read.clearParam()
        return positions, velocities, loads

    def get_feedback(self):
        positions_raw, velocities_raw, loads_raw = self.get_feedback_raw()
        positions = [self.dxl_units_to_pos(pos) for pos in positions_raw]
        positions = [pos - offset for pos, offset in zip(positions, self.offset)]
        velocities = [self.dxl_units_to_vel(vel) for vel in velocities_raw]
        loads = [load/1000 for load in loads_raw]
        return positions, velocities, loads

    def get_error_string(self, hw_error_status):
        errors = []
        if hw_error_status & (1 << 0):
            errors.append("Input voltage")
        if hw_error_status & (1 << 2):
            errors.append("Overheat")
        if hw_error_status & (1 << 3):
            errors.append("Encoder")
        if hw_error_status & (1 << 4):
            errors.append("Electrical")
        if hw_error_status & (1 << 5):
            errors.append("Overload")
        return ", ".join(errors)

    def check_errors(self):
        sync_read = dynamixel_sdk.GroupSyncRead(self.port, self.packet, self.ADDR_HARDWARE_ERROR_STATUS, 1)
        for motor in self.motor_list:
            sync_read.addParam(motor['id'])
        self.sync_read_rxtx_retry(sync_read)
        errors = []
        for motor in self.motor_list:
            if sync_read.isAvailable(motor['id'], self.ADDR_HARDWARE_ERROR_STATUS, 1):
                data = sync_read.getData(motor['id'], self.ADDR_HARDWARE_ERROR_STATUS, 1)
                # if data & (0xFF - (1 << 5)) != 0: # ignore overload error
                if data != 0:
                    errors.append((motor['id'], data, f"motor {motor['id']}: 0x{data:02X} errors: {self.get_error_string(data)}"))
                    self.logger.error(f"Motor {motor['id']} has errors: {self.get_error_string(data)}")
            else:
                self.logger.error(f"Motor {motor['id']} not found in sync read")
                raise MotorControllerError(f"Motor {motor['id']} not found in sync read")
        return errors

    def _recover_once(self):
        self._reopen_port()
        for motor in self.motor_list:
            res, err = self.packet.reboot(self.port, motor['id'])
            self._check_result(res, err, motor['id'], "reboot")
            time.sleep(0.1)
        self.enable()
        errors = self.check_errors()
        if len(errors) > 0:
            raise MotorControllerError(f"Errors remain after recovery: {errors}")

    def recover_from_error(self, retry_interval=0.1):
        # Block until recovery succeeds
        attempt = 0
        while True:
            attempt += 1
            try:
                self._recover_once()
                self.logger.info("Motor recovery succeeded after %d attempt(s)", attempt)
                return
            except (MotorControllerError, OSError, serial.SerialException) as e:
                self.logger.warning("Motor recovery attempt %d failed: %s; retrying in %.3fs",
                                    attempt, e, retry_interval)
                time.sleep(retry_interval)

    def get_temperature(self):
        sync_read = dynamixel_sdk.GroupSyncRead(self.port, self.packet, self.ADDR_PRESENT_TEMPERATURE, 1)
        for motor in self.motor_list:
            sync_read.addParam(motor['id'])
        self.sync_read_rxtx_retry(sync_read)
        temperatures = []
        for motor in self.motor_list:
            if sync_read.isAvailable(motor['id'], self.ADDR_PRESENT_TEMPERATURE, 1):
                data = sync_read.getData(motor['id'], self.ADDR_PRESENT_TEMPERATURE, 1)
                temperatures.append(data)
            else:
                self.logger.error(f"Motor {motor['id']} not found in sync read")
                raise MotorControllerError(f"Motor {motor['id']} not found in sync read")
        sync_read.clearParam()
        return temperatures


if __name__ == "__main__":
    import json
    import sys
    import time
    # drv = MotorController(port=sys.argv[1], motor_list=[
    #     {'id': 10, 'min_position': -0.79, 'max_position': 0.79, 'offset': 0.79},
    #     {'id': 11, 'min_position': -0.79, 'max_position': 0.79, 'offset': 0.79},
    #     {'id': 20, 'min_position': -0.79, 'max_position': 0.79, 'offset': -0.79},
    #     {'id': 21, 'min_position': -0.79, 'max_position': 0.79, 'offset': 0.79},
    #     {'id': 30, 'min_position': -0.79, 'max_position': 0.79, 'offset': 0.79},
    #     {'id': 31, 'min_position': -0.79, 'max_position': 0.79, 'offset': 0.79},
    #     {'id': 40, 'min_position': -0.79, 'max_position': 0.79, 'offset': -0.79},
    #     {'id': 41, 'min_position': -0.79, 'max_position': 0.79, 'offset': -0.79},
    # ])
    cfg = json.load(open(sys.argv[1]))
    drv = MotorController(port=cfg['motor_port'], motor_list=cfg['motor_list'])
    drv.disable()
    drv.enable()
    while True:
        t_start = time.time()
        pos, vel, load = drv.get_feedback()
        print(pos)
        # time.sleep(0.01)
        drv.set_positions([np.sin(time.time())*0.8]*len(drv.motor_list))
        t_end = time.time()
        print(f"Time taken: {t_end - t_start:.3f}s")
        time.sleep(0.001)
