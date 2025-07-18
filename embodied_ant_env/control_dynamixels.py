import os
import math
import time
import json
import numpy as np
from dynamixel_sdk import * # Uses Dynamixel SDK library

def enable_torque(motor_id: int) -> bool:
    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, motor_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
    if dxl_comm_result != COMM_SUCCESS:
        print("Failed to enable torque for motor %d: %s" % (motor_id, packetHandler.getTxRxResult(dxl_comm_result)))
        return False
    elif dxl_error != 0:
        print("Failed to enable torque for motor %d: %s" % (motor_id, packetHandler.getRxPacketError(dxl_error)))
        return False
    else:
        print("Motor %d torque enabled successfully" % motor_id)
        return True


def disable_torque(motor_id: int) -> None:
    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, motor_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
    if dxl_comm_result != COMM_SUCCESS:
        print("Failed to disable torque for motor %d: %s" % (motor_id, packetHandler.getTxRxResult(dxl_comm_result)))
    elif dxl_error != 0:
        print("Failed to disable torque for motor %d: %s" % (motor_id, packetHandler.getRxPacketError(dxl_error)))
    else:
        print("Motor %d torque disabled successfully" % motor_id)

def angle_rad_to_dxl_units(angle_rad):
    """
    Convert angle in radians (e.g., -pi to +pi) to Dynamixel raw units.
    :param angle: angle in radians, between -pi and +pi
    :return: Dynamixel raw units (0-4095)
    """
    return int(round((angle_rad * 4096.0 / (2 * np.pi) + 2048.0))) # 2048 is the center of the range.

def read_present_position_dxl_units(motor_id: int) -> int:
    ''' Read position in Dynamixel units from a motor. '''
    dxl_present_position, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, motor_id, ADDR_PRESENT_POSITION)
    if dxl_comm_result != COMM_SUCCESS:
        print("Failed to read present position for motor %d: %s" % (motor_id, packetHandler.getTxRxResult(dxl_comm_result)))
        return None
    elif dxl_error != 0:
        print("Failed to read present position for motor %d: %s" % (motor_id, packetHandler.getRxPacketError(dxl_error)))
        return None
    return dxl_present_position

def read_present_position_radians(motor_id: int) -> float:
    ''' Read present position in radians from a motor. '''
    position_units = read_present_position_dxl_units(motor_id)
    if position_units is not None:
        # Convert from Dynamixel units (0-4095) to radians (0-2π).
        return (position_units / 4096.0) * 2 * np.pi
    return None

def generate_ant_position_control_gait(t, freq=1.5, amp_hip=0.2, amp_ankle=1.2):
    """
    Generate joint positions for MuJoCo Ant to move forward.
    :param t: time in seconds
    :param freq: frequency of gait (Hz)
    :param amp_hip: amplitude for hip joints
    :param amp_ankle: amplitude for ankle joints
    :return: ndarray of shape (8,) representing desired joint positions
    """

    phase = 2 * np.pi * freq * t

    # Define legs in diagonal pairs
    hip_1 = amp_hip * np.sin(phase)
    ankle_1 = amp_ankle * np.sin(phase) if np.sin(phase) <= 0 else 0

    hip_2 = amp_hip * np.sin(phase + np.pi)
    ankle_2 = amp_ankle * np.cos(phase + np.pi) if np.cos(phase + np.pi) <= 0 else 0

    hip_3 = amp_hip * np.sin(phase + np.pi)
    ankle_3 = amp_ankle * np.cos(phase + np.pi) if np.cos(phase + np.pi) <= 0 else 0

    hip_4 = amp_hip * np.sin(phase)
    ankle_4 = amp_ankle * np.cos(phase) if np.cos(phase) <= 0 else 0

    return np.array([hip_1, ankle_1, hip_2, ankle_2, hip_3, ankle_3, hip_4, ankle_4])

if __name__ == "__main__":

    if os.name == 'nt':
        import msvcrt
        def getch():
            return msvcrt.getch().decode()
    else:
        import sys, tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        def getch():
            try:
                tty.setraw(sys.stdin.fileno())
                ch = sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            return ch

    MY_DXL = 'X_SERIES'
    if MY_DXL == 'X_SERIES' or MY_DXL == 'MX_SERIES':
        ADDR_TORQUE_ENABLE          = 64
        ADDR_GOAL_POSITION          = 116
        ADDR_PRESENT_POSITION       = 132
        DXL_MINIMUM_POSITION_VALUE  = 0         # Refer to the Minimum Position Limit of product eManual
        DXL_MAXIMUM_POSITION_VALUE  = 4096      # Refer to the Maximum Position Limit of product eManual
        BAUDRATE                    = 57600

    else:
        print("Invalid DYNAMIXEL model")
        quit()

    PROTOCOL_VERSION = 2.0

    # Load config file.
    print("Loading config file...")
    config = json.loads(open('config_motors.json').read())

    # Get the motor IDs.
    motor_list = config.keys()

    for motor in motor_list:
        print(motor)
        print(config[motor])

    # Use the actual port assigned to the U2D2.
    # ex) Windows: "COM*", Linux: "/dev/ttyUSB*", Mac: "/dev/tty.usbserial-*"
    DEVICE_NAME                  = '/dev/tty.usbserial-FT7WBGG8'

    TORQUE_ENABLE               = 1     # Value for enabling the torque
    TORQUE_DISABLE              = 0     # Value for disabling the torque

    # Initialize PortHandler instance
    # Set the port path
    # Get methods and members of PortHandlerLinux or PortHandlerWindows
    portHandler = PortHandler(DEVICE_NAME)

    # Initialize PacketHandler instance
    # Set the protocol version
    # Get methods and members of Protocol1PacketHandler or Protocol2PacketHandler
    packetHandler = PacketHandler(PROTOCOL_VERSION)

    # Open port
    if portHandler.openPort():
        print("Succeeded to open the port")
    else:
        print("Failed to open the port")
        print("Press any key to terminate...")
        getch()
        quit()

    # Set port baudrate
    if portHandler.setBaudRate(BAUDRATE):
        print("Succeeded to change the baudrate")
    else:
        print("Failed to change the baudrate")
        print("Press any key to terminate...")
        getch()
        quit()

    # Enable torque for both motors
    print("Enabling torque for all motors...")
    for motor in motor_list:
        if not enable_torque(config[motor]['ID']):
            print("Failed to enable torque for motor %s. Exiting..." % motor)
            # portHandler.closePort()
            # quit()

    print("Starting sinusoidal motion...")

    try:
        save_current_position = False
        t = 0.0
        time_ = 0.0
        DT = 0.01
        
        while True:
            # Check for ESC key to stop
            if os.name == 'nt':
                if msvcrt.kbhit():
                    if msvcrt.getch() == b'\x1b':
                        break
            else:
                # For Unix-like systems, we'll use a timeout approach
                pass

            # Read and display present positions in radians.
            if save_current_position == False:
                initial_pos_radians_list = []
                for motor in motor_list:
                    pos_radians = read_present_position_radians(config[motor]['ID'])
                    initial_pos_radians_list.append(pos_radians)
                    print(f"Motor {motor} initial position degrees: {np.rad2deg(pos_radians)}")
                print('Saving initial positions...')
                print(initial_pos_radians_list)

                save_current_position = True
            
            for motor in motor_list:
                if motor == 'ANKLE_4':
                    pos_units = read_present_position_dxl_units(config[motor]['ID'])
                    print(f"Motor {motor} position units: {pos_units}")

            # Calculate current time for sinusoidal motion.
            t = t + DT
            position_control_gait = generate_ant_position_control_gait(t)

            # Prepare all motor commands first.
            motor_commands = []
            for i, motor in enumerate(motor_list):
                # Limit the position to the range of the motor.
                clipped_position = np.clip(position_control_gait[i], config[motor]['MIN_POSITION'], config[motor]['MAX_POSITION'])
                desired_position = clipped_position + initial_pos_radians_list[i]
                motor_commands.append((config[motor]['ID'], desired_position))

            # Send all commands simultaneously using sync write.
            if motor_commands:
                # Create sync write packet for goal position
                sync_write = GroupSyncWrite(portHandler, packetHandler, ADDR_GOAL_POSITION, 4)

                # Add all motor positions to the sync write packet.
                for motor_id, position in motor_commands:
                    position_units = angle_rad_to_dxl_units(position)
                    sync_write.addParam(motor_id, [position_units & 0xFF, (position_units >> 8) & 0xFF,
                                                   (position_units >> 16) & 0xFF, (position_units >> 24) & 0xFF])

                # Execute sync write.
                dxl_comm_result = sync_write.txPacket()
                if dxl_comm_result != COMM_SUCCESS:
                    print("Failed to sync write: %s" % packetHandler.getTxRxResult(dxl_comm_result))

                # Clear sync write parameter storage
                sync_write.clearParam()

            time_ += DT

    except KeyboardInterrupt:
        print("\nStopping sinusoidal motion...")

    # Disable Dynamixel Torque for both motors
    print("Disabling torque for both motors...")
    for motor in motor_list:
        disable_torque(config[motor]['ID'])

    # Close port
    portHandler.closePort()
    print("Port closed. Program terminated.")