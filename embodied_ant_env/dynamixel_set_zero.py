import sys
import dynamixel_sdk

port = sys.argv[1]
if len(sys.argv) > 2:
    baudrate = int(sys.argv[2])
else:
    baudrate = 57600


port_handler = dynamixel_sdk.PortHandler(port)
if not port_handler.openPort():
    raise Exception(f"Failed to open port {port}")
if not port_handler.setBaudRate(baudrate):
    raise Exception(f"Failed to set baudrate {baudrate} on port {port}")
packet_handler = dynamixel_sdk.PacketHandler(2.0)

ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116

for dxl_id in range(0, 253):
    dxl_model_number, dxl_comm_result, dxl_error = packet_handler.ping(port_handler, dxl_id)
    if dxl_comm_result == dynamixel_sdk.COMM_SUCCESS and dxl_error == 0:
        print(f"Found Dynamixel ID: {dxl_id}, Model Number: {dxl_model_number}")

        print(f"Setting zero for motor {dxl_id}")

        res, err = packet_handler.write1ByteTxRx(port_handler, dxl_id, ADDR_TORQUE_ENABLE, 1)
        if res != dynamixel_sdk.COMM_SUCCESS:
            raise Exception(f"Failed to enable torque for motor {dxl_id}: {packet_handler.getTxRxResult(res)}")

        res, err = packet_handler.write4ByteTxRx(port_handler, dxl_id, ADDR_GOAL_POSITION, 0)
        if res != dynamixel_sdk.COMM_SUCCESS:
            raise Exception(f"Failed to set goal position for motor {dxl_id}: {packet_handler.getTxRxResult(res)}")

port_handler.closePort()
