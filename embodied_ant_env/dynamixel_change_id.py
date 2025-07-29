import sys
import dynamixel_sdk

port = sys.argv[1]
new_id = int(sys.argv[2])
if len(sys.argv) > 3:
    baudrate = int(sys.argv[3])
else:
    baudrate = 57600


port_handler = dynamixel_sdk.PortHandler(port)
if not port_handler.openPort():
    raise Exception(f"Failed to open port {port}")
if not port_handler.setBaudRate(baudrate):
    raise Exception(f"Failed to set baudrate {baudrate} on port {port}")
packet_handler = dynamixel_sdk.PacketHandler(2.0)

ADDR_ID = 7

found_ids = []
for dxl_id in range(0, 253):
    dxl_model_number, dxl_comm_result, dxl_error = packet_handler.ping(port_handler, dxl_id)
    if dxl_comm_result == dynamixel_sdk.COMM_SUCCESS and dxl_error == 0:
        print(f"Found Dynamixel ID: {dxl_id}, Model Number: {dxl_model_number}")
        found_ids.append(dxl_id)

if len(found_ids) != 1:
    raise Exception(f"Expected 1 Dynamixel, found {len(found_ids)}")

res, err = packet_handler.write1ByteTxRx(port_handler, found_ids[0], ADDR_ID, new_id)
if res != dynamixel_sdk.COMM_SUCCESS:
    raise Exception(f"Failed to change ID for motor {found_ids[0]}: {packet_handler.getTxRxResult(res)}")
else:
    print(f"Changed ID for motor {found_ids[0]} to {new_id}")

port_handler.closePort()
