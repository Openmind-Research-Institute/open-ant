import sys
import dynamixel_sdk

BAUDRATE_ADDRESS = 8

port = sys.argv[1]
new_baudrate = int(sys.argv[2])
if len(sys.argv) > 3:
    current_baudrate = int(sys.argv[3])
else:
    current_baudrate = 57600


port_handler = dynamixel_sdk.PortHandler(port)
if not port_handler.openPort():
    raise Exception(f"Failed to open port {port}")
if not port_handler.setBaudRate(current_baudrate):
    raise Exception(f"Failed to set baudrate {current_baudrate} on port {port}")
packet_handler = dynamixel_sdk.PacketHandler(2.0)


found_ids = []
for dxl_id in range(0, 253):
    dxl_model_number, dxl_comm_result, dxl_error = packet_handler.ping(port_handler, dxl_id)
    if dxl_comm_result == dynamixel_sdk.COMM_SUCCESS and dxl_error == 0:
        print(f"Found Dynamixel ID: {dxl_id}, Model Number: {dxl_model_number}")
        found_ids.append(dxl_id)

if len(found_ids) == 0:
    print(f"No Dynamixels found on port {port}")
    exit(1)

def baudrate_to_number(baudrate):
    baudrate_table = {
        9600:   0,
        57600:  1,
        115200: 2,
        1000000: 3,
        2000000: 4,
        3000000: 5,
        4000000: 6,
        4500000: 7
    }
    if baudrate in baudrate_table:
        return baudrate_table[baudrate]
    else:
        raise ValueError(f"Baudrate {baudrate} is not supported")


for dxl_id in found_ids:
    res, err = packet_handler.write1ByteTxRx(port_handler, dxl_id, BAUDRATE_ADDRESS, baudrate_to_number(new_baudrate))
    if res != dynamixel_sdk.COMM_SUCCESS:
        raise Exception(f"Failed to change baudrate for motor {dxl_id}: {packet_handler.getTxRxResult(res)}")
    else:
        print(f"Changed baudrate for motor {dxl_id} to {new_baudrate}")

port_handler.closePort()
