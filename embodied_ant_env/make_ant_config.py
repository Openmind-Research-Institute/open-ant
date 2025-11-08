import sys
import json
import math
from motor_controller import MotorController


def make_ant_motor_config(motor_port, baudrate=1000000):
    motor_list = []
    for idx, range_deg in [(10, 45), (11, 70), (20, 45), (21, 70), (30, 45), (31, 70), (40, 45), (41, 70)]:
        motor_list.append({
            'id': idx,
            'min_position': -math.radians(range_deg),
            'max_position': math.radians(range_deg),
            'offset': math.radians(45) # all motors are mounted at a multiple of 45deg
        })

    ctrl = MotorController(motor_port, motor_list, baudrate)
    zero_pos = ctrl.get_feedback()[0]
    print(zero_pos)
    for i, pos in enumerate(zero_pos):
        # find the multiple of 45deg that gets pos closest to 0
        offset = round(pos / (math.pi / 4)) * (math.pi / 4)
        motor_list[i]['offset'] += offset

    config = {
        "motor_port": motor_port,
        "motor_list": motor_list,
        "motor_baudrate": baudrate,
    }
    return config

if __name__ == "__main__":
    port = sys.argv[1]
    tag_id = sys.argv[2]
    input("move ant in neutral pose (legs straight out) and press enter to continue")
    file_name = f"ant{tag_id}.json"
    config = make_ant_motor_config(port)
    config['imu_port'] = "/dev/ttyUSB1"
    config['onboard_camera_id'] = 1
    config['camera_id'] = 0
    config['camera_fov_diagonal_deg'] = 58
    config['camera_tag_sizes'] = {'origin': 0.1, 'body': 0.045}
    config['camera_tag_ids'] = {'origin': 0, 'body': int(tag_id)}
    with open(file_name, "w") as f:
        json.dump(config, f, indent=2)
