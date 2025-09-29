# Embodied Ant

<!-- ![image info](./imgs/traj_ppo.gif) -->

<p align="center">
  <img src="./imgs/traj_ppo.gif" alt="Ant walking demo" width="600"/>
</p>

## Hardware

The hardware is designed to be easy to build and use:
- single computer operation via USB, python only, minimal dependencies (no embedded firmware / OS, networking, ROS...)
- no battery: continuous operation with wall adapter
- all COTS parts + 3D printed parts
- no soldering required
- no special tools required (only a screwdriver for M2, M2.5, M3)

### Specs

- hip range +/- 45deg
- knee range +/- 70deg


### Bill of Materials

| Part Name                | Quantity | Notes                        | Link                                      |
|--------------------------|----------|------------------------------|-------------------------------------------|
| Dynamixel XL430-W250-T   | 8        | Main actuators  (incl.  180mm cable)              | [Robotis](https://www.robotis.us/dynamixel-xl430-w250-t/) |
| HN11-I101 Set            | 8        | Idler bearing                | [Robotis](https://www.robotis.us/hn11-i101-set/) |
| U2D2 Starter Set         | 1        | Includes: USB to Dynamixel, Power Hub Board, 12V 5A Power Suppy    | [Robotis](https://www.robotis.us/dynamixel-starter-set-us/) |
| Kakute H7 Mini / TBS Lucid Freestyle mini | 1 | Quadcopter flight controller used as IMU (Any Betaflight compatible autopilot with 20x20mm mouts will work) | [getfpv](https://www.getfpv.com/tbs-lucid-freestyle-f4-mini-flight-controller-icm42688-20x20.html)
| Cable Matters Ultra Mini USB Hub | 1 | 4 Port USB Hub               | [Amazon](https://www.amazon.com/dp/B00PHPWLPA/) |
| short USB-A to USB-C Cable | 1      | For autopilot (IMU)          | [Amazon](https://www.amazon.com/dp/B01ASXBY62) |
| short USB-A to miro-USB cable | 1   | For Dynamixel U2D2           | [Amazon](https://www.amazon.com/dp/B08BZD66H4?th=1)
| USB-A extension cable    | 1       | As tether for the robot       | [Amazon](https://www.amazon.com/dp/B07ZV6FHWF/)
| Screw M2x5mm             | 64      | Output shaft, 8 per motor     | [McMaster](https://www.mcmaster.com/91290A012/)
| Screw M2x12mm            | 24      | 3D print assembly 5 per leg + IMU | [McMaster](https://www.mcmaster.com/91290A019/)
| Screw M2.5x16mm          | 32      | motor mount, 4 per motor      | [McMaster](https://www.mcmaster.com/91290a106/)
| Nut M2                   | 20      | 3D print assembly 5 per leg   | [McMaster](https://www.mcmaster.com/91828A111/)
| Screw M3x8mm             | 2       | U2D2 power board mount        | [McMaster](https://www.mcmaster.com/91290A113/)
| Nut M3                   | 2       | U2D2 power board mount        | [McMaster](https://www.mcmaster.com/91828A211/)
| 3D Printed Parts         | -       | STL files in `hardware/`      | -                                         |



### Dynamixel Setup (before assembling the robot!)

The Dynamixels should be configured to 1Mbaud, set to zero position (for assembly), and have their IDs changed to the following:

| Motor Position      | Motor ID |
|---------------------|----------|
| Rear Right Hip      |   10     |
| Rear Right Knee     |   11     |
| Front Right Hip     |   20     |
| Front Right Knee    |   21     |
| Front Left Hip      |   30     |
| Front Left Knee     |   31     |
| Rear Left Hip       |   40     |
| Rear Left Knee      |   41     |

Use the following script to change the IDs of the motors, connecting one at a time.
```
python3 embodied_ant_env/dynamixel_change_id.py /dev/tty.usbserial-XXXXXXX <NEW_ID> 57600
```

When done, the following command will change the baudrate of all connected motors on the port to 1Mbaud.
```
python3 embodied_ant_env/dynamixel_change_baud.py /dev/tty.usbserial-XXXXXXX 1000000
```

Finally, use the following script to move the motors to their zero position.
```
python3 embodied_ant_env/dynamixel_set_zero.py /dev/tty.usbserial-XXXXXXX 1000000
```


## Software Setup

Create a virtual environment and install the dependencies. (python >= 3.10)
```
python3.12 -m venv ant_env
source ant_env/bin/activate
pip install -r requirements.txt
```

To create a new config file, run:
```
python3 embodied_ant_env/make_ant_config.py /dev/tty.usbserial-XXXXXXX <APRIL_TAG_ID>
```
which will create a new config file `ant<APRIL_TAG_ID>.json` in the current directory.

Next, edit the config file to specify imu port, camera id and fov.

### Misc

if git push fails when adding large (~10MB) commits, try:
```
git config --global http.postBuffer 1048576000
```

### Possible signals

- Joint position
- Joint velocity
- Body angular rate
- Inertial up in body
- Commanded velocity in x and y

For reward: position of the body in x and y, and the angle of the body.


## Run the simulation

```
cd sim
python3 ant_mujoco.py
```


## Frequently Asked Questions (FAQ)

### Q: The position of the ant flickers. What should I do?

The ArUco system is designed to be quite robust. If you encounter problems, make sure the markers are clearly visible and the camera exposure is configured properly (you can adjust exposure using LogiTune).
Depending on your environment, you may need to disable auto-exposure for more consistent detection.