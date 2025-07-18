# Embodied Ant Environment





## Hardware

The hardware is designed to be easy to build and use:
- single computer operation via USB, python only, minimal dependencies (no embedded firmware / OS, networking, ROS...)
- no battery: continuous operation with wall adapter
- all COTS parts + 3D printed parts
- no soldering required
- no special tools required (only a screwdriver for M2 and M2.5)



### Dynamixel Setup

The following command will change the baudrate of all motors on the port to 1Mbaud.
```
python3 dynamixel_change_baud.py /dev/tty.usbserial-FT7WBGG8 1000000
```

### Misc

if git push fails when adding large (~10MB) commits, try:
```
git config --global http.postBuffer 1048576000
```
