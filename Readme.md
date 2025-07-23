# Embodied Ant

![image info](./imgs/ant_setup.png)

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


### Bill of Materials

| Category     | Item                                | Price (USD) | Quantity | Total (USD) |
|--------------|-------------------------------------|-------------|----------|-------------|
| Electronics  | Dynamixels XL430-W250-T             | 23.90       | 8        | 191.20      |
|              | HN11-I101 Set                       | 7.70        | 8        | 61.60       |
|              | Dynamixel starter kit               | 65.67       | 1        | 65.67       |
|              | Webcam                              | 100.00*     | 1        | 100.00      |
|              | Autopilot                           | 40.00*      | 1        | 40.00       |
|              | USB Extension cable (10 feet)       | 10.22       | 1        | 10.22       |
|              | Mini 4 port USB hub                 | 15.49       | 1        | 15.49       |
|              | Robot Cable-X4P 180mm (10pcs)       | 22.22       | 1        | 22.22       |
|              | Robot Cable-X3P 180mm (10 pcs)      | 20.90       | 1        | 20.90       |
| Mechanics    | 3D print material                   | 50.00*       | 1        | 50.00       |
|              | Screws M2.5 - set of 100            | 14.00*       | 1        | 14.00       |
|              | Screws M2 - set of 100              | 14.00*       | 1        | 14.00       |
|              | **Total**              |             |          | **605.30**  |


* approximated


Developers: Sorina Lupu (sorina.lupu@openmindresearch.org) and Patrick Spieler (patrick.spieler@me.com)


### Possible signals

- Joint position
- Joint velocity
- Body angular rate
- Inertial up in body
- Commanded velocity in x and y

For reward: position of the body in x and y, and the angle of the body.