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
- no special tools required (only the standard hex drivers for M2, M2.5, and M3 screws which are 1.5mm, 2.0mm, and 2.5mm, respectively)

### Specs

- hip range +/- 45deg
- knee range +/- 70deg


### Bill of Materials

<p align="center">
  <img src="./imgs/robots.png" alt="Robots" width="600"/>
</p>

Top camera: [Logitech Brio 101](https://www.logitech.com/en-us/shop/p/brio-100-webcam) for tracking.

#### Embodied Ant

| Part Name                                 | Quantity | Notes                                                                                                        | US link (Price Nov'25)                                                                                     | EU link (Price Dec'25)                                                                                                                                                                                              |
|-------------------------------------------|----------|--------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Dynamixel XL430-W250-T                    | 8        | Main actuators  (incl.  180mm cable)                                                                         | [Robotis](https://www.robotis.us/dynamixel-xl430-w250-t/)    $220 (27.50 each)                             | [GenerationRobots](https://www.generationrobots.com/en/402823-dynamixel-xl430-w250-t-servomotor.html) €288 (36 each)                                                                                                |
| HN11-I101 Set                             | 8        | Idler bearing                                                                                                | [Robotis](https://www.robotis.us/hn11-i101-set/)             $64.4 (8.05 each)                             | [GenerationRobots](https://www.generationrobots.com/en/403206-hn11-i101-horn-set.html)  €68.80 (8.60 each)                                                                                                          |
| U2D2 Starter Set                          | 1        | Includes: USB to Dynamixel, Power Hub Board, 12V 5A Power Suppy                                              | [Robotis](https://www.robotis.us/dynamixel-starter-set-us/)  $68.66                                        | [GenerationRobots](https://www.generationrobots.com/en/403718-official-dynamixel-starter-set-eu.html)  €103.95                                                                                                      |
| Kakute H7 Mini / TBS Lucid Freestyle mini | 1        | Quadcopter flight controller used as IMU (Any Betaflight compatible autopilot with 20x20mm mouts will work)  | [getfpv](https://www.getfpv.com/tbs-lucid-freestyle-f4-mini-flight-controller-icm42688-20x20.html)  $51.99 | [drone-fpv-racer](https://www.drone-fpv-racer.com/en/lucid-freestyle-f435-icm42688p-flight-controller-by-tbs-13319.html)  €39.90                                                                                    |
| Cable Matters Ultra Mini USB Hub          | 1        | 4 Port USB Hub                                                                                               | [Amazon](https://www.amazon.com/dp/B00PHPWLPA/)           $15.49                                           | [Amazon](https://www.amazon.com/dp/B00PHPWLPA/)  €13.17                                                                                                                                                             |
| Short USB-A to USB-C Cable                | 1        | For autopilot (IMU)                                                                                          | [Amazon](https://www.amazon.com/dp/B01ASXBY62)            $9.49                                            | [Amazon](https://www.amazon.com/dp/B01ASXBY62)  €10.71                                                                                                                                                              |
| Short USB-A to micro-USB cable            | 1        | For Dynamixel U2D2                                                                                           | [Amazon](https://www.amazon.com/dp/B08BZD66H4?th=1)       $6.99                                            | [Amazon](https://www.amazon.com/dp/B08BZD66H4?th=1)  €5.09                                                                                                                                                          |
| USB-A extension cable                     | 1        | As tether for the robot, 10ft (~3m)                                                                          | [Amazon](https://www.amazon.com/dp/B07ZV6FHWF/)           $5.99                                            | [Amazon](https://www.amazon.com/dp/B07ZV6FHWF/)  €5.09                                                                                                                                                              |
| Screw M2x5mm with socket head             | 80       | Output shaft, 8 per motor, 3D print assembly                                                                 | [McMaster](https://www.mcmaster.com/91290A012/)           $18.48 (pack of 100)                             | [Microschroeven](https://www.microschroeven.nl/nl/inbus-clinderkop-din-912/roest-vrij-staal/m2/inbusbout-roest-vrij-staal-m2-x-5mm-per-10-stuks/a-237-20000066)  €10 (1.25 per pack of 10)                          |
| Screw M2.5x16mm with socket head          | 32       | motor mount, 4 per motor                                                                                     | [McMaster](https://www.mcmaster.com/91290a106/)           $12.81 (pack of 50)                              | [Microschroeven](https://www.microschroeven.nl/nl/inbus-clinderkop-din-912/roest-vrij-staal/m2-5/inbusbout-roest-vrij-staal-m2-5-x-16mm-per-10-stuks/a-722-20000067)  €5 (1.25 per pack of 10)                      |
| Screw M3x8mm with socket head             | 6        | U2D2 power board mount, IMU                                                                                  | [McMaster](https://www.mcmaster.com/91290A113/)           $12.82 (pack of 100)                             | [Microschroeven](https://www.microschroeven.nl/nl/inbus-clinderkop-din-912/roest-vrij-staal/m3/inbusbout-roest-vrij-staal-m3-x-8mm-per-10-stuks/a-522-20000068)  €0.80 (pack of 10)                                 |
| Nut M2                                    | 16       | 3D print assembly 4 per leg                                                                                  | [McMaster](https://www.mcmaster.com/91828A111/)           $6.14 (pack of 100)                              | [Microschroeven](https://www.microschroeven.nl/nl/moeren/normale-moeren-din-934/normale-roest-vrij-stalen-moeren/moer-roest-vrij-staal-m2-volgens-din934-per-10-stuks/a-1805-20000023)  €1.70 (0.85 per pack of 10) |
| Nut M3                                    | 2        | U2D2 power board mount                                                                                       | [McMaster](https://www.mcmaster.com/91828A211/)           $4.73 (pack of 100)                              | [Microschroeven](https://www.microschroeven.nl/nl/moeren/normale-moeren-din-934/normale-roest-vrij-stalen-moeren/moer-roest-vrij-staal-m3-volgens-din934-per-10-stuks/a-1161-20000023)  €0.80 (pack of 10)          |
| Aluminum Heatsink 6x20x20mm               | 4        | Recommended to improve cooling of knee actuators                                                             | [Amazon](https://www.amazon.com/dp/B08HLZWKYN)            $9.99 (pack of 6)                                | [Amazon DE](https://www.amazon.de/dp/B0CST4ZVRM/?th=1)  €13.03 (pack of 20)                                                                                                                                         |
| 3D Printed Parts                          | -        | STL files in `hardware/rev3`. Print all `leg` files 4x, others 1x.                                           | -                                                                                                          | -                                                                                                                                                                                                                   |
| **Est. total** (excl. shipping)           |          |                                                                                                              | **$507.98**                                                                                                | **€566.04**                                                                                                                                                                                                         |



#### Embodied Ant X

| Part Name                                 | Quantity | Notes                                                                                                        | US link (Price Nov'25)                                                                                     | EU link (Price Dec'25)                                                                                                                                                                                              |
|-------------------------------------------|----------|--------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Dynamixel XM430-W350-T                    | 4        | Main actuators  (incl.  180mm cable)                                                                         | [Robotis](https://www.robotis.us/dynamixel-xm430-w350-t/)  $1241.56 (310.39 each)                          | [GenerationRobots](https://www.generationrobots.com/en/402710-dynamixel-xm430-w350-t-servo.html ) €1269.60 (317.40 each)                                                                                            | 
| Dynamixel XL430-W250-T                    | 4        | Main actuators  (incl.  180mm cable)                                                                         | [Robotis](https://www.robotis.us/dynamixel-xl430-w250-t/)  $110 (27.50 each)                               | [GenerationRobots](https://www.generationrobots.com/en/402823-dynamixel-xl430-w250-t-servomotor.html) €144 (36 each)                                                                                                |
| HN11-I101 Set                             | 4        | Idler bearing                                                                                                | [Robotis](https://www.robotis.us/hn11-i101-set/)  $32.20 (8.05 each)                                       | [GenerationRobots](https://www.generationrobots.com/en/403206-hn11-i101-horn-set.html)  €34.40 (8.60 each)                                                                                                          |
| HN12-I101 Set                             | 4        | Idler bearing                                                                                                | [Robotis](https://www.robotis.us/hn12-i101-set/)  $81.88 (20.47 each)                                      | [GenerationRobots](https://www.generationrobots.com/en/402543-hn12-i101-horn-set.html)  €87.40 (21.85 each)                                                                                                         |
| U2D2 Starter Set                          | 1        | Includes: USB to Dynamixel, Power Hub Board, 12V 5A Power Suppy                                              | [Robotis](https://www.robotis.us/dynamixel-starter-set-us/)  $68.66                                        | [GenerationRobots](https://www.generationrobots.com/en/403718-official-dynamixel-starter-set-eu.html)  €103.95                                                                                                      |
| Kakute H7 Mini / TBS Lucid Freestyle mini | 1        | Quadcopter flight controller used as IMU (Any Betaflight compatible autopilot with 20x20mm mouts will work)  | [getfpv](https://www.getfpv.com/tbs-lucid-freestyle-f4-mini-flight-controller-icm42688-20x20.html)  $51.99 | [drone-fpv-racer](https://www.drone-fpv-racer.com/en/lucid-freestyle-f435-icm42688p-flight-controller-by-tbs-13319.html)  €39.90                                                                                    |
| Cable Matters Ultra Mini USB Hub          | 1        | 4 Port USB Hub                                                                                               | [Amazon](https://www.amazon.com/dp/B00PHPWLPA/)   $15.49                                                   | [Amazon](https://www.amazon.com/dp/B00PHPWLPA/)  €13.17                                                                                                                                                             |
| Short USB-A to USB-C Cable                | 1        | For autopilot (IMU)                                                                                          | [Amazon](https://www.amazon.com/dp/B01ASXBY62)   $9.49                                                     | [Amazon](https://www.amazon.com/dp/B01ASXBY62)  €10.71                                                                                                                                                              |
| Short USB-A to micro-USB cable            | 1        | For Dynamixel U2D2                                                                                           | [Amazon](https://www.amazon.com/dp/B08BZD66H4?th=1)   $6.99                                                | [Amazon](https://www.amazon.com/dp/B08BZD66H4?th=1)  €5.09                                                                                                                                                          |
| USB-A extension cable                     | 1        | As tether for the robot, 10ft (~3m)                                                                          | [Amazon](https://www.amazon.com/dp/B07ZV6FHWF/)   $5.99                                                    | [Amazon](https://www.amazon.com/dp/B07ZV6FHWF/)  €5.09                                                                                                                                                              |
| Screw M2x3mm with socket head             | 32       | Output shaft, 8 per motor                                                                                    | [McMaster](https://www.mcmaster.com/91290a010/)   $15.62 (pack of 50)                                      | [Microschroeven](https://www.microschroeven.nl/nl/inbus-clinderkop-din-912/roest-vrij-staal/m2/inbusbout-roest-vrij-staal-m2-x-3mm-per-10-stuks/a-236-20000066)  €5.40 (1.35 per pack of 10)                        |
| Screw M2x5mm with socket head             | 48       | Output shaft, 8 per motor, 3D print assembly                                                                 | [McMaster](https://www.mcmaster.com/91290A012/)   $18.48 (pack of 100)                                     | [Microschroeven](https://www.microschroeven.nl/nl/inbus-clinderkop-din-912/roest-vrij-staal/m2/inbusbout-roest-vrij-staal-m2-x-5mm-per-10-stuks/a-237-20000066)  €6.25 (1.25 per pack of 10)                        |
| Screw M2.5x16mm with socket head          | 32       | motor mount, 4 per motor                                                                                     | [McMaster](https://www.mcmaster.com/91290a106/)   $12.81 (pack of 50)                                      | [Microschroeven](https://www.microschroeven.nl/nl/inbus-clinderkop-din-912/roest-vrij-staal/m2-5/inbusbout-roest-vrij-staal-m2-5-x-16mm-per-10-stuks/a-722-20000067)  €5 (1.25 per pack of 10)                      |
| Screw M3x8mm with socket head             | 6        | U2D2 power board mount + IMU                                                                                 | [McMaster](https://www.mcmaster.com/91290A113/)   $12.82 (pack of 100)                                     | [Microschroeven](https://www.microschroeven.nl/nl/inbus-clinderkop-din-912/roest-vrij-staal/m3/inbusbout-roest-vrij-staal-m3-x-8mm-per-10-stuks/a-522-20000068)  €0.80 (pack of 10)                                 |
| Nut M2                                    | 16       | 3D print assembly 5 per leg                                                                                  | [McMaster](https://www.mcmaster.com/91828A111/)   $6.14 (pack of 100)                                      | [Microschroeven](https://www.microschroeven.nl/nl/moeren/normale-moeren-din-934/normale-roest-vrij-stalen-moeren/moer-roest-vrij-staal-m2-volgens-din934-per-10-stuks/a-1805-20000023)  €1.70 (0.85 per pack of 10) |
| Nut M3                                    | 2        | U2D2 power board mount                                                                                       | [McMaster](https://www.mcmaster.com/91828A211/)   $4.73 (pack of 100)                                      | [Microschroeven](https://www.microschroeven.nl/nl/moeren/normale-moeren-din-934/normale-roest-vrij-stalen-moeren/moer-roest-vrij-staal-m3-volgens-din934-per-10-stuks/a-1161-20000023)  €0.80 (pack of 10)          |
| Aluminum Heatsink 6x20x20mm               | 4        | Recommended to improve cooling of knee actuators                                                             | [Amazon](https://www.amazon.com/dp/B08HLZWKYN)    $9.99 (pack of 6)                                        | [Amazon DE](https://www.amazon.de/dp/B0CST4ZVRM/?th=1)  €13.03 (pack of 20)                                                                                                                                         |
| On-board camera                           | 1        | Logitech Brio 101                                                                                            | [Amazon](https://www.amazon.com/dp/B0BXGFFSL1)    $29.99                                                   | [Amazon](https://www.amazon.com/dp/B0BXGFFSL1)     €31.59                                                                                                                                                           |
| 3D Printed Parts                          | -        | STL files in `hardware/rev3`. Print all `leg` files 4x, others 1x.                                           | -                                                                                                          | -                                                                                                                                                                                                                   |
| **Est. total** (excl. shipping)           |          |                                                                                                              | **$1734.83**                                                                                               | **€1777.88**                                                                                                                                                                                                        |


Note: The reason why more screw types were added compared to the cheaper version is because the 3D print attachment with the expensive motors (XM430-W350) needs M2 x 3mm, not M2 x 5mm.


### Dynamixel Setup (before assembling the robot!)

The Dynamixels should be configured to 1Mbaud and have their IDs changed to the following:

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

## Run SAC

for simulation:
```
cd agents/sac/
./run.sh sim
```

for hardware:
```
cd agents/sac/
./run.sh hw
```

## Frequently Asked Questions (FAQ)

### Q: The position of the ant flickers. What should I do?

The ArUco system is designed to be quite robust. If you encounter problems, make sure the markers are clearly visible and the camera exposure is configured properly (you can adjust exposure using LogiTune). 
Here are some suggestions:
* Depending on your environment, you may need to disable auto-exposure for more consistent detection.
* For the best performance, you should have the camera looking down at the playground.
* The origin marker should be mounted flat. Any warping can cause issues.
* Make sure to plot all system inputs and outputs to verify that the signals are clean. Learning from noisy or faulty signals can lead to poor results.

### Q: What’s the recommended way to specify a /dev/ device path in a JSON file on Ubuntu?

Use the persistent device path, for example:
```
/dev/serial/by-id/YYY
```

### Q: Will the screws become loose over time due to vibrations?

Yes, it can happen. For this, it is recommended to use Loctite Threadlocker 222. 
Ensure you don't apply too much because it can leak under the motor head and cause clogging.

### Q: How can we increase the friction on the lower legs?

We recommend dipping the lower portion of the legs into Plasti Dip 
([US link](https://shop.plastidip.com/products/plasti-dip-can?variant=49236227129646), [EU link](https://www.plasti-dip.nl/shop/plasti-dip-mat-429ml-3/)) 
rubber coating 2–3 times or 3D printing socks out of TPU.


