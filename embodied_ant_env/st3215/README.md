ST3215
-------

This remix of the open-ant is using Feetech ST3215 smart servos that can be controlled via the STS protocol. Waveshare ST3215 servos are a drop-in replacement even if they come with a different USB-to-TTL bus and power IO board. Specifically, this remix is using the STS3215-12V, which require a 12V power supply with sufficent amperage to drive all 8 servos. The STS3215-12 compares to the Dynamixel XM430-W350-T and XC430-W240-T as follows with respect to hardware specifications:

|        Specification        |  ST3215-12V  | XM430-W350-T | XC430-W240-T |
|:---------------------------:|:------------:|:------------:|:------------:|
|    Max stall (Nm) @ 12V     |     2.9      |     4.1      |     1.9      |
| Estimated rated torque (Nm) |     0.98     |     0.82     |     0.38     |
|    Dimensions (WxHxD mm)    | 24.7x45.2x35 | 28.5x46.5x34 | 28.5x46.5x34 |
|     Max Temperature (C)     |      70      |      80      |      80      |
|     No Load Speed (RPM)     |      45      |      46      |      70      |
|         Weight (g)          |      55      |      82      |      65      |

Source: [Feetech ST3215-12V](https://files.seeedstudio.com/products/Feetech/108090003_FEETECH_ST-3215-C047-Datasheet.pdf)

This remix uses ST3215 for both hip and knee motors.

## SDK

The `python-st3125` module is used to reimplement functionality provided by the Dynamixel SDK within the Python code for the open-ant. For example, `st3125_change_id.py` allows changing of the servo motor IDs in the same way as `dynamixel_change_id.py`.

## Baud Rate

The ST3125-12Vs do not need to be configured to use a higher baud rate, because they come out of the box configured for the baud rate of 1Mbps.
