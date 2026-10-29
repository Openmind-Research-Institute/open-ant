import sys
import time
from python_st3215 import ST3215, ServoNotRespondingError

port = sys.argv[1]
if len(sys.argv) > 2:
   baudrate = int(sys.argv[3])
else:
   baudrate = 1000000 # default baud rate out of the box for ST3125

with ST3215(port, baudrate) as bus:
   
   found_ids = bus.list_servos()
   if len(found_ids) == 0:
      raise Exception(f"Expected at least 1 ST3215, found {len(found_ids)}")

   for current_id in found_ids:
      servo = bus.wrap_servo(current_id)
      print(f"Found ST3215 ID: {current_id}")

      print(f"Setting zero for motor {current_id}")

      #servo.sram.torque_enable()
      #servo.sram.write_target_location(2048)
      #time.sleep(0.05)
      #while servo.sram.is_moving():
      #   time.sleep(0.05)

      current_pos = servo.sram.read_current_location()
      print(f"Motor is at position {current_pos}")
      
      servo.sram.torque_disable()
