import sys
from python_st3215 import ST3215, ServoNotRespondingError

port = sys.argv[1]
new_id = int(sys.argv[2])
if new_id < 0 or new_id > 253:
   raise Exception(f"Expected new ST3125 servo ID in range [0,253], given {new_id}")
   
if len(sys.argv) > 3:
   baudrate = int(sys.argv[3])
else:
   baudrate = 1000000 # default baud rate out of the box for ST3125

with ST3215(port, baudrate) as bus:
   
   found_ids = bus.list_servos()
   if len(found_ids) != 1:
      raise Exception(f"Expected 1 ST3215, found {len(found_ids)}")

   servo = bus.wrap_servo(found_ids[0])
   current_id = servo.eeprom.read_id()
   print(f"Found ST3215 ID: {current_id}")

   if current_id == new_id:
      raise Exception(f"Excepted different ID for motor {current_id}")

   servo.sram.unlock()
   servo.eeprom.write_id(new_id)
   servo.sram.lock()

   new_servo = bus.wrap_servo(new_id)
   current_id = new_servo.eeprom.read_id()
   if current_id != new_id:
      raise Exception(f"Failed to change ID for motor {found_ids[0]}")
   else:
      print(f"Changed ID for motor {found_ids[0]} to {current_id}")
   
