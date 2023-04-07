
import time
s_time=1000

def gardeAVous(Arm):
    Arm.Arm_serial_servo_write(1, 90, s_time)
    time.sleep(0.01)
    Arm.Arm_serial_servo_write(2, 90, s_time)
    time.sleep(0.01)
    Arm.Arm_serial_servo_write(3, 90, s_time)
    time.sleep(0.01)
    Arm.Arm_serial_servo_write(4, 90, s_time)
    time.sleep(0.01)
    Arm.Arm_serial_servo_write(5, 90, s_time)
    time.sleep(0.01)
    Arm.Arm_serial_servo_write(6, 40, s_time)
    time.sleep(0.01)

def allerADistance(Arm, i, angle) :
    if i == 0 :
        #position Extreme = E
        gardeAVous(Arm)
        time.sleep(1)
        Arm.Arm_serial_servo_write6_array([angle,0,90,68,270,170],s_time)
        
    if i == 1 :
        #position E-1 cm
        gardeAVous(Arm)
        time.sleep(1)
        Arm.Arm_serial_servo_write6_array([angle,15,69,68,270,170],s_time)
        
    if i == 2 :
        #position E-2
        gardeAVous(Arm)
        time.sleep(1)
        Arm.Arm_serial_servo_write6_array([angle,22,60,65,270,170],s_time)
        
    if i == 3 :
        #position E-3
        gardeAVous(Arm)
        time.sleep(1)
        Arm.Arm_serial_servo_write6_array([angle,28,55,60,270,170],s_time)
        
    if i == 4 :
        #position E-4
        gardeAVous(Arm)
        time.sleep(1)
        Arm.Arm_serial_servo_write6_array([angle,34,50,57,270,170],s_time)
        
    if i == 5 :
        #position E-5
        gardeAVous(Arm)
        time.sleep(1)
        Arm.Arm_serial_servo_write6_array([angle,38,46,54,270,170],s_time)
        
    if i == 6 :
        #position E-6
        gardeAVous(Arm)
        time.sleep(1)
        Arm.Arm_serial_servo_write6_array([angle,42,42,51,270,170],s_time)
        
    if i == 7 :
        #position E-7
        gardeAVous(Arm)
        time.sleep(1)
        Arm.Arm_serial_servo_write6_array([angle,46,38,48,270,170],s_time)
        
    if i == 8 :
        #position E-8
        gardeAVous(Arm)
        time.sleep(1)
        Arm.Arm_serial_servo_write6_array([angle,50,33,47,270,170],s_time)
        
    if i == 9 :
        #position E-9
        gardeAVous(Arm)
        time.sleep(1)
        Arm.Arm_serial_servo_write6_array([angle,54,29,45,270,170],s_time)
        
    if i == 10 :
        #position E-10
        gardeAVous(Arm)
        time.sleep(1)
        Arm.Arm_serial_servo_write6_array([angle,58,25,43,270,170],s_time)
        
    if i == 11:
        #position E-11
        gardeAVous(Arm)
        time.sleep(1)
        Arm.Arm_serial_servo_write6_array([angle,61,21,41,270,170],s_time)
        
    if i == 12:
        #position E-12
        gardeAVous(Arm)
        time.sleep(1)
        Arm.Arm_serial_servo_write6_array([70,64,17,40,270,170],s_time)
        
    if i == 13:
        #position E-13
        gardeAVous(Arm)
        time.sleep(1)
        Arm.Arm_serial_servo_write6_array([angle,67,15,37,270,170],s_time)
        
    if i == 14:
        #position E-14
        gardeAVous(Arm)
        time.sleep(1)
        Arm.Arm_serial_servo_write6_array([angle,70,13,34,270,170],s_time)
        
    if i == 15:
        #position E-15
        gardeAVous(Arm)
        time.sleep(1)
        Arm.Arm_serial_servo_write6_array([angle,73,11,31,270,170],s_time)
        
    if i == 16:
        #position E-16
        gardeAVous(Arm)
        time.sleep(1)
        Arm.Arm_serial_servo_write6_array([angle,76,9,28,270,170],s_time)
        
    if i == 17:
        #position E-17
        gardeAVous(Arm)
        time.sleep(1)
        Arm.Arm_serial_servo_write6_array([angle,79,7,25,270,170],s_time)
        
    if i == 18:
        #position E-18
        gardeAVous(Arm)
        time.sleep(1)
        Arm.Arm_serial_servo_write6_array([angle,82,5,22,270,170],s_time)
        
    if i == 19:
        #position E-19
        gardeAVous(Arm)
        time.sleep(1)
        Arm.Arm_serial_servo_write6_array([angle,85,3,19,270,170],s_time)
        
    if i == 20:
        #position E-20
        gardeAVous(Arm)
        time.sleep(1)
        Arm.Arm_serial_servo_write6_array([angle,87,0,18,270,170],s_time)
        
    if i == 21:
        #position E-21
        gardeAVous(Arm)
        time.sleep(1)
        Arm.Arm_serial_servo_write6_array([angle,90,0,13,270,170],s_time)
        
    if i == 22:
        #position E-22
        gardeAVous(Arm)
        time.sleep(1)
        Arm.Arm_serial_servo_write6_array([angle,92,0,10,270,170],s_time)
        
    if i == 23:
        #position E-23
        gardeAVous(Arm)
        time.sleep(1)
        Arm.Arm_serial_servo_write6_array([angle,96,0,4,270,170],s_time)
        
    if i == 24:
        #position E-24
        gardeAVous(Arm)
        time.sleep(1)
        Arm.Arm_serial_servo_write6_array([angle,96,0,1,270,170],s_time)
        
    if i == 25:
        #position E-25
        gardeAVous(Arm)
        time.sleep(1)
        Arm.Arm_serial_servo_write6_array([angle,94,0,0,270,170],s_time)