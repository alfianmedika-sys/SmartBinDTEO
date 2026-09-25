import time
import serial

# Initialize serial communication with ESP32
ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
time.sleep(2)  # beri waktu ESP32-S3 selesai reset setelah port dibuka
while True:
    # Wait for signal from ESP32
    signal = ser.read()
    if signal == b'1':
        # Dummy processing
        dummy_value = 1.75  # Replace this with actual processing if needed

        # Send dummy value to ESP32
        print(f"Sending dummy value: {dummy_value}")
        ser.write(f'{dummy_value:.2f}\n'.encode('utf-8'))
    else:
        time.sleep(1)  # Sleep for 1 second before checking again