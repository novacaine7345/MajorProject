import tensorflow as tf
import matplotlib.pyplot as plt
import serial
import time
from dataclasses import dataclass
import tkinter as tk
from tkinter import ttk

START_BYTE = 0xAA 
ACK_OK = 0x01
ACK_BAD_START = 0X02
ACK_BAD_LENGTH = 0X03
ACK_TIMEOUT = 0X04
ACK_BAD_CRC = 0X05

@dataclass
class Packet:
    packet_id: int
    payload: bytes

PORT = "COM19"
BAUD = 115200

def calculate_checksum(packet_id: int, payload: bytes) -> int:
    checksum = 0
    checksum += START_BYTE
    checksum += packet_id
    length = len(payload)
    checksum += (length >> 8) 
    checksum += (length & 0xFF)
    
    for byte in payload:
        checksum += byte
    
    checksum &= 0xFFFF
    return checksum

def build_packet(packet: Packet) -> bytes:
    length = len(packet.payload)
    checksum = calculate_checksum(packet.packet_id, packet.payload)
    tx = bytearray()
    tx.append(START_BYTE)
    tx.append(packet.packet_id)
    tx.extend(length.to_bytes(2, "big"))
    tx.extend(packet.payload)
    tx.extend(checksum.to_bytes(2, "big"))
    
    return bytes(tx)

def send_packet(ser, packet: Packet):
    data = build_packet(packet)
    ser.write(data)
    # print(f"Sent {len(data)} bytes")
    
def recieve_ack(ser):
    header = ser.read(4)
    if len(header) != 4:
        print("Timeout waiting for ACK")
        return

    start = header[0]
    packet_id = header[1]
    length = (header[2] << 8 ) | header[3]
    payload = ser.read(length)
    
    crc = ser.read(2)
    return True

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
plt.imshow(x_test[25], cmap="gray")
plt.title(f"Label: {y_test[25]}")
plt.axis("off")
plt.show()

serConn = serial.Serial(PORT, BAUD, timeout=2, write_timeout=2)
startTime = time.perf_counter() 

successCount = 0
PACKET_MNIST = 0x01

imageCount = len(x_test) 
time.sleep(2)

for index in range(imageCount):
    image = x_test[index]
    label = y_test[index]
    pixels = image.flatten()

    packet = Packet(packet_id=PACKET_MNIST, payload=bytes(pixels))
    send_packet(serConn, packet)
    if index%100 == 0:
        print(f"Sending Image {index}")
    ack = recieve_ack(serConn)
    if ack:
        successCount += 1
    else:
        print(f"Image {index} failed")
        
endTime = time.perf_counter()
elapsedTime = endTime - startTime

print(f"{successCount} / {imageCount} images transferred without errors")
print(f"Total time taken : {elapsedTime:.3f} seconds")

serConn.close()
    
