"""
LeNet-5 MNIST classifier, served to a microcontroller over SPI on a Raspberry Pi 4.

The Pi is the SPI master (spidev, /dev/spidev0.0) and the microcontroller is
the slave. The Pi polls the slave for a request, classifies the image and
writes the result back, using the framing from mnist_demo.py:

Request (microcontroller -> Pi):
    [0xAA][packet_id][len_hi][len_lo][784 pixel bytes][sum_hi][sum_lo]

Response (Pi -> microcontroller), same framing:
    [0xAA][packet_id][0x00][0x04][status][digit][conf_hi][conf_lo][sum_hi][sum_lo]
      status : RES_OK (0x00) or RES_BAD_REQUEST (0x03)
      digit  : predicted digit 0-9
      conf   : uint16 big-endian, confidence in hundredths of a percent (0-10000)

checksum = (sum of every byte from 0xAA to the end of the payload) & 0xFFFF

SPI is treated as a byte stream: chip select toggles between transfers, so the
slave must not reset its framing on CS. While it has nothing to send it should
shift out 0x00; the Pi reads one byte every POLL_INTERVAL until it sees 0xAA.

Usage:
    python3 major_1.py                 serve over SPI (default)
    python3 major_1.py --predict 52    classify one MNIST test image locally
    python3 major_1.py --train 10      train for 10 epochs and save the weights
"""
import argparse
import os
import struct
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lenet5_mnist.pth")
MNIST_DIR = os.environ.get("MNIST_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "MNIST", "raw"))

SPI_BUS = 0
SPI_DEVICE = 0  # CE0, GPIO 8
SPI_SPEED_HZ = 1_000_000
# Mode 3, not 0: an RP2040 SPI slave in mode 0 needs CS raised between every
# byte, but xfer2 holds CS low for the whole transfer
SPI_MODE = 3
POLL_INTERVAL = 0.001

START_BYTE = 0xAA
IMAGE_SIZE = 28 * 28
RES_OK = 0x00
RES_BAD_REQUEST = 0x03

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

class MNISTDataset(Dataset):

    def __init__(self, images_path, labels_path, transform=None):

        self.images = self.read_images(images_path)
        self.labels = self.read_labels(labels_path)
        self.transform = transform

    def read_images(self, path):
        with open(path, "rb") as f:
            magic, num_images, rows, cols = struct.unpack(
                ">IIII",
                f.read(16)
            )
            data = np.frombuffer(
                f.read(),
                dtype=np.uint8
            )
            data = data.reshape(
                num_images,
                rows,
                cols
            )
        return data

    def read_labels(self, path):
        with open(path, "rb") as f:
            magic, num_labels = struct.unpack(
                ">II",
                f.read(8)
            )
            labels = np.frombuffer(
                f.read(),
                dtype=np.uint8
            )
        return labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        image = self.images[index]
        label = int(self.labels[index])

        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, label

def load_dataset(split):
    prefix = "train" if split == "train" else "t10k"
    return MNISTDataset(
        images_path=os.path.join(MNIST_DIR, f"{prefix}-images.idx3-ubyte"),
        labels_path=os.path.join(MNIST_DIR, f"{prefix}-labels.idx1-ubyte"),
        transform=transform
    )

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=5
        )

        self.pool1 = nn.AvgPool2d(
            kernel_size=2,
            stride=2
        )

        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5
        )

        self.pool2 = nn.AvgPool2d(
            kernel_size=2,
            stride=2
        )

        self.fc1 = nn.Linear(
            16 * 5 * 5,
            120
        )

        self.fc2 = nn.Linear(
            120,
            84
        )

        self.fc3 = nn.Linear(
            84,
            10
        )

        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.pool1(x)
        x = self.activation(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

def load_model():
    model = LeNet5()
    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )
    model = model.to(device)
    model.eval()
    return model

def predict(model, image):
    """Run the CNN on one transformed image tensor [1, 32, 32].

    Returns (digit, confidence) with confidence as a percentage 0-100.
    """
    with torch.inference_mode():
        output = model(image.unsqueeze(0).to(device))
        probabilities = torch.softmax(output, dim=1)
        confidence, digit = torch.max(probabilities, dim=1)
    return digit.item(), confidence.item() * 100

def classify_pixels(model, pixels):
    """Classify 784 raw MNIST pixel bytes (28x28, row-major, 0-255)."""
    image = Image.frombytes("L", (28, 28), bytes(pixels))
    return predict(model, transform(image))

def train(model, train_loader, optimizer, criterion, epoch):

    model.train()

    running_loss = 0

    for batch_idx, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 100 == 0:

            print(
                f"Epoch: {epoch} "
                f"Batch: {batch_idx}/{len(train_loader)} "
                f"Loss: {loss.item():.4f}"
            )

    print(
        f"Average training loss: "
        f"{running_loss / len(train_loader):.4f}"
    )


def test(model, test_loader):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predictions = torch.max(
                outputs,
                dim=1
            )

            total += labels.size(0)

            correct += (
                predictions == labels
            ).sum().item()

    accuracy = 100 * correct / total

    print(
        f"Test Accuracy: "
        f"{accuracy:.2f}%"
    )

    return accuracy

def run_training(epochs):
    train_loader = DataLoader(
        load_dataset("train"),
        batch_size=64,
        shuffle=True
    )

    test_loader = DataLoader(
        load_dataset("test"),
        batch_size=1000,
        shuffle=False
    )

    model = LeNet5().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    for epoch in range(1, epochs + 1):

        train(model, train_loader, optimizer, criterion, epoch)

        test(model, test_loader)

    torch.save(
        model.state_dict(),
        MODEL_PATH
    )

    print(f"Model saved as {MODEL_PATH}")

# ---------------------------------------------------------------- SPI protocol

def calculate_checksum(packet_id, payload):
    length = len(payload)
    return (START_BYTE + packet_id + (length >> 8) + (length & 0xFF) + sum(payload)) & 0xFFFF


def build_packet(packet_id, payload):
    checksum = calculate_checksum(packet_id, payload)
    return (bytes([START_BYTE, packet_id]) + len(payload).to_bytes(2, "big")
            + payload + checksum.to_bytes(2, "big"))


def build_result(packet_id, status, digit=0, confidence=0.0):
    conf = max(0, min(10000, round(confidence * 100)))
    return build_packet(packet_id, bytes([status, digit]) + conf.to_bytes(2, "big"))


class SpiLink:
    """Byte-stream view of the SPI bus with the Pi as master."""

    def __init__(self, bus=SPI_BUS, dev=SPI_DEVICE):
        import spidev  # only on the Pi, so --train and --predict still run elsewhere
        self.spi = spidev.SpiDev()
        self.spi.open(bus, dev)
        self.spi.max_speed_hz = SPI_SPEED_HZ
        self.spi.mode = SPI_MODE

    def read(self, n):
        return bytes(self.spi.xfer2([0x00] * n))

    def write(self, data):
        self.spi.xfer2(list(data))

    def wait_for_start(self):
        while self.read(1)[0] != START_BYTE:
            time.sleep(POLL_INTERVAL)

    def read_request(self):
        """Wait for one request frame. Returns (packet_id, payload or None if invalid)."""
        self.wait_for_start()
        packet_id, len_hi, len_lo = self.read(3)
        length = (len_hi << 8) | len_lo
        if length != IMAGE_SIZE:
            return packet_id, None
        payload = self.read(length)
        checksum = int.from_bytes(self.read(2), "big")
        if checksum != calculate_checksum(packet_id, payload):
            return packet_id, None
        return packet_id, payload

    def close(self):
        self.spi.close()


def serve(model):
    link = SpiLink()
    print(f"Serving on /dev/spidev{SPI_BUS}.{SPI_DEVICE} at {SPI_SPEED_HZ} Hz, mode {SPI_MODE}")
    try:
        while True:
            packet_id, pixels = link.read_request()
            if pixels is None:
                link.write(build_result(packet_id, RES_BAD_REQUEST))
                print(f"Packet {packet_id}: bad request")
                continue
            start = time.perf_counter()
            digit, confidence = classify_pixels(model, pixels)
            link.write(build_result(packet_id, RES_OK, digit, confidence))
            print(f"Packet {packet_id}: digit {digit}, {confidence:.2f}% "
                  f"({(time.perf_counter() - start) * 1000:.1f} ms)")
    except KeyboardInterrupt:
        pass
    finally:
        link.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=int, metavar="EPOCHS", help="train from scratch and save the weights")
    parser.add_argument("--predict", type=int, metavar="INDEX", help="classify one MNIST test image")
    args = parser.parse_args()

    print("Using device:", device)

    if args.train:
        run_training(args.train)
        return

    model = load_model()
    print("Model loaded successfully.")

    if args.predict is not None:
        image, label = load_dataset("test")[args.predict]
        digit, confidence = predict(model, image)
        print("Actual label:", label)
        print(f"Predicted label: {digit} ({confidence:.2f}% confident)")
        return

    serve(model)


if __name__ == "__main__":
    main()
