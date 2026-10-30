# Brief: major_1.py ported to Raspberry Pi 4 as an SPI inference server

## What major_1.py is now
A PyTorch LeNet-5 MNIST classifier (`lenet5_mnist.pth`, ~99% test accuracy) that runs on a Raspberry Pi 4. It takes images from a microcontroller over SPI, classifies them and sends back the predicted digit and a confidence value. It fills the same role as the FPGA in the rest of the project: the microcontroller can forward PC requests to the Pi instead of the FPGA.

## Modes
- `python3 major_1.py`: serve over SPI (default).
- `python3 major_1.py --predict N`: classify MNIST test image N locally (needs `MNIST_DIR`).
- `python3 major_1.py --train E`: train from scratch for E epochs and overwrite `lenet5_mnist.pth`.

`MNIST_DIR` (env var) defaults to `./MNIST/raw` next to the script. Only `--train` and `--predict` need it.

## SPI link
- The Pi is the SPI **master** (`spidev`, `/dev/spidev0.0`, CE0) and the microcontroller is the **slave**. Settings: mode 3 (CPOL=1, CPHA=1), 1 MHz, MSB first. Mode 3 is needed because an RP2040 SPI slave in mode 0 needs CS raised between every byte, and `xfer2` holds CS low for the whole transfer. Linux spidev on the Pi cannot act as a slave.
- Pi 4 pins: SCLK GPIO11 (pin 23), MOSI GPIO10 (pin 19), MISO GPIO9 (pin 21), CE0 GPIO8 (pin 24), plus a common GND.
- SPI is treated as a byte stream. CS toggles between transfers, so the slave must keep its framing state across CS edges.
- The Pi polls one byte every 1 ms until it reads `0xAA`. While it has no request queued, the slave must shift out `0x00` (anything except `0xAA`).

## Protocol (same framing as mnist_demo.py / main.py)
- Request (slave → Pi): `[0xAA][id][len_hi][len_lo][784 pixels][sum_hi][sum_lo]`. The 784 pixels are raw 28×28 MNIST bytes in row-major order, 0–255.
- Response (Pi → slave): `[0xAA][id][0x00][0x04][status][digit][conf_hi][conf_lo][sum_hi][sum_lo]`.
  - status: `0x00` OK, `0x03` bad request (length ≠ 784 or checksum mismatch; digit and conf are 0).
  - conf: uint16 big-endian, in hundredths of a percent (0–10000) = softmax max × 100.
- `checksum = (sum of every byte from 0xAA through the end of the payload) & 0xFFFF`
- The response matches what `mnist_demo.py`'s `read_result()` already accepts. A slave that relays it unchanged over USB serial therefore works with the existing GUI.

## Key functions
- `predict(model, image_tensor) -> (digit, confidence_percent)`: the return function after the CNN (softmax, then argmax).
- `classify_pixels(model, pixels)`: 784 raw bytes → resize to 32×32 → normalize (0.1307/0.3081) → `predict`.
- `SpiLink.read_request()` / `build_result()` / `serve()`: the protocol loop.

## Changes from the original major_1.py
- Removed the code that dumped weights and biases to `lenet5_weights_biases.txt`.
- Training (previously 1000 epochs on every run) now runs only with `--train`.
- Fixed a bug: the loaded model was replaced by a fresh untrained `LeNet5()`.
- Datasets load only when needed. Paths are no longer hard-coded Windows paths.
- `spidev` is imported only in serve mode, so the file still runs on Windows for `--train` and `--predict`.

## Verified (on Windows, not yet on a Pi)
- `--predict 52`: label 5, predicted 5, 100% confident.
- `serve()` against a fake `spidev` slave: a valid frame got `aa 07 00 04 00 07 27 10 00 f3` (digit 7, 100.00%, correct checksum, ~8 ms inference). A corrupted frame got a status `0x03` reply.

## Not done / open
- No microcontroller SPI-slave firmware has been written for this link. The existing Pico firmware (`pico_fpga_bridge.ino`) acts as SPI *master* to the FPGA. On the Pico W (RP2040), the arduino-pico core's `SPISlave` library would be the starting point.
- Not yet run on real Pi hardware.

## Raspberry Pi setup
Requires **64-bit** Raspberry Pi OS: PyTorch has no 32-bit ARM wheels. Check that `uname -m` prints `aarch64`.

Python libraries (everything else the script imports is in the standard library):

| Library | Used for |
|---|---|
| `torch` | Running the CNN (LeNet-5) |
| `torchvision` | Image preprocessing (`transforms`) |
| `numpy` | Reading the MNIST files (only for `--train` and `--predict`) |
| `pillow` | Converting the received pixel bytes into an image |
| `spidev` | SPI communication |

Install into a venv (Bookworm blocks system-wide pip). `python3-dev` is needed because `spidev` compiles on install:

```bash
sudo apt update
sudo apt install python3-dev python3-venv
python3 -m venv ~/cnn
~/cnn/bin/pip install torch torchvision numpy pillow spidev
```

If pip pulls a large CUDA build of `torch`, add `--index-url https://download.pytorch.org/whl/cpu`.

Before running:
- Enable SPI: `sudo raspi-config` → Interface Options → SPI, then reboot.
- The user must be in the `spi` group (the default user is).
- Copy `lenet5_mnist.pth` into the same folder as `major_1.py`.

Run with the venv's Python:

```bash
~/cnn/bin/python major_1.py                            # serve over SPI
MNIST_DIR=/path/to/MNIST/raw ~/cnn/bin/python major_1.py --predict 52   # local check
```

Bring-up checks: `--predict 52` should print label 5 / predicted 5, which confirms torch and the model load. Plain `major_1.py` should print `Serving on /dev/spidev0.0 ...` and wait, which confirms SPI is enabled and accessible.
