"""
MNIST dual-device demo.

Streams the MNIST test set to two devices over serial at the same time, one
image at a time per device, and shows each device's predictions, time taken
and accuracy side by side.

Request (PC -> device), as in main.py:
    [0xAA][packet_id][len_hi][len_lo][784 pixel bytes][sum_hi][sum_lo]

Response (device -> PC), same framing:
    [0xAA][packet_id][len_hi][len_lo][status][digit][conf_hi][conf_lo]...[sum_hi][sum_lo]
      status : ACK_OK (0x00) once the image is received and classified
      digit  : predicted digit 0-9
      conf   : uint16 big-endian, confidence in hundredths of a percent (0-10000)
    Bytes after conf are ignored here; pico_fpga_bridge.ino puts the raw FPGA
    logits there for calibrate_fpga.py.
"""
import queue
import struct
import threading
import time
import tkinter as tk
from tkinter import ttk

import serial
import serial.tools.list_ports

START_BYTE = 0xAA
ACK_OK = 0x00
BAUD = 115200
READ_TIMEOUT = 5 
MAX_ATTEMPTS = 3
DEVICE_BOOT_DELAY = 2

MNIST_IMAGES = r"D:\Code\Python\MNIST\MNIST\raw\t10k-images.idx3-ubyte"
MNIST_LABELS = r"D:\Code\Python\MNIST\MNIST\raw\t10k-labels.idx1-ubyte"

ROW_H = 40
PREVIEW_ZOOM = 4

RED = "#d93025"
GREEN = "#1e8e3e"
GREY = "#c4c7c5"
MUTED = "#5f6368"
SENDING_BG = "#fff4ce"
DIVIDER = "#ececec"

PENDING, SENDING, DONE, FAILED = range(4)

NO_DEVICE = "None"

def load_mnist(images_path: str, labels_path: str):
    with open(images_path, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"{images_path} is not an MNIST image file")
        size = rows * cols
        data = f.read(count * size)
    with open(labels_path, "rb") as f:
        magic, label_count = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"{labels_path} is not an MNIST label file")
        labels = f.read(label_count)
    images = [data[i * size:(i + 1) * size] for i in range(count)]
    return images, labels, (cols, rows)


def calculate_checksum(packet_id: int, payload: bytes) -> int:
    length = len(payload)
    return (START_BYTE + packet_id + (length >> 8) + (length & 0xFF) + sum(payload)) & 0xFFFF


def build_packet(packet_id: int, payload: bytes) -> bytes:
    checksum = calculate_checksum(packet_id, payload)
    return (bytes([START_BYTE, packet_id]) + len(payload).to_bytes(2, "big")
            + payload + checksum.to_bytes(2, "big"))


def read_frame(ser):
    """Read one response frame. Returns (packet_id, payload) or None if invalid."""
    header = ser.read(4)
    if len(header) != 4 or header[0] != START_BYTE:
        return None
    packet_id = header[1]
    length = (header[2] << 8) | header[3]
    if length > 64:
        return None
    payload = ser.read(length)
    crc = ser.read(2)
    if len(payload) != length or len(crc) != 2:
        return None
    if int.from_bytes(crc, "big") != calculate_checksum(packet_id, payload):
        return None
    return packet_id, payload


def read_result(ser, expected_id: int):
    """Read one response frame. Returns (digit, confidence %) or None if invalid."""
    frame = read_frame(ser)
    if frame is None:
        return None
    packet_id, payload = frame
    if packet_id != expected_id or len(payload) < 4 or payload[0] != ACK_OK or payload[1] > 9:
        return None
    return payload[1], int.from_bytes(payload[2:4], "big") / 100


class DeviceWorker(threading.Thread):
    """Sends images to one device, waiting for each result before the next."""

    def __init__(self, key, ser, images, count, t0, events, stop):
        super().__init__(daemon=True)
        self.key, self.ser, self.images, self.count = key, ser, images, count
        self.t0, self.events, self.stop = t0, events, stop

    def run(self):
        try:
            for i in range(self.count):
                if self.stop.is_set():
                    break
                self.events.put((self.key, "sending", i))
                self.events.put((self.key, "result", i, self._classify(i)))
        except serial.SerialException as e:
            self.events.put((self.key, "error", str(e)))
        finally:
            elapsed = time.perf_counter() - self.t0
            self.ser.close()
            self.events.put((self.key, "finished", elapsed))

    def _classify(self, i):
        packet_id = i & 0xFF
        frame = build_packet(packet_id, self.images[i])
        for _ in range(MAX_ATTEMPTS):
            if self.stop.is_set():
                return None
            # Drop stale bytes (boot banner, late replies) so they aren't read as this result
            self.ser.reset_input_buffer()
            self.ser.write(frame)
            result = read_result(self.ser, packet_id)
            if result:
                return result
        return None


# ---------------------------------------------------------------- UI

class Thumbnails:
    """Lazily built Tk images for MNIST digits, shared by both panes."""

    def __init__(self, images, size):
        self.images, self.size = images, size
        self.cache = {}

    def get(self, i):
        img = self.cache.get(i)
        if img is None:
            w, h = self.size
            pgm = f"P5 {w} {h} 255\n".encode() + self.images[i]
            img = self.cache[i] = tk.PhotoImage(data=pgm, format="PPM")
        return img

    def preview(self, i):
        return self.get(i).zoom(PREVIEW_ZOOM)


class ImageList(ttk.Frame):
    """Scrollable list of every test image; only the visible rows are drawn."""

    def __init__(self, master, labels, thumbs):
        super().__init__(master)
        self.labels, self.thumbs = labels, thumbs
        n = len(labels)
        self.state = bytearray(n)
        self.pred = [None] * n
        self.run_count = n
        self.current = None

        self.canvas = tk.Canvas(self, bg="white", highlightthickness=0,
                                yscrollincrement=ROW_H, scrollregion=(0, 0, 1, n * ROW_H))
        bar = ttk.Scrollbar(self, orient="vertical", command=self._yview)
        self.canvas.configure(yscrollcommand=bar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        bar.grid(row=0, column=1, sticky="ns")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.canvas.bind("<Configure>", lambda e: self.redraw())
        self.canvas.bind("<MouseWheel>", lambda e: self._yview("scroll", -3 * (e.delta // 120), "units"))
        self.canvas.bind("<Button-4>", lambda e: self._yview("scroll", -3, "units"))
        self.canvas.bind("<Button-5>", lambda e: self._yview("scroll", 3, "units"))

    def _yview(self, *args):
        self.canvas.yview(*args)
        self.redraw()

    def reset(self, run_count):
        self.state[:] = bytes(len(self.state))
        self.pred = [None] * len(self.pred)
        self.run_count = run_count
        self.current = None
        self.canvas.yview_moveto(0)

    def scroll_to(self, i):
        top = self.canvas.canvasy(0)
        bottom = top + self.canvas.winfo_height()
        if i * ROW_H < top or (i + 1) * ROW_H > bottom:
            rows_visible = max(1, self.canvas.winfo_height() // ROW_H)
            first = max(0, i - rows_visible // 3)
            self.canvas.yview_moveto(first / len(self.labels))

    def redraw(self):
        c = self.canvas
        c.delete("all")
        w = c.winfo_width()
        first = int(c.canvasy(0)) // ROW_H
        last = min(len(self.labels), int(c.canvasy(c.winfo_height())) // ROW_H + 1)
        for i in range(first, last):
            y = i * ROW_H
            mid = y + ROW_H / 2
            state = self.state[i]
            if i == self.current:
                c.create_rectangle(0, y, w, y + ROW_H, fill=SENDING_BG, width=0)
            c.create_line(0, y + ROW_H - 1, w, y + ROW_H - 1, fill=DIVIDER)

            dot = GREEN if state == DONE else GREY if i >= self.run_count else RED
            c.create_oval(12, mid - 6, 24, mid + 6, fill=dot, outline="")
            c.create_image(36, mid, image=self.thumbs.get(i), anchor="w")
            c.create_text(76, mid, text=f"test_{i:05d}", anchor="w", font=("Segoe UI", 10))
            c.create_text(170, mid, text=f"label {self.labels[i]}", anchor="w",
                          font=("Segoe UI", 9), fill=MUTED)

            if state == DONE:
                digit, conf = self.pred[i]
                text = f"→ {digit}   {conf:5.1f}%"
                colour = GREEN if digit == self.labels[i] else RED
            elif state == FAILED:
                text, colour = "no valid response", RED
            elif state == SENDING:
                text, colour = "sending…", MUTED
            else:
                text, colour = "", MUTED
            c.create_text(w - 14, mid, text=text, anchor="e", fill=colour,
                          font=("Consolas", 10, "bold"))


class DevicePane(ttk.Frame):
    def __init__(self, master, title, labels, thumbs):
        super().__init__(master, padding=(10, 8))
        self.labels, self.thumbs = labels, thumbs
        self.port = tk.StringVar()
        self.running = False
        self.sent = self.correct = self.failed = 0

        head = ttk.Frame(self)
        head.grid(row=0, column=0, sticky="ew")
        ttk.Label(head, text=title, font=("Segoe UI", 14, "bold")).pack(side="left")
        self.port_box = ttk.Combobox(head, textvariable=self.port, width=10, state="readonly")
        self.port_box.pack(side="right")
        ttk.Label(head, text="Port").pack(side="right", padx=(0, 6))

        now = ttk.Frame(self, padding=(0, 10))
        now.grid(row=1, column=0, sticky="ew")
        self._preview_image = tk.PhotoImage(width=28 * PREVIEW_ZOOM, height=28 * PREVIEW_ZOOM)
        self.preview = tk.Label(now, bg="black", image=self._preview_image, borderwidth=0)
        self.preview.grid(row=0, column=0, rowspan=4, padx=(0, 14))
        self.now_name = ttk.Label(now, text="—", font=("Segoe UI", 11, "bold"))
        self.now_name.grid(row=0, column=1, sticky="w")
        self.now_pred = ttk.Label(now, text="Waiting to start", font=("Segoe UI", 11))
        self.now_pred.grid(row=1, column=1, sticky="w")
        self.elapsed = ttk.Label(now, text="0.000 s", font=("Consolas", 18, "bold"))
        self.elapsed.grid(row=2, column=1, sticky="w", pady=(6, 0))
        self.counts = ttk.Label(now, text="", foreground=MUTED)
        self.counts.grid(row=3, column=1, sticky="w")

        self.progress = ttk.Progressbar(self, mode="determinate")
        self.progress.grid(row=2, column=0, sticky="ew", pady=(0, 8))

        self.list = ImageList(self, labels, thumbs)
        self.list.grid(row=3, column=0, sticky="nsew")

        self.summary = ttk.Label(self, text="", font=("Segoe UI", 12, "bold"), padding=(0, 8, 0, 0))
        self.summary.grid(row=4, column=0, sticky="w")

        self.rowconfigure(3, weight=1)
        self.columnconfigure(0, weight=1)

    def reset(self, count):
        self.count = count
        self.sent = self.correct = self.failed = 0
        self.running = False
        self.list.reset(count)
        self.progress.configure(maximum=count, value=0)
        self.elapsed.configure(text="0.000 s")
        self.now_name.configure(text="—")
        self.now_pred.configure(text="Connecting…", foreground="")
        self.summary.configure(text="", foreground="")
        self._update_counts()

    def _update_counts(self):
        text = f"{self.sent} / {self.count} done   ·   {self.correct} correct"
        if self.failed:
            text += f"   ·   {self.failed} failed"
        self.counts.configure(text=text)

    def on_sending(self, i):
        self.list.state[i] = SENDING
        self.list.current = i
        self.list.scroll_to(i)
        self._preview_image = self.thumbs.preview(i)
        self.preview.configure(image=self._preview_image)
        self.now_name.configure(text=f"test_{i:05d}   (label {self.labels[i]})")
        self.now_pred.configure(text="Waiting for result…", foreground=MUTED)

    def on_result(self, i, result):
        self.sent += 1
        self.list.current = None
        if result is None:
            self.failed += 1
            self.list.state[i] = FAILED
            self.now_pred.configure(text="No valid response", foreground=RED)
        else:
            digit, conf = result
            self.list.state[i] = DONE
            self.list.pred[i] = result
            ok = digit == self.labels[i]
            self.correct += ok
            self.now_pred.configure(text=f"Predicted {digit}   ·   {conf:.1f}% confident",
                                    foreground=GREEN if ok else RED)
        self.progress.configure(value=self.sent)
        self._update_counts()

    def on_error(self, message):
        self.running = False
        self.now_pred.configure(text=message, foreground=RED)
        self.summary.configure(text="Device error", foreground=RED)

    def on_finished(self, elapsed):
        self.running = False
        self.list.current = None
        self.elapsed.configure(text=f"{elapsed:.3f} s")
        accuracy = 100 * self.correct / self.sent if self.sent else 0
        text = f"Time {elapsed:.3f} s   ·   Accuracy {accuracy:.2f}% ({self.correct}/{self.sent})"
        if self.sent < self.count:
            text += "   ·   stopped early"
        if self.summary.cget("text") != "Device error":
            self.summary.configure(text=text, foreground="")
        self.final_time = elapsed


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MNIST Device Race")
        self.geometry("1200x800")
        self.minsize(820, 560)

        self.images, self.labels, size = load_mnist(MNIST_IMAGES, MNIST_LABELS)
        self.thumbs = Thumbnails(self.images, size)
        self.events = queue.Queue()
        self.stop_event = threading.Event()
        self.t0 = None
        self.active = set()
        self.in_run = False

        ribbon = ttk.Frame(self, padding=(12, 8))
        ribbon.pack(fill="x")
        self.start_btn = ttk.Button(ribbon, text="▶  Start", width=12, command=self.toggle)
        self.start_btn.pack(side="left")
        self.timer = ttk.Label(ribbon, text="00:00.000", font=("Consolas", 22, "bold"))
        self.timer.pack(side="left", padx=20)
        self.status = ttk.Label(ribbon, text="", foreground=MUTED)
        self.status.pack(side="left")

        self.refresh_btn = ttk.Button(ribbon, text="Refresh ports", command=self.refresh_ports)
        self.refresh_btn.pack(side="right")
        self.count = tk.IntVar(value=len(self.images))
        self.count_box = ttk.Spinbox(ribbon, from_=1, to=len(self.images), width=7,
                                     textvariable=self.count)
        self.count_box.pack(side="right", padx=(0, 16))
        ttk.Label(ribbon, text="Images").pack(side="right", padx=(0, 6))
        ttk.Separator(self).pack(fill="x")

        panes = ttk.PanedWindow(self, orient="horizontal")
        panes.pack(fill="both", expand=True)
        self.panes = {}
        for key in ("A", "B"):
            pane = DevicePane(panes, f"Device {key}", self.labels, self.thumbs)
            panes.add(pane, weight=1)
            self.panes[key] = pane

        self.refresh_ports()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(30, self.poll)

    # ---- controls

    def refresh_ports(self):
        ports = sorted(p.device for p in serial.tools.list_ports.comports())
        for pane in self.panes.values():
            pane.port_box.configure(values=ports + [NO_DEVICE])
        defaults = (["COM19"] if "COM19" in ports else []) + [p for p in ports if p != "COM19"]
        defaults += [NO_DEVICE] * len(self.panes)
        for pane, default in zip(self.panes.values(), defaults):
            if pane.port.get() not in ports:
                pane.port.set(default)
        self.status.configure(text=f"{len(ports)} serial port(s) found" if ports else "No serial ports found")

    def toggle(self):
        if self.in_run:
            self.stop_event.set()
            self.start_btn.configure(text="Stopping…", state="disabled")
        else:
            self.start()

    def start(self):
        # A pane set to NO_DEVICE sits the run out, so the demo also works with one device
        ports = {key: pane.port.get() for key, pane in self.panes.items()
                 if pane.port.get() not in ("", NO_DEVICE)}
        if not ports:
            self.status.configure(text="Choose a port for at least one device", foreground=RED)
            return
        if len(set(ports.values())) != len(ports):
            self.status.configure(text="The two devices need different ports", foreground=RED)
            return
        try:
            count = max(1, min(len(self.images), int(self.count.get())))
        except (tk.TclError, ValueError):
            count = len(self.images)
        self.count.set(count)

        for key, pane in self.panes.items():
            if key in ports:
                pane.reset(count)
            else:
                pane.reset(0)
                pane.now_pred.configure(text="No device selected", foreground=MUTED)
        self.t0 = None
        self.timer.configure(text="00:00.000")
        self.status.configure(text="Connecting…", foreground=MUTED)
        self.stop_event = threading.Event()
        self.in_run = True
        self.start_btn.configure(text="■  Stop")
        for widget in (self.count_box, self.refresh_btn,
                       *(p.port_box for p in self.panes.values())):
            widget.configure(state="disabled")
        threading.Thread(target=self.connect_and_run, args=(ports, count), daemon=True).start()

    def connect_and_run(self, ports, count):
        connections = {}
        for key, port in ports.items():
            try:
                connections[key] = serial.Serial(port, BAUD, timeout=READ_TIMEOUT, write_timeout=READ_TIMEOUT)
            except serial.SerialException as e:
                self.events.put((key, "error", f"Could not open {port}: {e}"))
        if connections:
            time.sleep(DEVICE_BOOT_DELAY)
        if self.stop_event.is_set() or not connections:
            for ser in connections.values():
                ser.close()
            self.events.put((None, "aborted"))
            return
        # One shared start time so both devices are timed from the same instant
        t0 = time.perf_counter()
        self.events.put((None, "started", t0, list(connections)))
        for key, ser in connections.items():
            DeviceWorker(key, ser, self.images, count, t0, self.events, self.stop_event).start()

    def end_run(self, message):
        self.in_run = False
        self.t0 = None
        self.active.clear()
        self.start_btn.configure(text="▶  Start", state="normal")
        self.count_box.configure(state="normal")
        self.refresh_btn.configure(state="normal")
        for pane in self.panes.values():
            pane.port_box.configure(state="readonly")
        self.status.configure(text=message, foreground=MUTED)

    # ---- event loop

    def poll(self):
        dirty = set()
        try:
            while True:
                key, kind, *args = self.events.get_nowait()
                pane = self.panes.get(key)
                if kind == "started":
                    self.t0, keys = args
                    self.active = set(keys)
                    for k in keys:
                        self.panes[k].running = True
                        self.panes[k].now_pred.configure(text="Starting…", foreground=MUTED)
                    self.status.configure(text="Running", foreground=MUTED)
                elif kind == "aborted":
                    self.end_run("Stopped" if self.stop_event.is_set() else "Could not connect to any device")
                elif kind == "sending":
                    pane.on_sending(*args)
                    dirty.add(key)
                elif kind == "result":
                    pane.on_result(*args)
                    dirty.add(key)
                elif kind == "error":
                    pane.on_error(*args)
                elif kind == "finished":
                    pane.on_finished(*args)
                    dirty.add(key)
                    self.active.discard(key)
                    if not self.active:
                        self.finish_run()
        except queue.Empty:
            pass

        for key in dirty:
            self.panes[key].list.redraw()
        if self.t0 is not None:
            now = time.perf_counter() - self.t0
            self.timer.configure(text=format_clock(now))
            for pane in self.panes.values():
                if pane.running:
                    pane.elapsed.configure(text=f"{now:.3f} s")
        self.after(30, self.poll)

    def finish_run(self):
        finished = [p for p in self.panes.values() if p.sent]
        if finished:
            self.timer.configure(text=format_clock(max(p.final_time for p in finished)))
        message = "Stopped" if self.stop_event.is_set() else "Done"
        if len(finished) == 2 and not self.stop_event.is_set():
            a, b = self.panes["A"], self.panes["B"]
            faster, slower = (a, b) if a.final_time <= b.final_time else (b, a)
            name = "A" if faster is a else "B"
            message += f"  ·  Device {name} was {slower.final_time - faster.final_time:.3f} s faster"
        self.end_run(message)

    def on_close(self):
        self.stop_event.set()
        self.destroy()


def format_clock(seconds):
    minutes, seconds = divmod(seconds, 60)
    return f"{int(minutes):02d}:{seconds:06.3f}"


if __name__ == "__main__":
    try:
        # Render crisply on scaled Windows displays instead of being bitmap-stretched
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (ImportError, AttributeError, OSError):
        pass
    App().mainloop()
