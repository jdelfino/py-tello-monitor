import asyncio
from collections import OrderedDict
from contextlib import contextmanager
import csv
import cv2
import logging
import math
import multiprocessing as mp
import numpy as np
import os
import queue
from threading import Thread, Event
import time
import multiprocessing as mp

import djitellopy
#from djitellopy import Tello as OrigTello, RESPONSE_TIMEOUT

log = logging.getLogger(__name__)

WEBCAM_SRC = 0
DATA_SOURCE = 'data'
DRONE_VIDEO_SOURCE = 'drone'
CAM_VIDEO_SOURCE = 'cam'

class Tello(djitellopy.Tello):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.monitor = None

	def __del__(self):
		print("GOT NEW DEL")
		pass

	def register_watcher(self, monitor):
		self.monitor = monitor

	def send_control_command(self, command: str, timeout: int = djitellopy.Tello.RESPONSE_TIMEOUT) -> bool:
		if self.monitor:
			self.monitor.record_command(command)
		return super().send_control_command(command, timeout)


def _make_combined_frame(drone_frame, cam_frame, data_frame):
	drone_height, drone_width, _ = drone_frame.shape
	_, total_width, _ = cam_frame.shape

	#cv2.putText(drone_frame, "%.2f" % drone_ts, (50, 75), font, 1, (0, 255, 255), 4)

	top_frame = np.zeros((drone_height, total_width, 3), np.uint8)
	top_frame[0:drone_height,0:drone_width] = drone_frame

	def putDataText(text, y_offset):
		cv2.putText(top_frame, text, (drone_width + 50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 255, 255), 4)

	putDataText("Height (cm): %d" % data_frame['height_cm'], 100)
	putDataText("Pitch (deg): %d" % data_frame['pitch_degrees'], 150)
	putDataText("Roll (deg): %d" % data_frame['roll_degrees'], 200)
	putDataText("Yaw (deg): %d" % data_frame['yaw_degrees'], 250)
	putDataText("Speed x: %d" % data_frame['speed_x'], 300)
	putDataText("Speed y: %d" % data_frame['speed_y'], 350)
	putDataText("Speed z: %d" % data_frame['speed_z'], 400)

	# timestamp already there from individual videos
	#cv2.putText(f2, "%.2f" % cam_ts, (50, 75), font, 1, (0, 255, 255), 4)

	return np.concatenate((top_frame, cam_frame), axis=0)


def _wrap_send_command(tello, m):
	'''
	Injects monitoring around the function that sends commands to the drone, so the commands
	show up in the captured data.
	'''
	def wrap(f):
		def wrapped(command: str, timeout: int = Tello.RESPONSE_TIMEOUT) -> bool:
			print("Executing command: {}".format(command))
			m.record_command(command)
			return f(tello, command, timeout)
		return wrapped
	# The check is to prevent double wrapping
	if not getattr(tello, 'orig_send_control_command', None):
		setattr(tello, 'orig_send_control_command', tello.__dict__['send_control_command'])
		setattr(tello, 'send_control_command', wrap(tello.__dict__['send_control_command']))


class BaseMonitor(Thread):
	def __init__(self):
		super().__init__()
		self.event = Event()
		self.dependents = []

	def attach(self, p):
		self.dependents.append(p)
		if not p.is_alive():
			p.start()

	def stop(self):
		self.event.set()
		for d in self.dependents:
			d.stop()

	def joinall(self):
		self.join()
		for d in self.dependents:
			d.join()

class VideoMonitor(BaseMonitor):
	def __init__(self, address, start_time, fps, source_name, ts_offset=0):
		super().__init__()
		self.address = address
		self.source_name = source_name
		self.fps = fps
		self.start_time = start_time
		self.ts_offset = ts_offset

	def run(self):
		cap = cv2.VideoCapture(self.address)
		next_time = time.time() + (1/self.fps)
		try:
			while not self.event.is_set():
				grabbed, frame = cap.read()

				if not grabbed:
					print("VM {} grab failed".format(self.address))
					break

				t = time.time()
				if t > next_time:
					next_time = time.time() + (1/self.fps)
					for d in self.dependents:
						d.send_frame([frame, time.time() - self.start_time - self.ts_offset, self.source_name])
		finally:
			if cap:
				cap.release()


class DataMonitor(BaseMonitor):
	def __init__(self, start_time, fps, tello):
		super().__init__()
		self.command_queue = []
		self.tello = tello
		self.fps = fps
		self.start_time = start_time

		self.datapoints = OrderedDict([
			('time_s', lambda t: None),
			('acceleration_x', lambda t: t.get_acceleration_x()),
			('acceleration_y', lambda t: t.get_acceleration_y()),
			('acceleration_z', lambda t: t.get_acceleration_z()),
			('barometer', lambda t: t.get_barometer()),
			('battery', lambda t: t.get_battery()),
			('flight_time', lambda t: t.get_flight_time()),
			('height_cm', lambda t: t.get_height()),
			('to_floor_cm', lambda t: t.get_distance_tof()),
			('pitch_degrees', lambda t: t.get_pitch()),
			('roll_degrees', lambda t: t.get_roll()),
			('speed_x', lambda t: t.get_speed_x()),
			('speed_y', lambda t: t.get_speed_y()),
			('speed_z', lambda t: t.get_speed_z()),
			('temperature_c', lambda t: t.get_temperature()),
			('yaw_degrees', lambda t: t.get_yaw()),
			('command', lambda t: (self.command_queue.pop() if len(self.command_queue) else None))
		])

	def run(self):
		self.tello.register_watcher(self)
		#_wrap_send_command(self.tello, self)
		while not self.event.is_set():
			data_frame = {}
			for k, get_fn in self.datapoints.items():
				data_frame[k] = get_fn(self.tello)

			t = time.time() - self.start_time
			data_frame['time_s'] = t

			for d in self.dependents:
				d.send_frame([data_frame, time.time() - self.start_time, DATA_SOURCE])
			time.sleep(1 / self.fps)

	def record_command(self, command):
		self.command_queue.append(command)


class BaseProcessor(Thread):
	def __init__(self):
		super().__init__()
		self.in_queue = queue.Queue()
		self.event = Event()

	def queue(self):
		return self.in_queue

	def stop(self):
		self.event.set()

	def send_frame(self, frame):
		self.in_queue.put(frame)

	def run(self):
		while not self.event.is_set():
			try:
				frame = self.in_queue.get(True, 1)
				self.handle_frame(frame)
			except queue.Empty:
				continue
		self.finalize()

	def finalize(self):
		pass

class FrameAccumulator(BaseProcessor):
	def __init__(self):
		super().__init__()
		self.frames = []

	def handle_frame(self, frame):
		self.frames.append(frame)


class VideoWriter(BaseProcessor):
	def __init__(self, filename, fps):
		super().__init__()
		self.filename = filename
		self.fps = fps
		self.writer = None

	def handle_frame(self, frame):
		if self.writer is None:
			height, width, _ = frame[0].shape
			self.writer = cv2.VideoWriter(self.filename, cv2.VideoWriter_fourcc(*'avc1'), self.fps, (width, height))

		cv2.putText(frame[0], "%.2f" % frame[1], (50, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 4)
		self.writer.write(frame[0])

	def finalize(self):
		if self.writer:
			self.writer.release()


class DataWriter(BaseProcessor):
	def __init__(self, filename):
		super().__init__()
		self.filename = filename
		self.dwriter = None
		self.outfile = None

	def handle_frame(self, frame):
		if self.outfile is None:
			self.outfile = open(self.filename, 'w', newline='')
		if self.dwriter is None:
			self.dwriter = csv.DictWriter(self.outfile, fieldnames=frame[0].keys())
			self.dwriter.writeheader()
		self.dwriter.writerow(frame[0])

	def finalize(self):
		self.outfile.close()


class CombinedWriter(BaseProcessor):
	def __init__(self, filename):
		super().__init__()
		self.filename = filename

		self.drone_accum = []
		self.cam_accum = []
		self.data_accum = []

		self.next_timestamp = 0
		self.writer = None

	def handle_frame(self, frame):
		if frame[2] == DATA_SOURCE:
			self.data_accum.append(frame)
		elif frame[2] == DRONE_VIDEO_SOURCE:
			self.drone_accum.append(frame)
		else:
			self.cam_accum.append(frame)

		if len(self.drone_accum) < 2 or len(self.cam_accum) < 2 or len(self.data_accum) < 2:
			return

		upto_ts = min(self.drone_accum[-1][1], self.cam_accum[-1][1], self.data_accum[-1][1])

		if upto_ts < self.next_timestamp:
			return

		if self.writer is None:
			drone_height, _, _ = self.drone_accum[0][0].shape
			cam_height, cam_width, _ = self.cam_accum[0][0].shape
			self.writer = cv2.VideoWriter(self.filename, cv2.VideoWriter_fourcc(*'avc1'), 30, (cam_width, drone_height + cam_height))


		dind = 0
		cind = 0
		dataind = 0
		for i in np.arange(self.next_timestamp, upto_ts, 1 / 30):
			while ((dind < (len(self.drone_accum) - 1) and self.drone_accum[dind][1] < i)):
				dind += 1
			while ((cind < (len(self.cam_accum) - 1) and self.cam_accum[cind][1] < i)):
				cind += 1
			while ((dataind < (len(self.data_accum) - 1) and self.data_accum[dataind][1] < i)):
				dataind += 1

			combined = _make_combined_frame(self.drone_accum[dind][0], self.cam_accum[cind][0], self.data_accum[dataind][0])
			self.writer.write(combined)
		self.next_timestamp = i

		self.drone_accum = self.drone_accum[dind:]
		self.cam_accum = self.cam_accum[cind:]
		self.data_accum = self.data_accum[dataind:]

	def finalize(self):
		if self.writer is not None:
			self.writer.release()


@contextmanager
def start_flight(
	fps,
	output_dir):

	os.makedirs(output_dir, exist_ok=True)

	tello = Tello(retry_count=2)
	tello.connect()
	tello.streamon()

	start_time = time.time()

	dvm = VideoMonitor(tello.get_udp_video_address(), start_time, fps, DRONE_VIDEO_SOURCE, ts_offset=0.4)
	cvm = VideoMonitor(WEBCAM_SRC, start_time, fps, CAM_VIDEO_SOURCE)
	dm = DataMonitor(start_time, fps, tello)

	combined_writer = CombinedWriter(output_dir + '/combined.mp4')
	dvm.attach(combined_writer)
	dvm.attach(VideoWriter(output_dir + '/drone_video.mp4', fps))

	cvm.attach(combined_writer)
	cvm.attach(VideoWriter(output_dir + '/cam_video.mp4', fps))

	dm.attach(combined_writer)
	dm.attach(DataWriter(os.path.join(output_dir, 'data.csv')))

	dvm.start()
	cvm.start()
	dm.start()

	time.sleep(5)
	try:
		print("yielding")
		yield (tello, dvm, cvm, dm)

	except Exception as e:
		log.exception("Exception in flight execution")
		raise

	finally:
		dvm.stop()
		cvm.stop()
		dm.stop()

		dvm.joinall()
		cvm.joinall()
		dm.joinall()

		print("monitor finished")

		tello.end()
		print("tello ended")
