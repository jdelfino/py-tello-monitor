import cv2
import logging
import time
from threading import Thread
import csv
from collections import OrderedDict
import numpy as np
import os
import math
from djitellopy import Tello
from contextlib import contextmanager,redirect_stderr,redirect_stdout
import multiprocessing as mp


def _writeFrames(frame_timestamps, filename):
    duration = frame_timestamps[-1][1] - frame_timestamps[0][1]
    print("Writing video buffer to '%s', frames = %d, duration = %.2f" % (filename, len(frame_timestamps), duration))
    fps = len(frame_timestamps) / duration
    height, width, _ = frame_timestamps[0][0].shape
    video = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'avc1'), fps, (width, height))

    font = cv2.FONT_HERSHEY_SIMPLEX
    for frame, timestamp in frame_timestamps:
        cv2.putText(frame, "%.2f" % timestamp, (50, 75), font, 1, (0, 255, 255), 4)
        video.write(frame)

    video.release()

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

def _write_combined(drone_frames, cam_frames, data_frames, filename):
    # make combined side-by-side video
    # assumption: drone cam is narrower than webcam, and data can fit into that extra width
    drone_height, _, _ = drone_frames[0][0].shape
    cam_height, cam_width, _ = cam_frames[0][0].shape

    # mp4v
    video = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'avc1'), 30, (cam_width, drone_height + cam_height))

    start = min(drone_frames[0][1], cam_frames[0][1])
    stop = max(drone_frames[-1][1], cam_frames[-1][1])
    total_duration = stop - start

    print("Writing combined video to '%s', frames = %d / %d, duration = %.2f" % (filename, len(drone_frames), len(cam_frames), total_duration))

    dind = 0
    cind = 0
    dataind = 0

    for i in np.arange(start, stop, 1 / 30):
        while ((dind < (len(drone_frames) - 1) and drone_frames[dind][1] < i)):
            dind += 1
        while ((cind < (len(cam_frames) - 1) and cam_frames[cind][1] < i)):
            cind += 1
        while ((dataind < (len(data_frames) -1) and data_frames[dataind]['time_s'] < i)):
            dataind += 1

        combined = _make_combined_frame(drone_frames[dind][0], cam_frames[cind][0], data_frames[dataind])
        video.write(combined)
    video.release()     

def wrap_send_command(tello, m):
    ''' 
    Injects monitoring around the function that sends commands to the drone, so the commands
    show up in the captured data. 
    '''
    def wrap(f):
        def wrapped(command: str, timeout: int = Tello.RESPONSE_TIMEOUT) -> bool:
            print("Executing command: {}".format(command))
            m.recordCommand(command)
            return f(tello, command, timeout)
        return wrapped
    # The check is to prevent double wrapping
    if not getattr(tello, 'orig_send_control_command', None):
        setattr(tello, 'orig_send_control_command', Tello.__dict__['send_control_command'])
        setattr(tello, 'send_control_command', wrap(Tello.__dict__['send_control_command']))

@contextmanager
def monitored_tello(output_dir, tello=None, webcam_src=1):
    '''
    Usage example:
    with monitored_tello('data_output') as tello:
        tello.takeoff()
        tello.move_up(50)
        tello.land()
    '''

    if tello is None:
        tello = Tello()

    m = DroneMonitor(webcam_src)

    tello.connect()
    m.start(tello, output_dir)
    print("Monitoring...")

    try:
        yield tello
    finally:
        m.stop()
        tello.end()

@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err:
            yield err
            
class DroneMonitor:
    def __init__(self, webcam_src):
        self.webcam_src = webcam_src
        self._reset()

    def _reset(self):
        self.output_dir = None
        self.keep_recording = False
        self.command_queue = []

        self.camFrames = []
        self.droneFrames = []
        self.dataFrames = []

        self.start_time = 0

        self.camRecorder = None
        self.droneRecorder = None
        self.dataRecorder = None

    def _recordWebcam(self):
        with suppress_stdout_stderr():
            cap = cv2.VideoCapture(self.webcam_src)
            while self.keep_recording:
                ret, frame = cap.read()
                if not ret:
                    print("Read failed")
                    break
                self.camFrames.append((frame.copy(), time.time() - self.start_time))
                time.sleep(1 / 30)

            _writeFrames(self.camFrames, os.path.join(self.output_dir, "cam_video.mp4"))
            cap.release()

    def _recordDroneVideo(self, tello):
        with suppress_stdout_stderr():
            frame_read = tello.get_frame_read()         
            while self.keep_recording:
                # slight lag on drone camera, fudge it by subtracting 0.25s
                self.droneFrames.append((frame_read.frame.copy(), time.time() - self.start_time - 0.25))
                time.sleep(1 / 30)

            _writeFrames(self.droneFrames, os.path.join(self.output_dir, "drone_video.mp4"))

    def _recordData(self, tello):
        datapoints = OrderedDict([
            ('time_s', lambda: time.time() - self.start_time),
            ('acceleration_x', tello.get_acceleration_x),
            ('acceleration_y', tello.get_acceleration_y),
            ('acceleration_z', tello.get_acceleration_z),
            ('barometer', tello.get_barometer),
            ('battery', tello.get_battery),
            ('flight_time', tello.get_flight_time),
            ('height_cm', tello.get_height),
            ('to_floor_cm', tello.get_distance_tof),
            ('pitch_degrees', tello.get_pitch),
            ('roll_degrees', tello.get_roll),
            ('speed_x', tello.get_speed_x),
            ('speed_y', tello.get_speed_y),
            ('speed_z', tello.get_speed_z),
            ('temperature_c', tello.get_temperature),
            ('yaw_degrees', tello.get_yaw),
            ('command', lambda: (self.command_queue.pop() if len(self.command_queue) else None))
        ])

        with open(os.path.join(self.output_dir, 'data.csv'), 'w', newline='') as csvfile:
            dwriter = csv.DictWriter(csvfile, fieldnames=datapoints)
            dwriter.writeheader()

            while self.keep_recording:
                to_write = {}
                for k, get_fn in datapoints.items():
                    to_write[k] = get_fn()

                self.dataFrames.append(to_write)
                dwriter.writerow(to_write)
                time.sleep(1 / 30)

    def recordCommand(self, command):
        self.command_queue.append(command)

    def start(self, tello, output_dir):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        wrap_send_command(tello, self)

        self.keep_recording = True

        tello.streamon()

        self.start_time = time.time()

        self.camRecorder = Thread(target=self._recordWebcam)
        self.camRecorder.start()

        self.droneRecorder = Thread(target=lambda: self._recordDroneVideo(tello))
        self.droneRecorder.start()

        self.dataRecorder = Thread(target=lambda: self._recordData(tello))
        self.dataRecorder.start()

        # Ugly, but give the video capture some time to start
        time.sleep(5)

    def stop(self):
        self.keep_recording = False
        self.camRecorder.join()
        self.droneRecorder.join()
        self.dataRecorder.join()

        _write_combined(self.droneFrames, self.camFrames, self.dataFrames, os.path.join(self.output_dir, "combined.mp4"))
        self._reset()
