import RPi.GPIO as IO
import numpy as np
import time
import multiprocessing as mp
import threading
import pickle
import TempKinematics as TK
import socket

HOST = "192.168.1.7"
PORT = 65432


IO.setwarnings(False)
IO.setmode(IO.BCM)
IO.cleanup()
IO.setup(2, IO.OUT)
IO.setup(3, IO.OUT)
IO.setup(4, IO.OUT)
IO.setup(14, IO.OUT)
IO.setup(15, IO.OUT)


class Board:
    def __init__(self, height, width, motorA, motorB):
        self.height = height
        self.width = width
        self.motorA = motorA
        self.motorB = motorB
        self.current_position = (50, 30)
        self.P0 = (0, 0)
        self.P1 = (0, 0)
        self.P3 = (0, 0)
        self.P4 = (0, 0)
        self.pointer = Pointer(15)
        print("Board created!!!")

    def saveParameters(self):
        parameters = np.array(
            [
                self.motorA.angle,
                self.motorB.angle,
                self.width,
                self.P0[0],
                self.P0[1],
                self.P1[0],
                self.P1[1],
                self.P3[0],
                self.P3[1],
                self.P4[0],
                self.P4[1],
            ]
        )
        np.save("parameters", parameters)
        print("Parameters saved!!!")

    def loadParameters(self):
        p = np.load("parameters.npy")
        self.motorA.angle = p[0]
        self.motorB.angle = p[1]
        self.width = p[2]
        self.P0 = (p[3], p[4])
        self.P1 = (p[5], p[6])
        self.P3 = (p[7], p[8])
        self.P4 = (p[9], p[10])
        print("Parameters loaded!!!")

    def len2coord(self, lens):
        (lA, lB) = lens
        r = lA * np.sqrt(
            1
            - ((lA * lA + self.width * self.width - lB * lB) / (2 * self.width * lA))
            ** 2
        )
        c = (lA * lA + self.width * self.width - lB * lB) / (2 * self.width)
        return (r, c)

    def coord2len(self, coord):
        (r, c) = coord
        lA = np.sqrt(r * r + c * c)
        lB = np.sqrt(r * r + (self.width - c) ** 2)
        return (lA, lB)

    def locate(self, r, c):
        lA = np.sqrt((r) ** 2 + (c) ** 2)
        lB = np.sqrt((r) ** 2 + (self.width - c) ** 2)

        angleA = (lA / (2 * np.pi * self.motorA.radius)) * -360
        angleB = (lB / (2 * np.pi * self.motorB.radius)) * 360
        deltaAngleA = angleA - self.motorA.angle
        deltaAngleB = angleB - self.motorB.angle

        p1 = threading.Thread(target=self.motorA.move, args=(deltaAngleA,))
        p2 = threading.Thread(target=self.motorB.move, args=(deltaAngleB,))
        p1.start()
        p2.start()
        p1.join()
        p2.join()

    def drawContours(self, contours, path_rlen, path_clen, rs=0, re=0, cs=0, ce=0):
        i = 0
        self.pointer.close()
        print("Moving to the first point...")
        time.sleep(0.1)
        for contour in contours:
            print("Contour {} out of {}...".format(i, len(contours)))
            i += 1
            time.sleep(0.5)
            path = np.squeeze(contour, 1)
            path = path.tolist()
            self.moveInPath(path, path_rlen, path_clen, rs, re, cs, ce)
            time.sleep(0.5)

    def moveInPath(self, path, path_rlen, path_clen, rs=0, re=0, cs=0, ce=0):
        rlen = re - rs
        clen = ce - cs
        fx = min(clen / path_clen, rlen / path_rlen)
        self.pointer.close()
        time.sleep(0.5)
        self.locate(rs + path[0][1] * fx, cs + path[0][0] * fx)
        self.pointer.open()
        time.sleep(0.5)
        for i in range(len(path)):
            self.locate(rs + path[i][1] * fx, cs + path[i][0] * fx)
        self.pointer.close()

    def wifi(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((HOST, PORT))
            print("listening")
            s.listen()
            conn, addr = s.accept()
            with conn:
                print("Connected by", addr)
                while True:
                    data = conn.recv(1024)
                    if not data:
                        break
                    conn.sendall(data)
                    data = data.decode("utf-8")
                    data = data.split()
                    self.menu(input=data[0], args=data[1:])

    def menu(self, input=-1, args=None):
        while True:
            if input == -1:
                text = input("?")
            else:
                text = input
            if text == "q":
                print("Calibrate finished!!!")
                break
            elif text == "aaa":
                self.motorA.move(-360)
            elif text == "sss":
                self.motorA.move(360)
            elif text == "aa":
                self.motorA.move(-45)
            elif text == "ss":
                self.motorA.move(45)
            elif text == "a":
                self.motorA.move(-5)
            elif text == "s":
                self.motorA.move(5)

            elif text == "Calibrate":
                # l1 = float(input('LA = '))
                # l2 = float(input('LB = '))
                # w = float(input('W = '))
                w = 68
                self.motorA.angle = (74) / (2 * np.pi * self.motorA.radius) * -360
                self.motorB.angle = (74) / (2 * np.pi * self.motorB.radius) * 360

            elif text == "Locate":
                r = float(input("r = "))
                c = float(input("c = "))
                self.locate(r, c)

            elif text == "Open":
                self.pointer.open()

            elif text == "Close":
                self.pointer.close()

            elif text == "Done":
                print("Calibration finished!")
                break

            elif text == "Save":
                self.saveParameters()

            elif text == "Load":
                self.loadParameters()

            elif text == "Image":
                if input == -1:
                    image = np.load("S17.npy")
                    rs, re, cs, ce = 45, 75, 20, 60
                    self.rectangle7(
                        rs,
                        re,
                        cs,
                        ce,
                        image,
                        8,
                        0.15,
                        skip_zeros=True,
                        max_color=20,
                        zero_threshold=14,
                    )
                else:
                    image = np.load(args[4])
                    self.rectangle7(
                        float(args[0]),
                        float(args[1]),
                        float(args[2]),
                        float(args[3]),
                        image,
                        8,
                        float(args[5]),
                    )

            elif text == "ddd":
                self.motorB.move(-360)
            elif text == "fff":
                self.motorB.move(360)
            elif text == "dd":
                self.motorB.move(-45)
            elif text == "ff":
                self.motorB.move(45)
            elif text == "d":
                self.motorB.move(-5)
            elif text == "f":
                self.motorB.move(5)

            if input != -1:
                break

    def setStepDelays(self, delay_time):
        self.motorA.delay_time = delay_time
        self.motorB.delay_time = delay_time

    def drawImage(self, rs, re, cs, ce, image, pixel_height=0.1):
        _, minlB = self.coord2len((rs, ce))
        _, MaxlB = self.coord2len((re, cs))
        white_delay = 0.004
        black_delay = 0.010
        pointer_delay = 0.15

        current_lB = minlB + pixel_height
        counter = 0
        while True:
            point1, point2 = self.findPoints(current_lB, rs, re, cs, ce)
            lAstart, _ = self.coord2len(point1)
            lAend, _ = self.coord2len(point2)
            colors = []
            current_lA = lAstart
            black_flag = False
            bs = 0
            be = 0
            blacklist = []
            while True:
                normalized_row, normalized_col = self.len2coord(
                    (current_lA, current_lB)
                )
                normalized_row, normalized_col = (normalized_row - rs) / (re - rs), (
                    normalized_col - cs
                ) / (ce - cs)
                image_row, image_col = int(normalized_row * image.shape[0]), int(
                    normalized_col * image.shape[1]
                )
                if image_row == image.shape[0]:
                    image_row = image.shape[0] - 1
                if image_col == image.shape[1]:
                    image_col = image.shape[1] - 1
                if image[image_row, image_col] == 0 and black_flag == False:
                    black_flag = True
                    bs = current_lA
                elif image[image_row, image_col] != 0 and black_flag == True:
                    black_flag = False
                    be = current_lA - pixel_height
                    blacklist.append((bs, be))
                current_lA = current_lA - 0.005
                if current_lA <= lAend:
                    break
            lB = current_lB
            if len(blacklist):
                counter += 1

                if counter % 2:
                    for i in range(len(blacklist)):
                        self.pointer.close()
                        time.sleep(pointer_delay)
                        (r1, c1) = self.len2coord((blacklist[i][0], lB))
                        (r2, c2) = self.len2coord((blacklist[i][1], lB))
                        self.setStepDelays(white_delay)
                        self.locate(r1, c1)
                        self.setStepDelays(black_delay)
                        time.sleep(pointer_delay)
                        self.pointer.open()
                        time.sleep(pointer_delay)
                        self.locate(r2, c2)
                else:
                    for i in range(len(blacklist) - 1, -1, -1):
                        self.pointer.close()
                        time.sleep(pointer_delay)
                        (r1, c1) = self.len2coord((blacklist[i][1], lB))
                        (r2, c2) = self.len2coord((blacklist[i][0], lB))
                        self.setStepDelays(white_delay)
                        self.locate(r1, c1)
                        self.setStepDelays(black_delay)
                        time.sleep(pointer_delay)
                        self.pointer.open()
                        time.sleep(pointer_delay)
                        self.locate(r2, c2)

            current_lB = current_lB + pixel_height
            if current_lB > MaxlB:
                print("finish")
                break
        print("Done")

    def findPoints(self, lB, rs, re, cs, ce):
        width = self.width
        dist = lambda x, y: np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)
        findRow = (
            lambda c, lB: np.sqrt(lB * lB - (c - width) * (c - width))
            if (lB * lB - (c - width) * (c - width) > 0)
            else -1
        )
        findCol = (
            lambda r, lB: -np.sqrt(lB * lB - r * r) + width
            if lB * lB - r * r > 0
            else -1
        )
        findCloser = lambda x, y: x if dist(x, p) < dist(y, p) else y
        point1 = (findRow(ce, lB), ce) if findRow(ce, lB) != -1 else None
        point2 = (rs, findCol(rs, lB)) if findCol(rs, lB) != -1 else None
        point3 = (re, findCol(re, lB)) if findCol(re, lB) != -1 else None
        point4 = (findRow(cs, lB), cs) if findRow(cs, lB) != -1 else None
        check = (
            lambda point: True
            if point[0] >= rs and point[0] <= re and point[1] >= cs and point[1] <= ce
            else False
        )
        print(point1, point2, point3, point4)
        result1, result2 = None, None
        if check(point1):
            result1 = point1
        elif check(point3):
            result1 = point3
        else:
            return None, None
        if check(point2):
            result2 = point2
        elif check(point4):
            result2 = point4
        else:
            return None, None
        return result1, result2


class Motor:
    def __init__(self, step_pin, dir_pin):
        self.step_pin = step_pin
        self.dir_pin = dir_pin
        self.radius = 0.9955
        self.height = 50
        self.width = 55
        self.length = 27.58
        self.angle = (self.length / (2 * np.pi * self.radius)) * 360
        self.delay_time = 0.015
        print("motor angle:", self.angle)

    def move(self, degree):
        steps = int(degree / 1.8)
        self.angle = self.angle + steps * 1.8
        if steps == abs(steps):
            IO.output(self.dir_pin, IO.LOW)
        else:
            IO.output(self.dir_pin, IO.HIGH)
        IO.output(self.step_pin, IO.LOW)
        for i in range(abs(steps)):
            IO.output(self.step_pin, IO.HIGH)
            time.sleep(self.delay_time)
            IO.output(self.step_pin, IO.LOW)
            time.sleep(self.delay_time)


class Pointer:
    def __init__(self, pin):
        IO.setup(pin, IO.OUT)
        self.pointer = IO.PWM(pin, 50)
        self.pointer.start(0)
        self.counter = 0

    def open(self):
        self.counter += 1
        if self.counter > 3:
            self.force()
            time.sleep(0.15)
            self.pointer.ChangeDutyCycle(7.5)
            self.counter = 0
            return
        self.pointer.ChangeDutyCycle(7.5)

    def close(self):
        self.pointer.ChangeDutyCycle(12.5)

    def force(self):
        self.pointer.ChangeDutyCycle(2.5)


motorA = Motor(2, 3)
motorB = Motor(4, 14)
motorB.length = 39.82
motorB.angle = (motorB.length / (2 * np.pi * motorB.radius)) * 360
motorA.angle = motorA.angle * -1
print("motor angles:", motorA.angle, motorB.angle)
height = 69
width = 69
path = [(25, 28), (22, 28), (25, 25)]
pathUp = [(i, 25) for i in np.arange(25, 17, -0.1)]
pathDown = [(i, 25) for i in np.arange(25, 30, 0.1)]
pathLeft = [(25, i) for i in np.arange(25, 35, 0.5)]
pathRight = [(25, i) for i in np.arange(25, 23, -0.1)]

board = Board(height, width, motorA, motorB)
r = 15
# board.calibrate()
board.wifi()
circle_path = [
    (25 + r * np.sin(theta), 25 + r * np.cos(theta))
    for theta in np.arange(0, 2 * np.pi + np.pi / 6, 0.05)
]
square_lower = 10
square_upper = 40
square_path = [(square_lower, i) for i in np.arange(square_lower, square_upper, 1)]
square_path.extend(
    [(i, square_upper) for i in np.arange(square_lower, square_upper, 1)]
)
square_path.extend(
    [(square_upper, i) for i in np.arange(square_upper, square_lower, -1)]
)
square_path.extend(
    [(i, square_lower) for i in np.arange(square_upper, square_lower, -1)]
)

f = open("AUT.pickle", "rb")
contours = pickle.load(f)
f.close()
print(len(contours))
time.sleep(3)
board.drawContours(contours, 600, 800, 53, 82, 18, 47)
print("Finished")
