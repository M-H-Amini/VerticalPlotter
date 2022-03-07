import sys
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QPushButton, QMainWindow, QApplication, QVBoxLayout, QFileDialog, QWidget, QInputDialog, QMessageBox
from PyQt5 import QtCore
from PyQt5.QtCore import QThread
from PyQt5 import uic
import time
import os
import cv2
import numpy as np
import ContoursOpenCV as cnt
import socket
import paramiko
from paramiko import SSHClient
from scp import SCPClient
from sklearn.cluster import KMeans
import threading


def button_clicked():
    print('hi')

Form = uic.loadUiType(os.path.join(os.getcwd(), 'TempGUI.ui'))[0]

class Matplotlib(Form, QMainWindow):
    def __init__(self):
        Form.__init__(self)
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.IP, ok = QInputDialog.getText(self, 'Connection', 'Enter the robot IP address...', text='192.168.43.247')
        self.connectText.setText(self.IP)
        #self.connectButtonClicked()
        time.sleep(1)
        self.thresh_file_name = None
        self.connectButtonClicked()
        #self.connection = Connection('192.168.1.7', 65432)
        #self.connection.start()
        print('Connected to the raspberry...')
        self.blank = np.ones((800, 600))*255
        self.axes_no = 2*2
        self.rows , self.cols = 2, 2
        '''
        self.fig = Figure()
        self.ax = self.fig.add_axes([0.1, 0.1, 0.8, 0.8])
        '''
        self.fig, self.axes = plt.subplots(2, 2)
        self.canvas = FigureCanvas(self.fig)
        self.navi = NavigationToolbar(self.canvas, self)
        l = QVBoxLayout(self.matplotlib_widget)
        l.addWidget(self.canvas)
        l.addWidget(self.navi)

        self.seg_fig, self.seg_axes = plt.subplots(1, 2)
        self.seg_canvas = FigureCanvas(self.seg_fig)
        self.seg_navi = NavigationToolbar(self.seg_canvas, self)
        seg_l = QVBoxLayout(self.preLib)
        seg_l.addWidget(self.seg_canvas)
        seg_l.addWidget(self.seg_navi)
        self.seg_axes[0].set_title('Original')
        self.seg_axes[1].set_title('Segmented\nNo: {}'.format(self.segmentText.text()))
        self.seg_axes[0].set_xticks([])
        self.seg_axes[0].set_yticks([])
        self.seg_axes[1].set_xticks([])
        self.seg_axes[1].set_yticks([])

        self.thresh_fig, self.thresh_axes = plt.subplots(1, 2)
        self.thresh_canvas = FigureCanvas(self.thresh_fig)
        self.thresh_navi = NavigationToolbar(self.thresh_canvas, self)
        thresh_l = QVBoxLayout(self.threshLib)
        thresh_l.addWidget(self.thresh_canvas)
        thresh_l.addWidget(self.thresh_navi)
        self.thresh_axes[0].set_title('Original')
        self.thresh_axes[1].set_title('Thresholded\nVal: {}'.format(self.thresholdSlider.value()))
        self.thresh_axes[0].set_xticks([])
        self.thresh_axes[0].set_yticks([])
        self.thresh_axes[1].set_xticks([])
        self.thresh_axes[1].set_yticks([])

        self.contour_fig, self.contour_axes = plt.subplots(1, 2)
        self.contour_canvas = FigureCanvas(self.contour_fig)
        self.contour_navi = NavigationToolbar(self.contour_canvas, self)
        contour_l = QVBoxLayout(self.contourLib)
        contour_l.addWidget(self.contour_canvas)
        contour_l.addWidget(self.contour_navi)
        self.contour_axes[0].set_title('Original')
        self.contour_axes[1].set_title('Contourized')
        self.contour_axes[0].set_xticks([])
        self.contour_axes[0].set_yticks([])
        self.contour_axes[1].set_xticks([])
        self.contour_axes[1].set_yticks([])

        x = np.linspace(0, np.pi * 2, 1000)
        self.calibrateButton.clicked.connect(self.calibrate)
        self.pointerOnButton.clicked.connect(self.pointerOn)
        self.pointerOffButton.clicked.connect(self.pointerOff)
        self.blackDelayButton.clicked.connect(self.blackDelay)
        self.whiteDelayButton.clicked.connect(self.whiteDelay)
        self.pointerDelayButton.clicked.connect(self.pointerDelay)
        self.motorAButton.clicked.connect(self.motorA)
        self.motorBButton.clicked.connect(self.motorB)
        self.importButton.clicked.connect(self.importButtonClicked)
        #self.threshSaveButton.clicked.connect(self.threshSaveButtonClicked)
        self.threshPrintButton.clicked.connect(self.threshPrintButtonClicked)
        self.connectButton.clicked.connect(self.connectButtonClicked)
        self.captureButton.clicked.connect(self.captureButtonClicked)
        self.thresholdSlider.sliderReleased.connect(self.thresholdChanged)
        self.contourEdit.returnPressed.connect(self.contourChanged)
        self.denoiseCheck.stateChanged.connect(self.denoiseCheckClicked)
        self.segmentCheck.stateChanged.connect(self.segmentCheckClicked)
        self.plotThread = None
        self.image = None
        self.image_path = None
        self.axes_images = [[None, None], [None, None]]
        self.seg_axes_images = [None, None]
        self.thresh_axes_images = [None, None]
        self.contour_axes_images = [None, None]
        #self.axes_images[0][0] = self.axes[0][0].imshow(self.blank)
        for i in range(self.rows):
            for j in range(self.cols):
                self.axes[i][j].set_xticks([])
                self.axes[i][j].set_yticks([])

    def captureButtonClicked(self):

        cap = cv2.VideoCapture(0)
        cascPath = 'haarcascade_frontalface_default.xml'

        faceCascade = cv2.CascadeClassifier(cascPath)

        while (True):
            _, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            for (x, y, w, h) in faces:
                raw_frame = frame.copy()
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow('MHA', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.imwrite('capture.png' ,raw_frame)
                QMessageBox.about(self, 'Image captured!', 'The image is saved in the same folder in capture.png file!!!')
                self.image_path = 'capture.png'
                self.axes_images = [[None, None], [None, None]]
                self.loadImage()
                self.thresholdSlider.setEnabled(True)
                break

        cap.release()
        cv2.destroyAllWindows()

    def threshPrintButtonClicked(self):
        if self.thresh_file_name is not None:
            self.copyfile(self.IP, 'pi', 'raspberry', self.thresh_file_name+'.npy')
        else:
            self.threshSaveButtonClicked()
            self.copyfile(self.IP, 'pi', 'raspberry', self.thresh_file_name+'.npy')
        QMessageBox.about(self, 'Drawing Thresholded Image', 'The robot will now start drawing...')
        '''
        message = 'Image 45 75 20 60 {} 0.2'.format(self.thresh_file_name.split('/')[-1]+'.npy')
        print('message...', message)
        time.sleep(3)
        self.connection.send(message)
        print('here')
        '''
        message = 'Image 45 75 20 60 me8.npy 0.2'
        self.connection.send(message)
        print('here')

    def copyfile(self, host, user, password, source, dest='/home/pi/Downloads/Project 4/'):
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(host, 22, user, password)
        scp = SCPClient(client.get_transport())
        scp.put(source, recursive=True, remote_path=dest)
        scp.close()

    def ssh(self, host, user, password, command):
        ssh = SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        print(ssh.load_system_host_keys())
        ssh.connect(host, username=user, password=password)
        ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(command)
        print(ssh_stdout.read().decode('ascii').strip("\n"))

    def denoiseCheckClicked(self, state):
        if state == QtCore.Qt.Checked:
            self.pre_image = cv2.GaussianBlur(self.gray,(5,5),0)
            self.seg_axes_images[1].set_data(self.pre_image)
            self.seg_fig.canvas.draw()
        else:
            #self.pre_image = self.rgb_image.copy()
            self.seg_axes_images[1].set_data(self.gray)
            self.seg_fig.canvas.draw()

    def segmentCheckClicked(self, state):
        if state == QtCore.Qt.Checked:
            self.segmentStateLabel.setText('Please wait...')
            self.pre_image, _, _ = self.kmeans_seg_gray(self.gray, int(self.segmentText.text()))
            self.seg_axes_images[1].set_data(self.pre_image)
            self.seg_fig.canvas.draw()
            self.segmentStateLabel.setText('Done.')
        else:
            self.seg_axes_images[1].set_data(self.gray)
            self.seg_fig.canvas.draw()

    def connectButtonClicked(self):
        self.IP = self.connectText.text()
        command = 'python3 Downloads/Project\ 4/LEDs.py'
        self.sshThread = threading.Thread(target=self.ssh, args=(self.IP, 'pi', 'raspberry', command,))
        self.sshThread.start()
        print('here')
        time.sleep(3)
        self.connection = Connection('192.168.43.247', 65432)
        self.connection.start()

    def blackDelay(self):
        message = 'BlackDelay {}'.format(self.blackDelayText.text())
        self.connection.send(message)

    def whiteDelay(self):
        message = 'WhiteDelay {}'.format(self.whiteDelayText.text())
        self.connection.send(message)

    def pointerDelay(self):
        message = 'PointerDelay {}'.format(self.pointerDelayText.text())
        self.connection.send(message)

    def pointerOn(self):
        message = 'Open'
        self.connection.send(message)

    def pointerOff(self):
        message = 'Close'
        self.connection.send(message)

    def calibrate(self):
        text, ok = QInputDialog.getText(self, 'Calibration', 'Enter lA, lB, width with spaces: (ex: 74 74 69)')
        if ok:
            message = text.split()
            print(message)
            message = 'Calibrate {} {} {}'.format(message[0], message[1], message[2])
            self.connection.send(message)

    def motorA(self):
        angle = self.getDouble('MotorA','What angle to rotate?')
        message = 'A {}'.format(angle)
        self.connection.send(message)

    def motorB(self):
        angle = self.getDouble('MotorB', 'What angle to rotate?')
        message = 'B {}'.format(angle)
        self.connection.send(message)

    def getDouble(self, title, message):
        d, okPressed = QInputDialog.getDouble(self, title, message, 10.05, -360, 360, 5)
        if okPressed:
            return d

    def kmeans_seg_gray(self, image, k):
        ##  Clustering...
        kmeans = KMeans(n_clusters=k, random_state=0) \
            .fit(np.reshape(image, (-1, 1)))
        centers = kmeans.cluster_centers_
        mask = np.reshape(kmeans.labels_, image.shape)

        ##  Reconstructing...
        segmented_image = mask.copy()
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                segmented_image[i, j] = centers[mask[i, j], 0]
        segmented_image = segmented_image.astype(np.uint8)

        return segmented_image, centers, mask

    def loadImage(self):
        self.image = cv2.imread(self.image_path)
        self.rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.progressBar.setValue(10)
        if self.axes_images[0][0] is None:
            self.axes_images[0][0] = self.axes[0][0].imshow(self.rgb_image)
            self.progressBar.setValue(30)
            self.seg_axes_images[0] = self.seg_axes[0].imshow(self.gray, cmap='gray')
            self.thresh_axes_images[0] = self.thresh_axes[0].imshow(self.rgb_image)
            self.progressBar.setValue(40)
            self.contour_axes_images[0] = self.contour_axes[0].imshow(self.rgb_image)
            self.progressBar.setValue(50)
            self.blank = np.ones((self.image.shape[0], self.image.shape[1])) * 255
            for i in range(self.rows):
                for j in range(self.cols):
                    if i!=0 or j!=0:
                        self.axes_images[i][j] = self.axes[i][j].imshow(self.rgb_image, cmap='gray')
                        self.axes_images[i][j].set_data(self.blank)
            self.progressBar.setValue(70)
            self.seg_axes_images[1] = self.seg_axes[1].imshow(self.rgb_image, cmap='gray')
            self.seg_axes_images[1].set_data(self.blank)
            self.thresh_axes_images[1] = self.thresh_axes[1].imshow(self.rgb_image, cmap='gray')
            self.thresh_axes_images[1].set_data(self.blank)
            self.progressBar.setValue(80)
            self.contour_axes_images[1] = self.contour_axes[1].imshow(self.rgb_image, cmap='gray')
            self.contour_axes_images[1].set_data(self.blank)
            self.progressBar.setValue(90)
            self.axes_images[0][0].set_data(self.rgb_image)
            self.thresh_axes_images[0].set_data(self.rgb_image)
            self.contour_axes_images[0].set_data(self.rgb_image)
            self.progressBar.setValue(100)
        else:
            self.axes_images[0][0].set_data(self.rgb_image)

        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.contour_fig.canvas.draw()
        self.seg_fig.canvas.draw()
        self.thresh_fig.canvas.draw()
        self.fig.canvas.draw()

    def showSaveDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(None, "Choose a location to save the file...", "",
                                                  "Threshold Files (*.npy)", options=options)
        return fileName

    def threshSaveButtonClicked(self):
        self.thresh_file_name = self.showSaveDialog()
        np.save(self.thresh_file_name, self.thresh)

    def contourChanged(self):
        contour_values = [int(i) for i in self.contourEdit.text().split(',')]
        print(contour_values)
        result, contours = cnt.contourize(self.image, contour_values, 1, self.image_path, False)
        print('result shape', result.shape)
        self.contour_axes_images[1].set_data(result)
        self.axes_images[1][0].set_data(result)
        self.fig.canvas.draw()
        self.contour_fig.canvas.draw()

    def importButtonClicked(self):
        self.axes_images = [[None, None], [None, None]]
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(None, "Choose your image...", "",
                                                  "All Files (*);;Image Files (*.jpg)", options=options)
        self.image_path = fileName
        if fileName:
            print(fileName)
            self.loadImage()
            self.thresholdSlider.setEnabled(True)

    def thresholdChanged(self):
        #self.connection.send('sss')
        self.thresholdThread = ThresholdThread(self.image, self.thresholdSlider.value())
        self.thresholdThread.update_thresh_signal.connect(self.update_thresh)
        self.thresh_axes[1].set_title('Thresholded\nVal: {}'.format(self.thresholdSlider.value()))
        self.thresholdThread.start()

    def update_thresh(self, thresh):
        self.thresh = thresh
        self.axes_images[0][1].set_data(thresh)
        self.thresh_axes_images[1].set_data(thresh)
        self.fig.canvas.draw()
        self.thresh_fig.canvas.draw()


    def button_clicked(self):
        num = np.random.random() * 100
        self.lineEdit.setText(str(num))
        self.progressBar.setValue(num)
        x = np.linspace(0, np.pi * 2, 1000)
        #self.line1.set_data(x, np.sin(num * x))
        self.fig.canvas.draw()
        self.plotThread = PlotThread(num)
        self.plotThread.update_plot.connect(self.update)
        self.plotThread.start()


    def update(self, x, y):
        self.axes[0][0].imshow(self.image)
        self.fig.canvas.draw()

class Connection(QThread):
    def __init__(self, host, port):
        QThread.__init__(self)
        self.host = host
        self.port = port  # The port used by the server

    def run(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((self.host, self.port))
        print('Connected...')
    def send(self, x):
            #while True:
                # s.sendall(b'Hello, world')
                # data = s.recv(1024)
                # s.sendall(b'Thanks ALLAH')
                # data = s.recv(1024)
                #x = input('')
            #if x == 'q':
            #    break
            self.s.sendall(bytes(x, 'utf-8'))
                #data = s.recv(1024)
                # if not data:
                #     print('break')
                #     break
                #print(data, '^^^')

        #print('Received', repr(data))


class ThresholdThread(QThread):
    update_thresh_signal = QtCore.pyqtSignal(np.ndarray)
    def __init__(self,image, threshold):
        QThread.__init__(self)
        self.image = image
        self.threshold = threshold
        self.thresh = None

    def run(self):
        _, self.thresh = cv2.threshold(self.image, self.threshold, 255, 0)
        self.update_thresh_signal.emit(self.thresh)

class PlotThread(QThread):
    update_plot = QtCore.pyqtSignal(np.ndarray, np.ndarray)
    def __init__(self, num):
        QThread.__init__(self)
        self.num = num

    def run(self):
        x = np.linspace(0, 2 * np.pi, 1000)
        for i in range(10):
            y = np.cos((i + 1 ) * x) * np.exp(-self.num * x)
            self.update_plot.emit(x, y)
            #time.sleep(0.2)


app = QApplication(sys.argv)
w = Matplotlib()
w.show()
sys.exit(app.exec_())