import numpy as np

class blochEquation():
    def __init__(self, M_0, T1, T2):
        self.mZ = None
        self.mXY = None
        self.m_0 = M_0
        self.t1 = T1
        self.t2 = T2

    def _zMagnetization(self, t):
        self.mZ = self.m_0 * (1 - np.exp(-t/self.t1))

    def _xyMagnetization(self, t):
        self.mXY = self.m_0 * (np.exp(-t/self.t2))


class magentization():
    def __init__(self, b0, b1, gyro):
        self.w0 = None
        self.w1 = None
        self.vector = np.array([0, 0, 1])
        self.b0 = b0
        self.b1 = b1
        self.gyro = gyro
        self.calcW()

    def calcW(self):
        self.w0 = self.gyro * self.b0
        self.w1 = self.gyro *self.b1

    def _b0Effect(self, t):
        self.vector *=  np.cos(self.w0*t)

    def _rotate(self, t):
        self.vector= [self.b1*np.cos(self.w1*t), -self.b1* np.sin(self.w1*t), self.b0]
    