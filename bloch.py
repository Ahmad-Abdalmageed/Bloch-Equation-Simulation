import numpy as np

class blochEquation():
    def __init__(self, M_0, T1, T2):
        self.mZ = None
        self.mXY = None
        self.m_0 = M_0
        self.t1 = T1
        self.t2 = T2

    def zMagnetization(self, t):
        self.mZ = self.m_0 * (1 - np.exp(-t/self.t1))

    def xyMagnetization(self, t):
        self.mXY = self.m_0 * (np.exp(-t/self.t2))


class magentization():
    def __init__(self, t1, t2, m0):
        self.array = []
        self.vector = np.array([0, 0, 1.0])
        self.bloch = blochEquation(m0, t1, t2)
        print(self.vector)


    def rotate(self, t):

        for sec in np.arange(0, t, 0.01):
            self._rotate(sec)
            self.array.append(self.vector)
        self.array = np.array(self.array)


    def _rotate(self, t):
        self.bloch.zMagnetization(t)
        self.bloch.xyMagnetization(t)

        self.vector= [self.bloch.mXY, self.bloch.mXY, self.bloch.mZ]


if __name__ == '__main__':
    m = magentization(812*10**-3, 42*10**-3, 3)
    m.rotate(2)
    print(m.array)
    print(m.array[1, 0])