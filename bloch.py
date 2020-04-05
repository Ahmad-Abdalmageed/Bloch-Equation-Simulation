import numpy as np

class blochEquation():
    """
    Responsible for the calculation of Bloch Equations.
    Implement the Following:
    * Mxy calculation : Mxy = Mo * (e ^(-t/ T2)
    * Mz calculation :  Mz = Mo * (1 - e ^(-t/ T1)
    """
    def __init__(self, M_0: [int, float], T1: float, T2: float):
        """
        Instantiates the class`s main parameters
        ================== =================================================
        **Parameters**
        M_0                The value of the main magnetic field applied
        T1                 T1 value of a specific tissue
        T2                 T2 value of a specific tissue
        ================== =================================================
        """
        self.m_0 = M_0
        self.t1 = T1
        self.t2 = T2

    def zMagnetization(self, t):
        """
        Calculates the Mz Component of the magnetization
        ================== =========================================
        **Parameters**
        t                  Time in seconds
        ================== =========================================
        **Returns**
        Mz                 The Z component of the magnetization
        ================== =========================================
        """
        return self.m_0 * (1 - np.exp(-t/self.t1))

    def xyMagnetization(self, t):
        """
        Calculates the Mz Component of the magnetization
        ================== =========================================
        **Parameters**
        t                  Time in seconds
        ================== =========================================
        **Returns**
        Mxy                The XY component of the magnetization
        ================== =========================================
        """
        return self.m_0 * (np.exp(-t/self.t2))


class magentization():
    """
    Responsible for calculating the magnetization vector.
    Implements the following:
    * calculate the magnetization vector after application of Mo [0 0 Mo]
    * Returns the vector into its relaxation state
    """
    def __init__(self, t1, t2, m0):
        """
        Instantiates the class`s main parameters
        ================== =================================================
        **Parameters**
        m0                 The value of the main magnetic field applied
        t1                 T1 value of a specific tissue
        t2                 T2 value of a specific tissue
        ================== =================================================
        """
        self.bloch = blochEquation(m0, t1, t2)
        self.vector = [[self.bloch.xyMagnetization(0), self.bloch.xyMagnetization(0), self.bloch.zMagnetization(0)]]

    def rotate(self, t):
        """
        Rotates the magnetization vector by application of an RF pulse for a given time t
        ================== =================================================
        **Parameters**
        t                  Time in seconds
        ================== =================================================
        """
        for sec in np.arange(0, t, 0.01):
            self.vector = np.append(self.vector, [[self.bloch.xyMagnetization(sec), self.bloch.xyMagnetization(sec),
                                   self.bloch.zMagnetization(sec)]], axis=0)

if __name__ == '__main__':
    m = magentization(812*10**-3, 42*10**-3, 3)
    m.rotate(2)
    print(m.vector)
