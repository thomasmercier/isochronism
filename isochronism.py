import numpy as np
import numpy.polynomial.polynomial as Poly
from scipy.integrate import quad

class BalanceSpring:

    def __init__(self, inertia, datafile, options={}):

        '''
        AVAILABLE OPTIONS :
            'angle unit': 'rad', 'deg'
            'data type': 'moment', 'potential'
            'delimiter': any character
            'polynomial degree': any positive integer
        '''

        self.options = { 'angle unit':       'rad', \
                         'delimiter':         None, \
                         'polynomial degree': 9, \
                         'data type': 'moment' }
        for key in options:
            self.options[key] = options[key]

        self.inertia = inertia

        data = np.loadtxt(datafile, delimiter=self.options['delimiter'])
        if self.options['angle unit'] == 'deg':
            data[:,0] *= np.pi / 180.
        self.deg = self.options['polynomial degree']
        if self.options['data type'] == 'moment':
            temp = Poly.polyfit(data[:,0], data[:,1], self.deg-1)
            self.VPoly = Poly.polyint(temp)
        elif self.options['data type'] == 'potential':
            self.VPoly = Poly.polyfit(data[:,0], data[:,1], self.deg)
        self.VPoly[0] = 0
        self.VPoly[1] = 0

        self.frequency = np.vectorize(self.frequency1)


    def amplitude(self, energy):
        # we compute the 2 real roots closest to 0
        righthandside = np.zeros(self.deg+1)
        righthandside[0] = energy
        roots = Poly.polyroots(self.VPoly-righthandside)
        rplus = np.inf
        rminus = -np.inf
        for r in np.real(roots[np.isreal(roots)]):
            if r > 0 and r < rplus:
                rplus = r
            elif r < 0 and r > rminus:
                rminus = r
        return [rminus, rplus]

    def amplitude_positive(self, energy):
        temp, _ = self.amplitude(energy)
        return temp

    def frequency1(self, energy):
        x1, x2 = self.amplitude(energy)
        def f(x):
            return 1. / np.sqrt( energy-Poly.polyval(x, self.VPoly) )
        integral = quad(f, x1, x2)
        return 1. / ( np.sqrt(2*self.inertia) * integral[0] )

    def angle_with_unit(self, angle_in_rad):
        if self.options['angle unit'] == 'rad':
            return angle_in_rad
        elif self.options['angle unit'] == 'deg':
            return angle_in_rad * np.pi / 180.
