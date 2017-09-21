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
                         'delimiter':         ' ', \
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
        elif self.options['data type'] == 'moment':
            self.VPoly = Poly.polyfit(data[:,0], data[:,1], self.deg)
        self.VPoly[0] = 0
        self.VPoly[1] = 0


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

    def frequency(self, energy):
        x1, x2 = self.amplitude(energy)
        def f(x):
            return 1. / np.sqrt( energy-Poly.polyval(x, self.VPoly) )
        integral = quad(f, x1, x2)
        return 1. / ( np.sqrt(2*self.inertia) * integral[0] )

N = 100
fname = 'temp.txt'
a = np.empty((100, 2))
a[:,0] = np.linspace(-10, 10, 100)
a[:,1] = np.linspace(-10, 10, 100)
np.savetxt(fname, a)


I = 1. / 4. / np.pi**2
bs = BalanceSpring(I, fname)
print(bs.frequency(50))
