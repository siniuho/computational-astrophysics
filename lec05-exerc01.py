
#--------------------------------------------------------------------
# Convolution with DFT
#--------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt


# constants
L       = 1.0           # 1-D domain size
N       = 20            # number of equally spaced sampling points
k_min   = 2.0*np.pi/L   # minimum wavenumber
k1      = k_min*1       # wavenumber of component 1
k2      = k_min*5       # wavenumber of component 1
amp1    = 7.0e-1        # amplitude of component 1
amp2    = 2.0e-1        # amplitude of component 2
dc      = 5.0e-1        # DC (i.e., background)
dx      = L/N           # spatial resolution

# set the input data
x  = np.arange( 0.0, L, dx )   # x=L/N*n, n=0,1,...,N-1
u1 = amp1*np.cos( k1*x )
u2 = amp2*np.sin( k2*x )
dc = np.ones(x.size)*dc
u  = u1 + u2 + dc


# compute the coefficients of all sin() and cos() using np.fft.rfft
# ref: https://docs.scipy.org/doc/numpy-1.14.0/reference/routines.fft.html#module-numpy.fft
# =============================================================


k  = 2.0 * np.pi * np.fft.rfftfreq( N, dx )

uk = np.fft.rfft( u - u.mean() )
spectrum = 2*np.abs( uk/N )

#k = np.linspace(0, sampling_rate/2, len(spectrum))

print(k1,k2)
print(amp1,amp2)

print(2*np.abs( uk/N ))

# =============================================================


# create figure
fig       = plt.figure()
ax        = plt.axes()
ax.set_xlabel( 'x' )
ax.set_ylabel( 'u' )

ax.plot( x, u1, 'r', ls='-', label='Component 1' )
ax.plot( x, u2, 'b', ls='-', label='Component 2' )
ax.plot( x, dc, 'm', ls='-', label='DC' )
ax.plot( x, u,  'k', ls='-', label='Sum' )

ax.legend( loc='upper center', fontsize=12 )

fig2       = plt.figure()
ax2        = plt.axes()
ax2.set_xlabel( '$k$' )
ax2.set_ylabel( '$|F[u]|$' )
ax2.plot(k, spectrum)

plt.show()
