
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
#u1 = amp1*np.cos( k1*x )
#u2 = amp2*np.sin( k2*x )
#dc = np.ones(x.size)*dc
u  = 1.6 + 1 * np.cos(2*k_min*x) + 0.5 * np.cos(3*k_min*x)  - 0.5 * np.sin(1*k_min*x) 


# compute the coefficients of all sin() and cos() using np.fft.rfft
# ref: https://docs.scipy.org/doc/numpy-1.14.0/reference/routines.fft.html#module-numpy.fft
# =============================================================


uk = np.fft.rfft( u / N )
uk[0] *= 0.5
print( '%10s'*3%('n','cos','sin') )
for t in range(0,N//2+1):
    print( '%10d' % t + '%10.3f'*2 % (2.0*uk[t].real, 2.0*uk[t].imag)    )
    
k = np.arange(0,N//2+1)

spectrum = np.abs( 2*uk )



#k = np.linspace(0, sampling_rate/2, len(spectrum))


# =============================================================


# create figure
fig       = plt.figure()
ax        = plt.axes()
ax.set_xlabel( '$x$' )
ax.set_ylabel( '$u$' )

#ax.plot( x, u1, 'r', ls='-', label='Component 1' )
#ax.plot( x, u2, 'b', ls='-', label='Component 2' )
#ax.plot( x, dc, 'm', ls='-', label='DC' )
ax.plot( x, u,  'k', ls='-', label='Sum' )

ax.legend( loc='best', fontsize=12 )

fig2       = plt.figure()
ax2        = plt.scatter(k, spectrum, color = 'b')
for j in k:
    ax2 = plt.plot([k[j], k[j]], [0,spectrum[j]], color = 'b')
plt.show()
