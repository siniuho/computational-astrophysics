# -*- coding: utf-8 -*-
"""
Spyder Editor

crawl test 001
"""

#--------------------------------------------------------------------
# Solve the advection equation with the FTCS/Lax/upwind schemes
# Periodic Boundary Condition
#--------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# constants
L   = 1.0   # 1-D computational domain size
N   = 100   # number of computing cells
v   = 1.0   # advection velocity
u0  = 1.0   # background density
amp = 0.5   # sinusoidal amplitude
cfl = 0.8   # Courant condition factor
colors = [ 'r', 'b', 'g', 'y', 'c']
scheme_name = ['FTCS', 'Lax', 'upwind', 'downwind']
scheme_ani = 1

# derived constants
dx     = L/N      # spatial resolution
dt     = cfl*dx/v # time interval for data update
period = L/v      # time period

# define a reference analytical solution
def ref_func( x, t ):
   k = 2.0*np.pi/L   # wavenumber
   return u0 + amp*np.sin( k*(x-v*t) )

# initial condition
t = 0.0
x = np.arange( 0.0, L, dx ) + 0.5*dx   # cell-centered coordinates
u = np.array([ref_func( x, t ),ref_func( x, t ) ,ref_func( x, t ) ,ref_func( x, t ) ])

# plotting parameters
end_time        = 2.0*period  # simulation time
nstep_per_image = 1           # plotting frequency

# create figure for animation
fig       = plt.figure( figsize=(6,6), dpi=140 )
ax        = plt.axes( xlim=(0.0, L), ylim=(u0 - amp*1.5, u0 + amp*1.5) )
line_ref, = ax.plot( [], [], 'k', ls='--', label='Reference' )
line_num, = ax.plot( [], [], colors[scheme_ani], ls='-', 
					 label = 'Scheme %i: ' % (scheme_ani+1) + scheme_name[scheme_ani] )
ax.set_xlabel( '$x$' )
ax.set_ylabel( '$u$' )
ax.tick_params( top=False, right=True, labeltop=False, labelright=True )

def init():
   line_num.set_xdata( x )
   line_ref.set_xdata( x )
   return line_num, line_ref

def update( frame ):
   global t, u

   for step in range( nstep_per_image ):
#     back up the input data
      u_in = u.copy()

#     update all cells
      for i in range( N ):
#        assuming periodic boundary condition
         ip = (i+1  ) % N
         im = (i-1+N) % N

#       (1) FTCS scheme (unconditionally unstable)
         u[0][i] = u_in[0][i] - dt*v*( u_in[0][ip] - u_in[0][im] )/(2.0*dx)

#       (2) Lax scheme (conditionally stable)
         u[1][i] = 0.5*( u_in[1][im] + u_in[1][ip] ) - dt*v*( u_in[1][ip] - u_in[1][im] )/(2.0*dx)

#       (3) upwind scheme (assuming v>0; conditionally stable)
         u[2][i] = u_in[2][i] - dt*v*( u_in[2][i] - u_in[2][im] )/dx

#       (4) downwind scheme (assuming v>0; unconditionally unstable)
         u[3][i] = u_in[3][i] - dt*v*( u_in[3][ip] - u_in[3][i] )/dx

#     update time
      t = t + dt
      if ( t >= end_time ):   break

#  calculate the reference analytical solution and estimate errors
   u_ref = ref_func( x, t )
   err   = np.abs( u_ref - u[scheme_ani] ).sum()/N

#  plot
   line_num.set_ydata( u[scheme_ani] )
   line_ref.set_ydata( u_ref )
   ax.legend( loc='upper right', fontsize=12 )
   ax.set_title( 't/T = %6.3f, error = %10.3e' % (t/period, err) )

   return line_num, line_ref


# create movie
nframe = int( np.ceil( end_time/(nstep_per_image*dt) ) )
anim   = animation.FuncAnimation( fig, func=update, init_func=init,
                                  frames=nframe, interval=10, repeat=False )
plt.show()