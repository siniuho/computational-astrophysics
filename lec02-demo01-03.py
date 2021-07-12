#--------------------------------------------------------------------
# Solve the advection equation with the FTCS/Lax/upwind schemes
# revised by HÔ, SÌN-IŪ
#--------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.metrics import mean_squared_error

# constants
L   = 1.0   # 1-D computational domain size
N   = 100   # number of computing cells
v   = 1.0   # advection velocity
u0  = 1.0   # background density
amp = 0.5   # sinusoidal amplitude
cfl = 0.8   # Courant condition factor
colors = [ 'r', 'b', 'g', 'y', 'c']
scheme_name = ['FTCS', 'Lax', 'upwind', 'downwind', 'Lax-Wendroff']
scheme_err_plot = [0,1,2,3,4]				# select the schemes
scheme_num = 5
scheme_ani = 0
err_type = {1:'Absolute deviation', 2:'Deviation', 3:'Mean squared error'}
err_sel = 1					    # select the preferred error type

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

# empty list for data recording
u = np.empty((scheme_num,N))
err_data, t_data = [], []  

for S in range(scheme_num):
    err_data.append([])
    u[S] = ref_func(x, t)

# plotting parameters
end_time        = 2.0 * period  # simulation time
nstep_per_image = 1             # plotting frequency

# create figure for animation
fig       = plt.figure( figsize=(6,6), dpi=140 )
ax        = plt.axes( xlim=(0.0, L), ylim=(u0 - amp*1.5, u0 + amp*1.5) )
line_ref, = ax.plot( [], [], 'k', ls='--', label='Reference' )
line_num, = ax.plot( [], [], colors[scheme_ani], ls='-', 
					 label = 'Scheme %i: ' % (scheme_ani+1) + scheme_name[scheme_ani] )
ax.set_xlabel( '$x$' )
ax.set_ylabel( '$u$' )
ax.tick_params( top=False, right=True, labeltop=False, labelright=True )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def init():
   line_num.set_xdata( x )
   line_ref.set_xdata( x )
   return line_num, line_ref

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def err_eval( actual_data, ref_data, _type_ ):
    if  _type_ == 1:
        return np.abs( actual_data - ref_data ).sum()/N
    elif _type_ == 2:
        return ( actual_data - ref_data ).sum()/N
    elif _type_ == 3:
        return mean_squared_error( actual_data, ref_data, squared=False )
    else: 
        return None

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def update( frame ):
   global t, u, t_data, err_data
    
   for step in range( nstep_per_image ):
#	back up the input data
        u_in = u.copy()
#	calculate the half-time step solution for 
        u_half = np.empty( N )
#	update all cells
        for i in range( N ):
#	assuming periodic boundary conditions
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
		
#		(5) the Richtmyer two-step Lax-Wendroff scheme
        for i in range( N ):
            ip = (i+1  ) % N
            u_half[i] = 0.5*( u_in[4][ip] + u_in[4][i] ) - 0.5*dt*v*( u_in[4][ip] - u_in[4][i] )/dx
        for i in range( N ):
            im = (i-1+N) % N
            u[4][i] = u_in[4][i] - dt*v*( u_half[i] - u_half[im] )/dx

# record & update time
        t_data.append(t)  
        print('t/T = %6.3f' % (t/period))
        t += dt
        if ( t > end_time ):   break
        


# calculate the reference analytical solution and estimate errors
   u_ref = ref_func( x, t )
   err_ani = err_eval( u[scheme_ani], u_ref, err_sel )
   for S in scheme_err_plot:
        err = err_eval( u[S], u_ref, err_sel )
        err_data[S].append( err )

#  plot
   line_num.set_ydata( u[scheme_ani] )
   line_ref.set_ydata( u_ref )
   ax.legend( loc='upper right', fontsize=12 )
   ax.set_title( 't/T = %6.3f, error = %10.3e' % (t/period, err_ani) )

   return line_num, line_ref

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# create movie
nframe = int( np.ceil( end_time/(nstep_per_image*dt) ) )
anim   = animation.FuncAnimation( fig, func=update, init_func=init,
                                  frames=nframe, interval=10, repeat=False )
plt.show()

while t <= end_time:
    nframe = int( np.ceil( end_time/(nstep_per_image*dt) ) )
    update(nframe)


# error vs. t plot
fig1 = plt.figure( )
ax1 = fig1.add_subplot( xlabel = 'Time $t/T$', ylabel = err_type[err_sel] )
ax1.grid()
for S in scheme_err_plot:
    ax1.plot( t_data, err_data[S], 
			 label = 'Scheme %i: ' % (S+1) + scheme_name[S], color = colors[S] )
ax1.set_yscale('log')
ax1.legend(loc='best')
plt.show()
