#--------------------------------------------------------------------
# Solve the advection equation with the FTCS/Lax/upwind/downwind/Lax-Wendroff schemes
# Last modified on Tue Mar 9, 2021 13:14
# Originated with prof. Hsi-Yu Schive
# Revised by Sìn-iū Hô b05202054@ntu.edu.tw
#--------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.metrics import mean_squared_error

# constants
L   = 1.0   # 1-D computational domain size
N   = 100   # number of computing cells
v   = +1.5  # advection velocity (can be positive or negative)
u0  = 1.0   # background density
amp = 0.5   # sinusoidal amplitude
cfl = 1.0   # Courant condition factor

# Available schemes, error types and functions
scheme_color = [ 'r', 'orange', 'g', 'brown', 'b']
scheme_name = ['FTCS', 'Lax', 'upwind', 'downwind', 'Lax-Wendroff', 'Matsuno']
scheme_num = np.size(scheme_name)
err_types = {1:'Absolute deviation', 2:'Deviation', 3:'Mean squared error'}
func_types = {1: 'sinusoidal', 2: 'Gaussian'}

animat_play = True             # DECIDE whether to play the animation
err_time_plot = True            # DECIDE whether to plot the error-vs-time plot
err_sel = 1					    # SELECT one from the error types {1,2,3}
func_sel = 1                    # SELECT one from the function types {1,2}
scheme_ani = 1                  # SELECT one scheme to display in the animation {0,1,2,3,4}
scheme_err_plot = [0,1,2,3,4]	# SELECT the scheme(s) that displays in the error-vs-time plot {0,1,2,3,4}

# derived constants
dx     = L/N                    # spatial resolution
dt     = cfl*dx/np.abs(v)       # time interval for data update
alpha  = v*dt/dx                # Courant number
period = L/np.abs(v)            # time period (for sinusiudal function)

# plotting parameters
end_time        = 10.0 * period  # simulation time
nstep_per_image = 1             # plotting frequency

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Define reference analytical solutions
def ref_func( x, t, _type_ ):
    global k, sigma
    if _type_ == 1:             # sinusoidal
        k = 2.0*np.pi/L         # wavenumber
        return u0 + amp * np.sin( k * (x - v*t) )
    if _type_ == 2:             # Gaussian (unnormalized)
        sigma = .15             # standard deviation
        return amp * np.exp( -( (x - v*t) - 0.5*L )**2 / sigma**2 )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def boundary_cond( index, num_cell, _type_ ):
    if _type_ == 1:             # sinusoidal --> periodic boundary condition
        return (index + 1) % num_cell, (index - 1 + num_cell) % num_cell
    if _type_ == 2:             # Gaussian --> free end boundary condition
        if (index == 0):
            return index + 1, index
        elif (index == num_cell - 1):
            return index, index - 1
        else:
            return index + 1, index - 1

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# initial conditions
t = 0.0
x = np.arange( 0.0, L, dx ) + 0.5*dx   # cell-centered coordinates

# empty lists/arrays for data recording
u = np.empty((scheme_num,N))
err_data, t_data = [], []  
for S in range(scheme_num):
    err_data.append([])
    u[S] = ref_func( x, t, func_sel )

# create figure for animation
if animat_play == True:
    fig       = plt.figure( figsize=(6,4), dpi=140 )
    if func_sel == 1:
        ax = plt.axes( xlim=(0.0, L), ylim=(u0 - amp*1.5, u0 + amp*1.5 ) )
    if func_sel == 2:
        ax = plt.axes( xlim=(0.0, L), ylim=(0.0, amp*1.5) )   
    line_ref, = ax.plot( [], [], 'k', ls='--', label='Analytical (%s function)' % func_types[func_sel] )
    line_num, = ax.plot( [], [], scheme_color[scheme_ani], ls='-', 
    					 label = 'Numerical (%s scheme)' % scheme_name[scheme_ani] )
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
    global t, u
    
    for step in range( nstep_per_image ):
#	back up the input data
        u_in = u.copy()
        
#	update all cells
        for i in range( N ):
            ip, im = boundary_cond( i, N, func_sel )

#       (1) FTCS scheme (unconditionally unstable)
            u[0][i] = u_in[0][i] - 0.5*alpha*( u_in[0][ip] - u_in[0][im] )

#       (2) Lax scheme (conditionally stable)
            u[1][i] = 0.5*( u_in[1][im] + u_in[1][ip] ) - 0.5*alpha*( u_in[1][ip] - u_in[1][im] )

#       (3) upwind scheme (assuming v>0; conditionally stable)
            u[2][i] = u_in[2][i] - alpha*( u_in[2][i] - u_in[2][im] )

#       (4) downwind scheme (assuming v>0; unconditionally unstable)
            u[3][i] = u_in[3][i] - alpha*( u_in[3][ip] - u_in[3][i] )
		
#		(5) the Richtmyer two-step Lax-Wendroff scheme
        u_half = np.empty( N )
        if v > 0:
            for i in range( N ):
                ip = boundary_cond( i, N, func_sel )[0]
                u_half[i] = 0.5*( u_in[4][ip] + u_in[4][i] ) - 0.5*alpha*( u_in[4][ip] - u_in[4][i] )
            for i in range( N ):
                im = boundary_cond( i, N, func_sel )[1]
                u[4][i] = u_in[4][i] - dt*v*( u_half[i] - u_half[im] )/dx
        if v < 0:
            for i in range( N ):
                im = boundary_cond( i, N, func_sel )[1]
                u_half[i] = 0.5*( u_in[4][i] + u_in[4][im] ) - 0.5*alpha*( u_in[4][i] - u_in[4][im])
            for i in range( N ):
                ip = boundary_cond( i, N, func_sel )[0]
                u[4][i] = u_in[4][i] - dt*v*( u_half[ip] - u_half[i] )/dx
        
        
#		(6) the two-step Matsuno scheme
        u_half = np.empty( N )
        if v > 0:
            for i in range( N ):
                ip = boundary_cond( i, N, func_sel )[0]
    
# Record & update time
        t_data.append(t)  
        print('t/T = %6.3f' % (t/period))
        t += dt

# Calculate the reference analytical solution  
    u_ref = ref_func( x, t, func_sel )
# Evaluate and record the errors
    err_ani = err_eval( u[scheme_ani], u_ref, err_sel )
    for S in scheme_err_plot:
        err = err_eval( u[S], u_ref, err_sel )
        err_data[S].append( err )

#  Plot for the animation
    if animat_play == True:
        line_num.set_ydata( u[scheme_ani] )
        line_ref.set_ydata( u_ref )
        ax.legend( loc='upper right', fontsize=12 )
        ax.set_title( 't/T = %6.3f, error = %10.3e' % (t/period, err_ani) )

        return line_num, line_ref

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# create the animation
if animat_play == True:
    nframe = int( np.ceil( end_time/(nstep_per_image*dt) ) )
    anim   = animation.FuncAnimation( fig, func=update, init_func=init,
                                  frames=nframe, interval=10, repeat=False )
    plt.show()

# create the error-vs-time plot
if err_time_plot == True:
    while t <= end_time:
        nframe = int( np.ceil( end_time/(nstep_per_image*dt) ) )
        update(nframe)
    
    fig1 = plt.figure( )
    
    ax1 = fig1.add_subplot( xlabel = 'Time $t/T$', ylabel = err_types[err_sel])
    ax1.set_xlim(0.0,end_time)
    ax1.set_ylim(1.0e-20,1.0e10)
    ax1.set_title('Error vs. time (%s)' % func_types[func_sel])
    ax1.grid()
    for S in scheme_err_plot:
        ax1.plot( t_data, err_data[S], 
    			 label = 'Scheme %i: ' % (S+1) + scheme_name[S], color = scheme_color[S] )
    ax1.set_yscale('log')
    ax1.legend(loc='best')
    plt.show()
