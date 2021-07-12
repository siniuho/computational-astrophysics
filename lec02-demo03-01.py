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
D   = 1.0   # diffusion coefficient
u0  = 1.0   # background density
amp = 0.5   # sinusoidal amplitude
cfl = 0.8   # Courant condition factor

# Available schemes, error types and functions
scheme_color = ['r', 'g']
scheme_name = ['FTCS', 'BTCS', 'Crank Nicolson']
scheme_num = np.size(scheme_name)
err_types = {1:'Absolute deviation', 2:'Deviation', 3:'Mean squared error'}
func_types = {1: 'sinusoidal', 2: 'Gaussian'}

animat_play = True             # DECIDE whether to play the animation
err_time_plot = True            # DECIDE whether to plot the error-vs-time plot
err_sel = 1					    # SELECT one from the error types {1,2,3}
func_sel = 1                    # SELECT one from the function types {1,2}
scheme_ani = 1                  # SELECT one scheme to display in the animation {0,1,2,3,4}
scheme_err_plot = [0,1]           # SELECT the scheme(s) that displays in the error-vs-time plot {0,1,2,3,4}

# derived constants
dx      = L/(N-1)                # spatial resolution
dt      = cfl*0.5*dx**2.0/D      # time interval for data update
t_scale = (0.5*L/np.pi)**2.0/D   # diffusion time scale across L
alpha   = dt * D / dx**2

# set the coefficient matrices A with A*u(t+dt)=u(t)
A = np.diagflat( np.ones(N-3)*(-alpha),       -1 ) + \
    np.diagflat( np.ones(N-2)*(1.0+2.0*alpha), 0 ) + \
    np.diagflat( np.ones(N-3)*(-alpha),       +1 );

# plotting parameters
end_time        = 1.0 * t_scale # simulation time
nstep_per_image = 1             # plotting frequency

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Define reference analytical solutions
def ref_func( x, t, _type_ ):
    global k, sigma
    if _type_ == 1:             # sinusoidal
        k = 2.0*np.pi/L         # wavenumber
        return u0 + amp*np.sin( k*x )*np.exp( -k**2.0*D*t )
    if _type_ == 2:             # Gaussian (unnormalized)
        sigma = .15             # standard deviation
        pass

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
x = np.linspace( 0.0, L, N )   # cell-centered coordinates

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
        u_bk = np.copy( u[1][1:-1] )
        
#	update all cells
        for i in range( 1, N-1 ):
#           ip, im = boundary_cond( i, N, func_sel )

#       (1) FTCS scheme (unconditionally unstable)
            u[0][i] = u_in[0][i] + alpha*( u_in[0][i+1] - 2*u_in[0][i] + u_in[0][i-1] )

#       (2) BTCS scheme (unconditionally stable)
        u_bk[ 0] += alpha * u0
        u_bk[-1] += alpha * u0
        u[1][1:-1] = np.linalg.solve( A, u_bk )

# Record & update time
        t_data.append(t)  
        print('t = %6.5f' % (t))
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
        ax.set_title( 't/T = %7.3f, error = %10.3e' % (t/t_scale, err_ani) )

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
    
    ax1 = fig1.add_subplot( xlabel = 'Time $t$', ylabel = err_types[err_sel])
    ax1.set_xlim(0.0, end_time)
    ax1.set_title('Error vs. time (%s)' % func_types[func_sel])
    ax1.grid()
    for S in scheme_err_plot:
        ax1.plot( t_data, err_data[S], ls = '--' if (S == 5 or S == 6) else None,
    			 label = 'Scheme %i: ' % (S+1) + scheme_name[S], color = scheme_color[S] )
    ax1.set_yscale('log')
    ax1.legend(loc='best')
    plt.show()
