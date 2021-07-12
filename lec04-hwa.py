
#--------------------------------------------------------------------
# relaxation method -- Jacobi's method
#--------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# constants
L   = 1.0   # 2-D computational domain size
N   = 10    # number of equally spaced sampling points

# derived constants
Delta   = L/(N-1)                    # spatial resolution
dt      = Delta**2/4.0

# define a reference analytical solution
D = np.zeros((N,N))
for col in range(N):
	R[N-1][t] = 1

# initial condition

pri

# plotting parameters
end_time        = 100         # simulation time
nstep_per_image = 1          # plotting frequency

# create figure


'''
def update( frame ):
   global t, u

   for step in range( nstep_per_image ):
         
#     Jacobi's method
      for i in range( 1, N-1 ):
         for j in range( 1, N-1 ):
            F[i][j] = 0.25*( phi[i+1][j] + phi[i-1][j] + phi[i][j+1] + phi[i][j-1] \
                               - Delta**2 * rho(i,j))

#     update time
      t = t + dt
      if ( t >= end_time ):   break

#  calculate the reference analytical solution and estimate errors
   u_ref = ref_func( x, t )
   err   = np.abs( u_ref - u ).sum()/N**2

#  plot
   line_num.set_ydata( u )
   line_ref.set_ydata( u_ref )
   ax.legend( loc='upper right', fontsize=12 )
   ax.set_title( 't/T = %7.4f, error = %10.3e' % (t/t_scale, err) )

   return line_num, line_ref


# create movie
nframe = int( np.ceil( end_time/(nstep_per_image*dt) ) )
anim   = animation.FuncAnimation( fig, func=update, init_func=init,
                                  frames=nframe, interval=10, repeat=False )
'''
plt.show()
