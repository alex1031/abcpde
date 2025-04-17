import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# I am using a finite difference method, so let's define a grid of values for x, y, z (3D grid)
nx, ny, nz = 50, 50, 50 
Lx, Ly, Lz = 1.0, 1.0, 1.0  
dx, dy, dz = Lx/nx, Ly/ny, Lz/nz  

# Time step and max time
dt = 0.001 
Tmax = 0.1 

## Physical parameters
# Diffusion coefficient
D = 0.01  
# Velocieties
u_x, nu_y, nu_z = 0.1, 0.1, 0.1 

# Stability condition (for numeric stability); this ensures the explicit diffusion update remains stable.
alpha = D * dt / dx**2
beta = D * dt / dy**2
gamma = D * dt / dz**2
assert alpha < 0.5 and beta < 0.5 and gamma < 0.5, "Time step too large!"

# Initial condition (the concentration starts from the central peak)
C = np.zeros((nx, ny, nz))
C[nx//2, ny//2, nz//2] = 1.0

# Time stepping loop
t = 0.0
while t < Tmax:
    C_new = np.copy(C)
    
    # Diffusion term (Finite Difference Laplacian)
    C_new[1:-1, 1:-1, 1:-1] += D * dt * (
        (C[2:, 1:-1, 1:-1] - 2*C[1:-1, 1:-1, 1:-1] + C[:-2, 1:-1, 1:-1]) / dx**2 +
        (C[1:-1, 2:, 1:-1] - 2*C[1:-1, 1:-1, 1:-1] + C[1:-1, :-2, 1:-1]) / dy**2 +
        (C[1:-1, 1:-1, 2:] - 2*C[1:-1, 1:-1, 1:-1] + C[1:-1, 1:-1, :-2]) / dz**2
    )
    
    # Advection term (Upwind scheme)
    C_new[1:-1, 1:-1, 1:-1] -= dt * (
        nu_x * (C[1:-1, 1:-1, 1:-1] - C[:-2, 1:-1, 1:-1]) / dx +
        nu_y * (C[1:-1, 1:-1, 1:-1] - C[1:-1, :-2, 1:-1]) / dy +
        nu_z * (C[1:-1, 1:-1, 1:-1] - C[1:-1, 1:-1, :-2]) / dz
    )
    
    C = np.copy(C_new)
    t += dt

#### I am not great in plotting in python....so I have asked chatgpt to provide a code for it
# Visualization (2D slice)
x, y = np.linspace(0, Lx, nx), np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, C[:, :, nz//2], cmap='viridis')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Concentration")
ax.set_title("Advection-Diffusion in 3D (Slice at z = Lz/2)")
plt.show()

