


# FYS1120 - Oblig 1 (Benjamin Borge)
"""TASK 1"""
# importing libraries
import numpy as np
import matplotlib.pyplot as plt


# finding electric potenital
def findVfieldinapoint(r,Q,R):
    V_point = 0
    for i in range(len(R)):
        R_i = r - R[i]
        Q_i = Q[i]
        V_point += Q_i / (np.linalg.norm(R_i))

    return V_point

# model of the two rings
a = 1 # distance from origo in z-direction
dl = 100 # number of circle partitions
dtheta = 2*np.pi/dl
R_top = []
Q_top = []
R_bot = []
Q_bot = []
q_top = -100 # total charge in top ring
q_bot = 100 # total charge in bottom ring
top_ring_radius = 2
bot_ring_radius = 3

for i in range(dl):
    theta_i = i*dtheta

    # top
    rx = top_ring_radius*np.cos(theta_i)
    ry = top_ring_radius*np.sin(theta_i)
    ri = np.array([rx, ry, a])
    R_top.append(ri)
    Q_top.append(q_top/dl)

    # bottom
    rx = bot_ring_radius*np.cos(theta_i)
    ry = bot_ring_radius*np.sin(theta_i)
    ri = np.array([rx, ry, -a])
    R_bot.append(ri)
    Q_bot.append(q_bot/dl)


# mesh
dim = 6
density = 64
x = np.linspace(-dim, dim, density)
z = x # square grid
rx,rz = np.meshgrid(x,z)


# field calcuation
V = np.zeros((density, density), float)
for i in range(len(rx.flat)):
    r = np.array([rx.flat[i],0,rz.flat[i]])

    V.flat[i] = findVfieldinapoint(r,Q_top,R_top) + findVfieldinapoint(r,Q_bot,R_bot)

Ez,Ex = np.gradient(-V) # finding the electric field

# plot
fig, axs = plt.subplots(1, 2)

axs[0].contourf(rx,rz,V,levels = 40)
axs[0].set_title("Electric Potential")
axs[0].set_xlabel("x")
axs[0].set_ylabel("z")
axs[0].set_aspect('equal', adjustable='box')

ax0 = axs[0]

axs[1].streamplot(rx, rz, Ex, Ez,color="k")
axs[1].set_title("Electric Field")
axs[1].set_xlabel("x")
axs[1].set_aspect('equal', adjustable='box')
x = [-top_ring_radius, top_ring_radius, -bot_ring_radius, bot_ring_radius]
y = [a, a, -a, -a]
c = ["b", "b","r", "r"]
axs[1].scatter(x,y,s=128,c=c)


fig.show()
"""TASK 2"""

dl = 1000
z_array = np.linspace(-10,10, dl)
z_vec = []
for i in range(dl):
    z_vec.append(np.array([0, 0, 1])*z_array[i])

z_efield = np.zeros(dl)
for i in range(dl):
    z_efield[i] = findVfieldinapoint(z_vec[i],Q_top,R_top) + findVfieldinapoint(z_vec[i],Q_bot,R_bot)

z_line = np.linspace(-dim,dim, 100)

# plot
fig2, axs = plt.subplots(1, 2)

axs[0].contourf(rx,rz,V,levels = 40)
axs[0].set_title("Electric Potential")
axs[0].set_xlabel("x")
axs[0].set_ylabel("z")
axs[0].set_aspect('equal', adjustable='box')
axs[0].plot(np.zeros(100), z_line, "r-")

axs[1].plot(z_array, z_efield, "r-")
axs[1].set_aspect('equal', adjustable='box')
axs[1].set_xlabel("z")
axs[1].set_ylabel("Electric potential")


fig2.show()
plt.show()
