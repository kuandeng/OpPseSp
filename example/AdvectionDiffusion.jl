# Pseudospectra of Advection Diffusion operator
include("../DiffOp.jl")

T = ComplexF64

# L^2([0, 1])
dom = 0.0..1.0

# Lengendre basis
sp = Ultraspherical(0.5, dom)

# differential expression
D = one(T)*Derivative(sp)
η = 0.015
expr = η*D^2+D
expr_ad = η*D^2-D

# boundary conditon
B = Dirichlet(sp, 0)
B_ad = Dirichlet(sp, 0)

# boundary values
valB = [[0.0, 0.0]]
valB_ad = [[0.0, 0.0]]

# differential operator
L = DiffOp(expr, B, valB)
L_ad = DiffOp(expr_ad, B_ad, valB_ad)

# grid points
nptx = 200
npty = 200
ax = [-60, 20]
ay = [-40, 40]
ptx = Vector(range(ax[1], ax[2], nptx))
pty = Vector(range(ay[1], ay[2], npty))

# initial input function u1
u1 = Fun(one(T), sp)
u1 = u1/Norm(u1)

# compute the pseudospectra
ops = Options(40, 1, 1e-12) 
pse = PseSp(L, L_ad, u1, ptx, pty, ops) 

# plot the pseudospectra
level = -12:1:0
contour(ptx, pty, log10.(pse), levels = level)