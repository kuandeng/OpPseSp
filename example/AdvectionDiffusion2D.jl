# Pseudospectra of 2D Advection Diffusion operator
include("../PartialDiffOp.jl")

T = ComplexF64

# L^2([-1,1]^2), Lengendre basis 
sp = Ultraspherical(0.5, -1..1)^2

# differential expression
eta = T(0.05)
sp = Ultraspherical(0.5, -1..1)^2
expr = -eta*Laplacian(sp) + Derivative(sp,[0, 1])
expr_ad = -eta*Laplacian(sp) - Derivative(sp,[0, 1])

# boundary conditon
B = Dirichlet(sp);

# boundary values
valB = zeros(∂(sp));

# differential operator
L = PartialDiffOp(operator, B, valB, 1e-8)
L_ad = PartialDiffOp(operator_ad, B, valB, 1e-8)

# grid points
nptx = 100
npty = 100
ax = [0, 20]
ay = [-15, 15]
ptx = Vector(range(ax[1], ax[2], nptx))
pty = Vector(range(ay[1], ay[2], npty))

# initial input function u1
u1 = Fun(one(T), sp)
u1 = u1/Norm(u1)

# compute the pseudospectra
ops = Options(40, 1, 1e-8) 
pse = PseSp(L, L_ad, u1, ptx, pty, ops)

# plot the pseudospectra
level = -6:1:-1
contour(ptx, pty, log10.(pse), levels = level)