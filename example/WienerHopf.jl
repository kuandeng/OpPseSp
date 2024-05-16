# Pseudospectra of Advection Diffusion operator
include("../VoltOp.jl")

T = ComplexF64

# L^2([0, 10])
dom_u = 0.0..10.0

# Lengendre basis
sp = Ultraspherical(0.5, dom_u)

# kernel coefficients under Lengendre basis   
dom_ker = -10..0 
kernel = x->exp(x) 
kernelCoeffs = T.(Fun(kernel, Ultraspherical(0.5, dom_ker)).coefficients)

# preallocated matrix dimension N
N = 5000
# right Volterra Operator
side = 'r'

L = VoltOp(kernelCoeffs, N, dom_u, side)

# kernel coefficients under Lengendre basis   
dom_ker_ad = 0..10 
kernel_ad = x->exp(-x) 
kernelCoeffs_ad = T.(Fun(kernel_ad, Ultraspherical(0.5, dom_ker_ad)).coefficients)

L_ad = VoltOp(kernelCoeffs_ad, N, dom_u, 'l')

# grid points
nptx = 400
npty = 400
ax = [-0.1, 0.8]
ay = [-0.5, 0.5]
ptx = Vector(range(ax[1], ax[2], nptx))
pty = Vector(range(ay[1], ay[2], npty))

# initial input function u1
u1 = zeros(T, N)
u1[1] = T(1)

# compute the pseudospectra 
ops = Options(20, 1, 1e-14) 
pse = PseSp(L, L_ad, u1, ptx, pty, ops) 

# plot the pseudospectra
level = -13:1:-1
contour(ptx, pty, log10.(pse), levels = level)