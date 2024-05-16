# Pseudospectra of Huygens Fresnel operator
include("../FredOp.jl")

T = ComplexF64

# L^2([-1, 1])
dom_u = -1.0..1.0

# Lengendre basis
sp = Ultraspherical(0.5, dom_u)

# kernel coefficients under Lengendre basis
dom_ker = -2..2
kernel =  x->sqrt(16.0im)*exp(-16.0im*pi*x^2)
kernelCoeffs = T.(Fun(kernel, Ultraspherical(0.5, dom_ker)).coefficients)  

L = FredOp(kernelCoeffs, dom_u)

# kernel coefficients under Lengendre basis
dom_ker_ad = -2..2
kernel_ad =  x->sqrt(16.0im)'*exp(16.0im*pi*x^2)
kernelCoeffs = T.(Fun(kernel_ad, Ultraspherical(0.5, dom_ker_ad)).coefficients)  

L_ad = FredOp(kernelCoeffs_ad, dom_u)

# grid points
nptx = 400
npty = 400
ax = [-1.2, 1.2]
ay = [-1.2, 1.2]
ptx = Vector(range(ax[1], ax[2], nptx))
pty = Vector(range(ay[1], ay[2], npty))

# compute the pseudospectra 
ops = Options(30, 1, 1e-14) 
pse = PseSp(L, L_ad, u1, ptx, pty, ops) 

# plot the pseudospectra
level = -3:0.5:-1
contour(ptx, pty, log10.(pse), levels = level)