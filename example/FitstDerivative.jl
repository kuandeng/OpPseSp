# Pseudospectra of Fitst derivative operator
include("../DiffOp.jl")

T = ComplexF64

# L^2([0, 2])
dom = 0.0..2.0

# Lengendre basis
sp = Ultraspherical(0.5, dom)

# differential expression
D = one(T)*Derivative(sp)
expr = D
expr_ad = -D

# boundary conditon
B = Evaluation(sp, dom.right)
B_ad =  Evaluation(sp, dom.left)

# boundary values
valB = 0.0
valB_ad = 0.0

# differential operator
L = DiffOp(expr, B, valB)
L_ad = DiffOp(expr_ad, B_ad, valB_ad)

# grid points
nptx = 50
npty = 50
ax = [-12, 0]
ay = [-4, 4]
ptx = Vector(range(ax[1], ax[2], nptx))
pty = Vector(range(ay[1], ay[2], npty))

# initial input function u1
u1 =  Fun(one(T), sp)
u1 = u1/Norm(u1)

# compute the pseudospectra
ops = Options(40, 1, 1e-14) 
pse = PseSp(L, L_ad, u1, ptx, pty, ops) 

# plot the pseudospectra
level = -8:1:-1
contour(ptx, pty, log10.(pse), levels = level)