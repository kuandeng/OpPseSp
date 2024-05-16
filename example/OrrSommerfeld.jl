# Pseudospectra of Fitst derivative operator
include("../GepDiffOp.jl")

T = ComplexF64

# L^2([-1, 1])
dom = -1.0..1.0

# Lengendre basis
sp = Ultraspherical(0.5, dom)

# operator parameters
Re = 10000
a = 1.02

# differential expression
D = one(T)*Derivative(sp)
c0 = Fun(x->1.0im*a^3*(1-x^2)+a^4/Re-2.0im*a, sp)
c2 = Fun(x->1.0im*a*(x^2-1)-2*a^2/Re, sp)
c4 = Fun(1/Re, sp)
expr_L = c4*D^4 + c2*D^2 + c0

c0_ad = Fun(x->-1.0im*a^3*(1-x^2)+a^4/Re, sp)
c1_ad = Fun(x->-4.0im*a*x, sp)
c2_ad = Fun(x->1.0im*a*(1-x^2)-2a^2/Re, sp)
c4_ad = Fun(1/Re, sp)
expr_L_ad = c4_ad*D^4 + c2_ad*D^2 + c1_ad*D + c0_ad

expr_R = D^2 - a^2

# boundary conditon
B_L = [Dirichlet(sp, 0); Dirichlet(sp, 1)]
B_R = Dirichlet(sp, 0)

# boundary values
val_B_L = [[0, 0], [0, 0]]
val_B_R = [[0, 0]]

# differential operator
opL = DiffOp(expr_L, B_L, val_B_L)
opL_ad = DiffOp(expr_L_ad, B_L, val_B_L)
opR = DiffOp(expr_R, B_R, val_B_R)

L = GepDiffOp(opL, opR)
L_ad = GepDiffOp_ad(opL_ad, opR)

# grid points
nptx = 400
npty = 400
ax = [-1, 0.2]
ay = [-1.2, 0.2]
ptx = Vector(range(ax[1], ax[2], nptx))
pty = Vector(range(ay[1], ay[2], npty))

# initial input function u1
u1 = Fun(one(T), sp)
u1 = u1/Norm(u1)

# compute the pseudospectra
ops = Options(30, 1, 1e-14) 
pse = PseSp(L, L_ad, u1, ptx, pty, ops) 

# plot the pseudospectra
level = -8:1:-1
contour(ptx, pty, log10.(pse), levels = level)