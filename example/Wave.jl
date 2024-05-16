# Pseudospectra of Wave operator
include("../BlockDiffOp.jl")

T = ComplexF64

# L^2([0, pi])*L^2([0, pi])
dom = 0.0..pi

# Lengendre basis
sp = Ultraspherical(0.5, dom)

# differential expression
delta = 0.5
D = one(T)*Derivative(sp)
expr = [0.0im*D D; D 0.0im*D]
expr_ad = [0.0im*D -D; -D 0.0im*D]

# boundary conditon
B_left = Evaluation(sp, dom.left)
B_right = Evaluation(sp, dom.right)
B = [0 B_left;
    B_right delta*B_right]
B_conj = [0 B_left;
    B_right -delta*B_right]

# boundary values
valB = [0.0, 0.0]

# differential operator
L = BlockDiffOp(expr, B, valB)
L_ad = BlockDiffOp(expr_ad, B_conj, valB)

# grid points
nptx = 400
npty = 400
ax = [-5, 3]
ay = [-4, 4]
ptx = Vector(range(ax[1], ax[2], nptx))
pty = Vector(range(ay[1], ay[2], npty))

# initial input function u1
u1 = [Fun(sp, [1.0+0.0im]);
Fun(sp, [1.0+0.0im])]
u1 = u1/Norm(u1)

# compute the pseudospectra
ops = Options(40, 1, 1e-14) 
pse = PseSp(L, L_ad, u1, ptx, pty, ops) 

# plot the pseudospectra
level = -1.0:0.2:0
contour(ptx, pty, log10.(pse), levels = level)
