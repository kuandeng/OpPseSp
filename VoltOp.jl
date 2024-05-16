include("./OpPseSp.jl")

# defintion of Volterra integral operator, with preallocated matrix dimension n, TODO resize the matrix representation 
struct VoltOp{T<:FloatOrComplex} <: Op{T}
    volt::BandedMatrix{T} # matrix representation of Volterra integral operator
    N::Int # preallocated matrix dimension n
    dom::Domain 
    side::Char
end


# definition of shift operator L - zI
struct ShiftVoltOp{T<:FloatOrComplex} <: Op{T}
    voltOp::VoltOp{T}
    shift::FloatOrComplex
    qrData::BandedQrData{T}
end


# default constructor
VoltOp(kernelCoeffs::Vector{T}, N::Int, dom::Domain, side::Char) where T = 
VoltOp{T}(voltConvMatrix(kernelCoeffs, N, dom, side), N, dom, side)

# redefine "-"
-(op::VoltOp{T}, z::FloatOrComplex) where T = ShiftVoltOp(op, z, BandedQrData(op.volt))

# redefine "\"
function \(op::ShiftVoltOp{T}, y::AbstractVector{T}) where T
    ny = 0
    normy2 = 0
    for i in eachindex(y)
        if (y[i] == 0 && normy2 != 0) 
            break
        else 
            ny += 1
            normy2 += y[i]^2
        end
    end
    qrData = op.qrData
    A = op.voltOp.volt.data
    N = op.voltOp.N
    nx, qrStep = baqsv!(A, N, qrData.bu, qrData.bl, (@view y[1:ny]), op.shift, qrData.qrStep, qrData.tau, qrData.refl, qrData.workA, qrData.worky, qrData.workH)
    op.qrData.qrStep = qrStep

    x = zeros(T, N)
    x[1:nx] .= qrData.worky[1:nx]
    return x
end

# innerproduct between functions
InnerProduct(x::AbstractVector{T}, y::AbstractVector{T}) where T = (n = min(length(x), length(y)); x[1:n]'*y[1:n])

# norm of function
Norm(y::AbstractVector{T}) where T = norm(y)

# construct the matrix representation of Volterra convolution integral operator
function subtraction(v::AbstractVector{T}, start::Int) where T <: FloatOrComplex
    n = length(v)
    u = v./(2start+1:2:2n+2start-1)
    b = zeros(T,n)
    for i = 1:n-2
        b[i] = u[i]-u[i+2]
    end
    b[end] = u[end]
    b[end-1] = u[end-1]
    return b
end


function voltConvMatrix(coeffs::AbstractVector{T}, n::Int,  dom::Domain, side::Char) where T <: FloatOrComplex
    bu = bl = length(coeffs)
    mat = BandedMatrix{T}(undef,(n+bl,n),(bu,bl))

    #Initial the 0th column
    mat[2:bu+1,1] = side == 'l' ? subtraction(coeffs, 0) : -subtraction(coeffs, 0)
    mat[1,1] = side == 'l' ? coeffs[1]-coeffs[2]/3 : coeffs[1]+coeffs[2]/3

    #Initial the 1th column
    mat[2:bu+2,2] =  subtraction((@view mat[1:bu+1,1]), 0) - 
        (side == 'l' ? [mat[2:bu+1,1]; 0] : -[mat[2:bu+1,1]; 0])

    #Recursion below the diagonal
    for i = 2:n-1
        mat[i+1:bu+i+1,i+1] = (2*i-1)*subtraction((@view mat[i:bu+i,i]), i-1) + [mat[i+1:bu+i-1,i-1];0;0]
    end

    #Reflection
    v = ones(T, bu)
    v[1:2:end] .= -1
    for i = 1:n-bu
        mat[i,i+1:i+bu] = (2*i-1)*v.*(@view mat[i+1:i+bu,i])./(2*(i:i+bu-1).+1)
    end
    for i = n-bu+1:n-1
        mat[i,i+1:n] = (2*i-1)*v[1:n-i].*(@view mat[i+1:n,i])./(2*(i:n-1).+1)
    end

    # normalized Legendre basis
    w_right = spdiagm(n, n, sqrt.(Vector{T}((2*(0:n-1).+1)/T(2))))
    w_left =  spdiagm(n+bl, n+bl, sqrt.(Vector{T}(T(2)./(2*(0:n+bl-1).+1))))
    mat = BandedMatrix{T}(w_left)*mat*BandedMatrix{T}(w_right)

    return (dom.right - dom.left)/2*mat
end


