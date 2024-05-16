include("./OpPseSp.jl")


# defintion of differential operator using ApproxFun
struct DiffOp{T<:FloatOrComplex} <: Op{T}
    diff::Operator{T} # differential expression
    B::Operator  # boundary condition operator
    valB::Union{Vector, FloatOrComplex} # boundary condition values
    qrData::Operator{T} # resolvent operator  
    DiffOp{T}(diff::Operator{T}, B::Operator, valB::Union{Vector, FloatOrComplex}) where T = new(diff, B, valB, qr([B; diff]))
end

# default constructor
DiffOp(diff::Operator{T}, B::Operator, valB::Union{Vector, FloatOrComplex}) where T = DiffOp{T}(diff, B, valB)

# redefine "-"
-(op::DiffOp{T}, z::FloatOrComplex) where T = DiffOp(op.diff-z*I, op.B, op.valB)

# redefine "\"
\(op::DiffOp{T}, y::Fun{<:Space}) where T = op.qrData\[op.valB; y]

# innerproduct between functions
function InnerProduct(x::Fun{<:Space}, y::Fun{<:Space})
    n = min(length(x.coefficients), length(y.coefficients))
    temp = [y.coefficients[i]*2/(2i-1) for i in 1:n]
    dom = y.space.domain
    scale = (dom.right - dom.left)/2
    scale*x.coefficients[1:n]'*temp
end

# norm of function
Norm(y::Fun{<:Space}) = sqrt(real(InnerProduct(y, y)))
