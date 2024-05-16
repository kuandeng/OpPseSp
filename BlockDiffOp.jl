include("./DiffOp.jl")


# defintion of Block differential operator using ApproxFun
struct BlockDiffOp{T<:FloatOrComplex} <: Op{T}
    diff::Operator{T} # differential operator
    B::Operator  # boundary condition operator
    valB::Vector # boundary condition values
end

# definition of shift operator L - zI
struct ShiftBlockDiffOp{T<:FloatOrComplex} <: Op{T}
    blockDiffOp::BlockDiffOp{T}
    shift::FloatOrComplex
    qrData::Operator{T}
    ShiftBlockDiffOp{T}(op::BlockDiffOp{T}, shift::FloatOrComplex) where T = new(op, shift, qr([op.B; op.diff-shift*I]))
end

# redefine "-"
-(op::BlockDiffOp{T}, z::FloatOrComplex) where T = ShiftBlockDiffOp{T}(op, z)

# redefine "\"
\(op::ShiftBlockDiffOp{T}, y::Fun{<:ApproxFunBase.ArraySpace}) where T = op.qrData\[op.blockDiffOp.valB, y]

# innerproduct between functions
function InnerProduct(x::Fun{<:ApproxFunBase.ArraySpace}, y::Fun{<:ApproxFunBase.ArraySpace})
    temp = zero(eltype(x.coefficients))
    for i = 1:length(x.space)
        temp += InnerProduct(x[i], y[i])
    end
    return temp
end

# norm of function
Norm(y::Fun{<:ApproxFunBase.ArraySpace}) = sqrt(real(InnerProduct(y, y)))





