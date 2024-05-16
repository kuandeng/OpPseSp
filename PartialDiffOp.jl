include("./OpPseSp.jl")


# definition of 2D partial differential operator using ApproxFun
struct PartialDiffOp{T<:FloatOrComplex} <: Op{T}
    diff::Operator{T} # differential operator
    B::Operator  # boundary condition operator
    valB # boundary condition values
    qrData::Operator{T} # resolvent operator  
    tol::Float64 # tolerance for solver for speedup
    PartialDiffOp{T}(diff::Operator{T}, B::Operator, valB, tol::Float64) where T = new(diff, B, valB, qr([B; diff]), tol)
end

# default constructor
PartialDiffOp(diff::Operator{T}, B::Operator, valB, tol::Float64) where T = PartialDiffOp{T}(diff, B, valB, tol)

# redefine "-"
-(op::PartialDiffOp{T}, z::FloatOrComplex) where T = PartialDiffOp(op.diff-z*I, op.B, op.valB, op.tol)

# redefine "\"
\(op::PartialDiffOp{T}, y::Fun{<:TensorSpace}) where T = \(op.qrData, [op.valB, y]; tolerance = op.tol)

# innerproduct between functions under Lengendre basis
function InnerProduct(x::Fun{<:TensorSpace}, y::Fun{<:TensorSpace})
    n = min(length(x.coefficients), length(y.coefficients))
    count = 0
    cur = 0
    temp = Vector{eltype(y.coefficients)}(undef, n)

    for i = 1:n
        temp[i] =  4/(2*cur+1)/(2*(count-cur)+1)*y.coefficients[i]
        if cur < count
            cur += 1
        else 
            cur = 0
            count += 1   
        end
    end

    dom1 = y.space.spaces[1].domain 
    dom2 = y.space.spaces[2].domain
    scale1 = (dom1.right - dom1.left)/2
    scale2 = (dom2.right - dom2.left)/2

    scale1*scale2*x.coefficients[1:n]'*temp
end

# norm of function
Norm(y::Fun{<:TensorSpace}) = sqrt(real(InnerProduct(y, y)))