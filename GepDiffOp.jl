include("./DiffOp.jl")

for opname in (:GepDiffOp, :GepDiffOp_ad)
    @eval begin

        # defintion of generalized differential operator R^{-1}L and its adjoint 
        struct $opname{T<:FloatOrComplex} <: Op{T}
            L::DiffOp{T}
            R::DiffOp{T}
        end
        
        # definition if shift operator R^{-1}L - zI and its adjoint
        struct $(Symbol("Shift", opname)){T<:FloatOrComplex} <: Op{T}
            gepDiffOp::$opname{T}
            shift::FloatOrComplex
            qrData::Operator{T}
            $(Symbol("Shift", opname)){T}(op::$opname{T}, shift::FloatOrComplex) where T = new(op, shift, qr([op.L.B; op.L.diff-shift*(op.R.diff)]))
        end
        
        $(Symbol("Shift", opname))(op::$opname{T}, shift::FloatOrComplex) where T = $(Symbol("Shift", opname)){T}(op, shift)

        # redefine "-"
        -(op::$opname{T}, z::FloatOrComplex) where T = $(Symbol("Shift", opname))(op, z)        
    end
end

# redefine "\" for ShiftGepDiffOp
\(op::ShiftGepDiffOp{T}, y::Fun{<:Space}) where T = (op.qrData\[op.gepDiffOp.L.valB; op.gepDiffOp.R.diff*y])

# redefine "\" for ShiftGepDiffOp_ad
\(op::ShiftGepDiffOp_ad{T}, y::Fun{<:Space}) where T = op.gepDiffOp.R.diff*(op.qrData\[op.gepDiffOp.L.valB; y])

