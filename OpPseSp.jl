using ApproxFun
using LinearAlgebra
using SparseArrays
using BandedMatrices
using Plots
import Base: -, \

const AbstractComplex = Complex{T} where T <: AbstractFloat
const FloatOrComplex = Union{AbstractFloat, AbstractComplex}

abstract type Op{T<:FloatOrComplex} end

include("./BandedQr.jl")
include("./PseSp.jl")
