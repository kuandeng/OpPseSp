using LinearAlgebra.LAPACK: larf!, larfg!, chkside
using LinearAlgebra.BLAS: blascopy!, @blasfunc, chkuplo, vec_pointer_stride
using LinearAlgebra: checksquare, BlasFloat, BlasInt, chkstride1
using LinearAlgebra
using libblastrampoline_jll
using Base: require_one_based_indexing

mutable struct BandedQrData{T<:FloatOrComplex}
    N :: Integer
    bu :: Integer
    bl :: Integer
    qrStep :: Integer
    workA :: Matrix{T}
    worky :: Vector{T}
    workH :: Vector{T}
    tau :: Vector{T}
    refl :: Matrix{T}
end

function BandedQrData(b::BandedMatrix{T}) where T
    bu = b.u
    bl = b.l
    N = size(b, 2)
    return BandedQrData(N, bu, bl, 0, Matrix{T}(undef, 1+bu+2*bl, N), Vector{T}(undef, N), Vector{T}(undef, bu+bl+1), Vector{T}(undef, N), Matrix{T}(undef, bl+1, N))
end

for (fname, elty) in ((:dtbsv_,:Float64),
    (:stbsv_,:Float32),
    (:ztbsv_,:ComplexF64),
    (:ctbsv_,:ComplexF32))
    @eval begin
                #       SUBROUTINE DTBSV(UPLO,TRANS,DIAG,N,A,LDA,X,INCX)
                #       .. Scalar Arguments ..
                #       INTEGER INCX,LDA,N
                #       CHARACTER DIAG,TRANS,UPLO
                #       .. Array Arguments ..
                #       DOUBLE PRECISION A(LDA,*),X(*)
        function tbsv!(uplo::AbstractChar, trans::AbstractChar, diag::AbstractChar, k::Integer, A::AbstractMatrix{$elty}, x::AbstractVector{$elty})
            chkuplo(uplo)
            require_one_based_indexing(A, x)
            n = size(A,2)
            if n != length(x)
                throw(DimensionMismatch(lazy"size of A is $n != length(x) = $(length(x))"))
            end
            chkstride1(A)
            px, stx = vec_pointer_stride(x, ArgumentError("input vector with 0 stride is not allowed"))
            GC.@preserve x ccall((@blasfunc($fname), libblastrampoline), Cvoid,
                (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
                Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                Clong, Clong, Clong),
                uplo, trans, diag, n, k,
                A, max(1,stride(A,2)), px, stx, 1, 1, 1)
            x
        end
        function tbsv(uplo::AbstractChar, trans::AbstractChar, diag::AbstractChar, k::Integer, A::AbstractMatrix{$elty}, x::AbstractVector{$elty})
            tbsv!(uplo, trans, diag, k, A, copy(x))
        end
    end
end

# shift banded matrix copy
function sbcopy!(X::Matrix{T}, Y::Matrix{T}, z::T, bu::Integer, bl::Integer, m::Integer, n::Integer) where T    
    @inbounds @views begin
        copyto!(Y[bl+1:end, m:n], X[1:end, m:n])
        Y[1:bl, m:n] .= T(0)
        Y[bu+bl+1, m:n] .-= z
    end
end

# banded matrix adaptive qr factor
for (larf, elty) in
    ((:dlarf_, Float64),
     (:slarf_, Float32),
     (:zlarf_, ComplexF64),
     (:clarf_, ComplexF32))
    @eval begin
        function baqrf!(A::Matrix{$elty}, bu::Integer, bl::Integer, qrStep::Integer, n::Integer, tau::Vector{$elty}, refl::Matrix{$elty}, workH::Vector{$elty})
            @inbounds @views for i = qrStep+1:n
                refl[1:end, i] = A[bu+1:bu+bl+1, i]
                v = refl[1:end, i]
                tau[i] = larfg!(v)
                ccall((@blasfunc($larf), libblastrampoline), Cvoid,
                (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                Ref{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Clong),
                'L', bl+1, bu+1, v, 1,
                tau[i]', A[bu+1:bu+bl+1, i:i+bu], bu+bl, workH, 1)
                # larf!('L', v, tau[i]', A[bu+1:bu+bl+1, i:i+bu], bu+bl, workH)
            end
        end
    end
end

# banded matrix adaptive qr solve 
function baqsv!(A::Matrix{T}, N::Integer, bu::Integer, bl::Integer, y::AbstractVector{T}, z::T, qrStep::Integer, tau::Vector{T}, refl::Matrix{T}, workA::Matrix{T}, worky::Vector{T}, workH::Vector{T}) where T
    @inbounds @views begin

        # parm initialize
        workbu = bu+bl # upper bandwith of workA
        tol = 1*eps(real(T)) # tolerance for adaptive qr
        incStep = 1 # increment step for adaptive qr

        # initial
        ny = length(y)
        if ny+incStep+workbu > N
            error("please try larger dimension")
        end
        normy = norm(y) 
        copyto!(worky, y)
        worky[ny+1:ny+bl] .= T(0)

        # make the valid columns number of workA always equals to qrStep plus workbu
        if qrStep == 0
            sbcopy!(A, workA, z, bu, bl, 1, qrStep+workbu)
        end

        # adaptive qr for banded matrix
        if (qrStep < ny)
            sbcopy!(A, workA, z, bu, bl, qrStep+workbu+1, ny+workbu)
            baqrf!(workA, workbu, bl, qrStep, ny, tau, refl, workH) 
            qrStep = ny
        end

        for i = 1:ny
            larf!('L', refl[1:end, i], tau[i]', worky[i:i+bl, 1:1], workH)
        end

        # main loop
        n = ny
        while (n+incStep+workbu <= N && norm(worky[n+1:n+bl]) > tol*normy)
            n_next = n+incStep 
            worky[n+bl+1:n_next+bl] .= 0
            if (qrStep < n_next)
                sbcopy!(A, workA, z, bu, bl, qrStep+workbu+1, n_next+workbu)
                baqrf!(workA, workbu, bl, qrStep, n_next, tau, refl, workH) 
                qrStep = n_next
            end
            for i = n+1:n_next
                larf!('L', refl[1:end, i], tau[i]', worky[i:i+bl, 1:1], workH)
            end
            n = n_next
        end
        if n+incStep+workbu > N
            error("please try larger dimension")
        else
            tbsv!('U', 'N', 'N', workbu, workA[1:workbu+1, 1:n], worky[1:n])
            return n, qrStep
        end
    end
end





