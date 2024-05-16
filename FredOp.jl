include("./OpPseSp.jl")


# defintion of Fredholm integral operator
struct FredOp{T<:FloatOrComplex} <: Op{T}
    fred::AbstractMatrix{T} # matrix representation of Fredholm integral operator
    dom::Domain 
    qrdata::Matrix{T}
end

FredOp(fred::AbstractMatrix{T}, dom::Domain) where T = FredOp{T}(fred, dom, schur(fred).T)

# default constructor
FredOp(kernelCoeffs::Vector{T}, dom::Domain) where T = 
FredOp(fredConvMatrix(kernelCoeffs, dom), dom)

# redefine "-"
-(op::FredOp{T}, z::FloatOrComplex) where T = FredOp(op.fred-z*I, op.dom, op.qrdata-z*I)

# redefine "\"
\(op::FredOp{T}, y::AbstractVector) where T = op.qrdata\y

# innerproduct between functions
InnerProduct(x::AbstractVector{T}, y::AbstractVector{T}) where T = (n = min(length(x), length(y)); x[1:n]'*y[1:n])

# norm of function
Norm(y::AbstractVector{T}) where T = norm(y)

# construct the matrix representation of Fredholm convolution integral operator
function inte(v::Vector{T}) where T<:FloatOrComplex
    n = size(v,1)
    b = zeros(T, n+1)
    b[1] = v[1]-v[2]/3
    v = [v./(2*((0:n-1).+0.5)); 0; 0]
    b[2:end] = v[1:n]-v[3:n+2]
    return b
end

function Trans_p1(n::Int64, ::Type{T}) where T<:FloatOrComplex
    Tr = zeros(T,n,n)
    Tr[1,1] = 1
    if (n > 1) 
        Tr[1,2] = Tr[2,2] = 0.5
    end

    if (n > 2)
        Tr[1,3] = 0
        Tr[2,3] = 3/4
        Tr[3,3] = 1/4
    end

    if (n > 3)
        for k = 4:n
            Tr[1,k] = (2*k-3)/(2k-2)*Tr[1,k-1] - 
            (k-2)/(k-1)*Tr[1,k-2]+(2*k-3)/(6k-6)*Tr[2,k-1]
 
            for p = 2:k-1
                Tr[p,k] = (2k-3)/(2k-2)*Tr[p,k-1]-(k-2)/(k-1)*Tr[p,k-2] +
                    (2k-3)/(2k-2)*(p/(2p+1)*Tr[p+1,k-1] + (p-1)/(2p-3)*Tr[p-1,k-1])
            end    
 
            Tr[k,k] = Tr[k-1,k-1]/2
        end    
    end    
    return Tr
end


function Trans_n1(n::Int64, ::Type{T}) where T<:FloatOrComplex
    Tr = zeros(T,n,n)
    Tr[1,1] = 1
    if (n > 1) 
        Tr[1,2] = -1/2
        Tr[2,2] = 1/2
    end

    if (n > 2)
        Tr[1,3] = 0
        Tr[2,3] = -3/4
        Tr[3,3] = 1/4
    end

    if (n > 3)
        for k = 4:n
            Tr[1,k] = -(2*k-3)/(2k-2)*Tr[1,k-1] - 
            (k-2)/(k-1)*Tr[1,k-2]+(2*k-3)/(6k-6)*Tr[2,k-1]

        for p = 2:k-1
            Tr[p,k] = -(2k-3)/(2k-2)*Tr[p,k-1]-(k-2)/(k-1)*Tr[p,k-2] +
            (2k-3)/(2k-2)*(p/(2p+1)*Tr[p+1,k-1] + (p-1)/(2p-3)*Tr[p-1,k-1])
        end

        Tr[k,k] = Tr[k-1,k-1]/2
        end    
    end    
    return Tr
end 

function p1muti(v::Vector{T}) where T<:FloatOrComplex
    n = size(v,1)
    b = zeros(T, n+1)
    if (n > 1)
        b[1] = v[2] / T(3)
        for k = 2:(n-1)
            b[k] = v[k-1] * T(k-1) / T(2*k-3) + v[k+1] * T(k) / T(2*k+1)
        end
        if n > 2
            b[n] = v[n-1] * T(n-1) / T(2*n-3)
            b[n+1] = v[n] * T(n) / T(2*n-1)
        else
            b[2] = v[1]
            b[3] = T(2) * v[2] / T(3)
        end
    end
    return b
end


function first_list(v::Vector{T}) where T<:FloatOrComplex
    b = 2.0.*inte(v)
    n = size(b,1)
    T_p1 = Trans_p1(n, T)
    T_n1 = Trans_n1(n, T)
    b = T_p1*b-T_n1*b
    b = b[1:end-1]
    return b
end



function fredConvMatrix(kernel::Vector{T}, dom::Domain) where T<:FloatOrComplex
    n = size(kernel,1)
    mat = zeros(T,n,n)

    #Initial the 0th column
    mat[1:n,1] = first_list(kernel)

    #Initial the 1th column
    l = p1muti(mat[1:n,1])-2.0.*first_list(p1muti(kernel))
    mat[2:n-1,2] = l[2:end-2]


    #Recursion below the diagonal
    for j = 3:Int64(ceil(n/2))
        for i = j:n-j+1
            mat[i,j] = T(2j-3)*(mat[i-1,j-1]/(2i-3)-mat[i+1,j-1]/(2i+1))+
                mat[i,j-2]
        end
    end

    for i = Int64(ceil(n/2)):-1:2
        for j = i+1:n-i+1
            mat[i,j] = T(2i-1)/(2j-1)*(mat[i+1,j+1]-mat[i+1,j-1])+
                T(2i-1)/(2i+3)*mat[i+2,j]
        end
    end
    for j = 2:n-1
        mat[1,j] = 1/T(2j-1)*(mat[2,j+1]-mat[2,j-1])+
            T(1)/5*mat[3,j]
    end
    
    mat[1,n] =T(1)/(2*n-1)*(-mat[2,n-1])+T(1)/5*mat[3,n]

    # normalized Legendre basis
    w_right = spdiagm(n, n, sqrt.(Vector{T}((2*(0:n-1).+1)/T(2))))
    w_left =  spdiagm(n, n, sqrt.(Vector{T}(T(2)./(2*(0:n-1).+1))))
    mat = w_left*mat*w_right

    return (dom.right - dom.left)/2*mat
end


