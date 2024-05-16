# check convergence using adative stopping criterion
function checkConv(H, β, tol)
    d, Y = eigen(Symmetric(H))
    res = abs.(β*Y[end, :])
    # get largest rize values
    idx = sortperm(d, rev=true)
    d = d[idx]
    res = res[idx]
    isconv = sum(res[1] < tol*abs.(d[1]).^(3/2))
    return isconv, d, Y, idx
end

struct Options
    p::Int
    maxit::Int
    tol::Float64
end

# operator analogue of the core EigTool algorithm
function lanczos(Lz::Op{T}, Lz_ad::Op{T}, u1, ops::Options) where T

    # lanczos parameters
    p = ops.p
    maxit = ops.maxit
    tol = ops.tol

    # parameters
    U = Vector{typeof(u1)}(undef, p)
    u = u1
    sizeU = 1
    justRestarted = false
    α = zero(T)
    β = zero(T)
    H = zeros(real(T), p, p)
    d = zero(T)
    # main loop
    for mm = 1:maxit
        for jj = sizeU:p
            U[jj] = u

            # lanczos iteration
            w = Lz\u 
            norm_w = Norm(w)
            w = w/norm_w
            w = Lz_ad\w 
            w = w*norm_w
            if jj > 1
                w = w - β*U[jj-1]
            end 
            α = real(InnerProduct(u, w))
            if justRestarted
                for kk = 1:jj
                    w = w - InnerProduct(U[kk], w)*U[kk]
                end
                justRestarted = false
            else 
                w = w - α*u
            end

            # # simple reorthogonalize
            # for kk = 1:jj
            #     w = w - InnerProduct(U[kk], w)*U[kk]
            # end

            β = Norm(w)
            u = w/β
            
            # construct H
            H[jj, jj] = α
            if jj < p
                H[jj, jj+1] = H[jj+1, jj] = β
            end
            # check convergence
            @views isconv, d, Y, idx = checkConv(H[1:jj, 1:jj], β, tol)
            if isconv==1
                return 1/sqrt(d[1])
            end
        end

        if mm == maxit
            # maximum iteration times reached
            # @warn "please try more iteration"
            return 1/sqrt(d[1])
        else 
            # restart size
            k = ceil(Int, p/2)
        end

        # restart
        idx = idx[1:k]
        Y = Y[:,idx]
        U[1:k] = Y'*U

        # rebulid H
        H[1:k, 1:k] = diagm(d[1:k])
        H[1:k, k+1] = β*Y[end:end,:]
        H[k+1, 1:k] = β*Y[end:end,:]'
        justRestarted = true
        sizeU = k+1
    end
end

# computing resolvent norm for each grid points
function PseSp(L::Op{T}, L_ad::Op{T}, u1, ptx::Vector, pty::Vector, ops::Options) where T

    # grid parameters
    nptx = length(ptx)
    npty = length(pty)
    pse = zeros(real(T), npty, nptx)

    # main loop
    for i in 1:nptx
        println(i)
        for j in 1:npty 
            # shift operator
            z = ptx[i] + pty[j]*1.0im
            Lz = L - z
            Lz_ad = L_ad - z'
            pse[j, i] = lanczos(Lz, Lz_ad, u1, ops)
        end
    end             
    return pse                
end