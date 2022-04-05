function parallel_pl_and_grad!(grad, x, plmvar::PlmVar, file; dist = nothing, verbose = true)
    pg = pointer(grad)
    N = plmvar.N
    M = plmvar.M 
    H = plmvar.H 
    q = plmvar.q
    q2 = plmvar.q2
    lambda = plmvar.lambda
    weights = plmvar.W
    Z = plmvar.Z


    L = H*N*N + H*q2*N

    L == length(x) || error("Wrong dimension of parameter vector")
    L == length(grad) || error("Wrong dimension of gradient vector")

    W = reshape(x[1:H*N*N],H,N,N)
    V = reshape(x[H*N*N+1:end],H,q,q,N)

    pseudologlikelihood = zeros(Float64, N)
    grad .= 0.0

    Threads.@threads for site = 1:N #l'upgrade è fatto male non so perchè
        pseudologlikelihood[site] = parallel_update_gradW_site!(grad,Z,W,V,site,weights,lambda)
    end

    Threads.@threads for site = 1:N #l'upgrade è fatto male non so perchè
        parallel_update_gradV_site!(grad,Z,W,V,site,weights,lambda)
    end

    L2 = lambda*L2Tensor(W) + lambda*L2Tensor(V)
    total_loglike = sum(pseudologlikelihood)

    if dist !== nothing 
        score = compute_dcascore(W,V)
        roc = compute_referencescore(score, dist)
        rocN = roc[N][end]
        rocN5 = roc[div(N,5)][end]

        file !== nothing && write(file, "$total_loglike   "*"$L2   "*"$rocN5   "*"$rocN"*"\n")
        verbose && println("$total_loglike   "*"$L2   "*"$rocN5   "*"$rocN")
    else
        file !== nothing && write(file, "$total_loglike   "*"$L2"*"\n")
        verbose && println("$total_loglike   "*"$L2")
    end

    pg == pointer(grad) || error("Different pointer")
    return total_loglike + L2

end

function parallel_update_gradW_site!(grad,Z,W,V,site,weights,lambda)
    pg = pointer(grad)


    H,q,q,N = size(V)

    L = H*N*N+H*q*q*N

    W_site = view(W,:,:,site)
    V_site = view(V,:,:,:,site)
    Wsf_site = softmax(W_site,dims=2)
    @tullio threads=true fastmath=true avx=true grad=false J[a,b,j] := Wsf_site[h,j]*V_site[h,a,b]*(site!=j)

    @tullio threads=true fastmath=true avx=true grad=false mat_ene[a,m] := J[a,Z[j,m],j]

    pl = 0.0
    partition = sumexp(mat_ene,dims=1)
    @tullio threads=true fastmath=true avx=true grad=false prob[a,m] := exp(mat_ene[a,m])/partition[m]
    lge = log.(partition)
    Z_site = view(Z,site,:)
    @tullio threads=true fastmath=true avx=true grad=false pl = weights[m]*(mat_ene[Z_site[m],m] - lge[m])
    pl *= -1


    grad[(site-1)*N*H+1:site*N*H] .= 0.0

    @tullio threads=true fastmath=true avx=true grad=false mat[j,a,b] := weights[m]*(Z[j,m]==b)*((Z_site[m]==a)-prob[a,m]) (a in 1:q, b in 1:q)
    mat[site,:,:] .= 0.0
    @tullio threads=true fastmath=true avx=true grad=false fact[j,h] := mat[j,a,b]*V_site[h,a,b]

    @inbounds for counter in (site-1)*N*H+1:site*N*H 
        h,i,r = counter_to_index(counter,N,q,H)
        if r == site 
            @simd for j = 1:N 
                grad[counter] += fact[j,h]*((i==j)*Wsf_site[h,i]-Wsf_site[h,i]*Wsf_site[h,j]) 
            end
            grad[counter] *= -1
        end
        grad[counter] += 2*lambda*W[h,i,r]
    end

    pg == pointer(grad) || error("Different pointer")

    return pl
end


function parallel_update_gradV_site!(grad,Z,W,V,site,weights,lambda)
    pg = pointer(grad)

    H,q,q,N = size(V)

    L = H*N*N + H*q*q*N

    W_site = view(W,:,:,site)
    V_site = view(V,:,:,:,site)
    Wsf_site = softmax(W_site,dims=2)

    grad[(site-1)*H*q*q+H*N*N+1:site*H*q*q+H*N*N] .= 0.0

    @tullio threads=true fastmath=true avx=true grad=false J[a,b,j] := Wsf_site[h,j]*V_site[h,a,b]*(site!=j)

    @tullio threads=true fastmath=true avx=true grad=false mat_ene[a,m] := J[a,Z[j,m],j]

    partition = sumexp(mat_ene,dims=1)
    @tullio threads=true fastmath=true avx=true grad=false prob[a,m] := exp(mat_ene[a,m])/partition[m]

    Z_site = view(Z,site,:)
    @tullio threads=true fastmath=true avx=true grad=false mat[j,a,b] := weights[m]*(Z[j,m]==b)*((Z_site[m]==a)-prob[a,m]) (a in 1:q, b in 1:q)
    mat[site,:,:] .= 0.0


    @inbounds for counter = (site-1)*H*q*q+H*N*N+1:site*H*q*q+H*N*N 
        h,c,d,i = counter_to_index_V_parallel(counter,N,q,H)
        if i==site
            mat_cd = view(mat,:,c,d)
            Wsf_h = view(Wsf_site,h,:)
            @tullio threads=true fastmath=true avx=true grad=false g = mat_cd[j]Wsf_h[j]
            grad[counter] = g        
            grad[counter] *= -1 
            grad[counter] += 2*lambda*V[h,c,d,i]
        end
    end 


    pg == pointer(grad) || error("Different pointer")

    return 
end

function counter_to_index_V_parallel(l::Int, N::Int, Q::Int, H::Int; verbose::Bool=false)
    l > H*N*N || error("Counter value too small")
    l -=H*N*N 
    i::Int = ceil(l/(H*Q*Q))
    l-=(H*Q*Q)*(i-1)
    b::Int = ceil(l/(Q*H))
    l-=(Q*H)*(b-1)
    a::Int = ceil(l/H)
    h = l-H*(a-1)
    verbose && println("h = $h \na = $a \nb = $b \ni = $i \n")
    return h, a, b, i
end



