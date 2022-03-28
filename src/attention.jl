function counter_to_index(l::Int, N::Int, Q::Int, H::Int; verbose::Bool=false)
    h::Int = 0
    if l <= H*N*N
            j::Int = ceil(l/(N*H))
            l = l-(N*H)*(j-1)
            i::Int = ceil(l/H)
            h = l-H*(i-1)
            verbose && println("h = $h \ni = $i \nj = $j\n")
            return h,i,j
    else
            l-=N*N*H
            b::Int = ceil(l/(Q*H))
            l-=(Q*H)*(b-1)
            a::Int = ceil(l/H)
            h = l-H*(a-1)
            verbose && println("h = $h \na = $a \nb = $b \n")
            return h, a, b
    end
end

function pl_and_grad!(x, grad, plmvar::PlmVar)

    pg = pointer(grad)
    H = plmvar.H 
    q = plmvar.q
    Z = plmvar.Z
    lambda = plmvar.lambdaJ
    N,M = size(Z)


    L = H*N*N + H*q*q 

    L == length(x) || error("Wrong dimension of parameter vector")
    L == length(grad) || error("Wrong dimension of gradient vector")

    W = reshape(x[1:H*N*N],H,N,N)
    V = reshape(x[H*N*N+1:end],H,q,q)

    pseudologlikelihood = zeros(Float64, N)
    grad .= 0.0
     
    Threads.@threads for site = 1:N #l'upgrade è fatto male non so perchè
        pseudologlikelihood[site] = update_gradW_site!(Z,W,V,grad,lambda,site)
    end

    update_gradV!(Z,W,V,grad,lambda)
    
    pg == pointer(grad) || error("Different pointer")
    L2 = lambda*L2Tensor(W) + lambda*L2Tensor(V)
    total_loglike = sum(pseudologlikelihood)
    println(total_loglike," ",L2)
    return total_loglike + L2

end

function update_gradW_site!(Z,W,V,grad,lambda,site)
    
    pg = pointer(grad)

    N,M = size(Z)
    H,q,q = size(V)

    L = H*N*N+H*q*q


    W_site = view(W,:,:,site)
    Wsf_site = softmax(W_site,dims=2)
    @tullio J[a,b,j] := Wsf_site[h,j]*V[h,a,b]*(site!=j)
    
    @tullio mat_ene[a,m] := J[a,Z[j,m],j]

    pl = 0.0
    partition = sumexp(mat_ene,dims=1)
    @tullio prob[a,m] := exp(mat_ene[a,m])/partition[m]
    lge = log.(partition)
    Z_site = view(Z,site,:)
    @tullio pl = mat_ene[Z_site[m],m] - lge[m]
    pl /= -M
    

    grad[(site-1)*N*H+1:site*N*H] .= 0.0

    @tullio mat[j,a,b] := (Z[j,m]==b)*((Z_site[m]==a)-prob[a,m]) (a in 1:q, b in 1:q)
    mat[site,:,:] .= 0.0

    @inbounds for counter in (site-1)*N*H+1:site*N*H 
        h,i,r = counter_to_index(counter,N,q,H)
        if r == site 
            for b = 1:q
                @simd for a = 1:q 
                    for j = 1:N 
                        grad[counter] += mat[j,a,b]*V[h,a,b]*((i==j)*Wsf_site[h,i]-Wsf_site[h,i]*Wsf_site[h,j])      
                    end
                end
            end
        end
        grad[counter] /= -M
        grad[counter] += 2*lambda*W[h,i,r]
    end

    pg == pointer(grad) || error("Different pointer")

    return pl
end


function update_gradV!(Z,W,V,grad,lambda)
    pg = pointer(grad)


    N,M = size(Z)
    H,q,q = size(V)
    L = H*N*N + H*q*q


    grad[H*N*N+1:end] .= 0.0

    Wsf = softmax(W,dims=2) #Ws[h,i,site]
    @tullio J[a,b,j,site] := Wsf[h,j,site]*V[h,a,b]*(site!=j)

    @tullio mat_ene[a,m,site] := J[a,Z[j,m],j,site]

    partition = sumexp(mat_ene,dims=1)
    part = view(partition,1,:,:)
    @tullio prob[a,m,site] := exp(mat_ene[a,m,site])/part[m,site]
    
    @tullio mat[site,j,a,b] := (Z[j,m]==b)*((Z[site,m]==a)-prob[a,m,site])*(site!=j) (a in 1:q, b in 1:q)
    # mat[site,:,:] .= 0.0


    @inbounds for counter = H*N*N+1:L 
        h,c,d = counter_to_index(counter,N,q,H)
        Wsf_h = view(Wsf,h,:,:)
        mat_cd = view(mat,:,:,c,d)
        @tullio g = Wsf_h[j,site]*mat_cd[site,j]
        grad[counter] = g        
        # grad[counter] += 2*lambdaJ*V[h,c,d]
        grad[counter] /= -M     
        grad[counter] += 2*lambda*V[h,c,d]
    end 


    pg == pointer(grad) || error("Different pointer")

    return 
end

function logsumexp(a::AbstractArray{<:Real}; dims=1)
    m = maximum(a; dims=dims)
    return m + log.(sum(exp.(a .- m); dims=dims))
end

function sumexp(a::AbstractArray{<:Real};dims=1)
    m = maximum(a; dims=dims)
    return sum(exp.(a .- m ).*exp.(m); dims=dims)
end





### Cerca di parallelizzare anche il calcolo di V e vedi se è più veloce
 
# function update_gradV_site!(Z,W,V,grad,site,lambdaJ)
#     pg = pointer(grad)

#     N,M = size(Z)
#     H,q,q = size(V)
#     L = H*N*N + H*q*q

#     grad[H*N*N+1:end] .= 0.0
#     W_site = view(W,:,:,site)
#     Wsf_site = softmax(W_site,dims=2) #Ws[h,i]
#     @tullio J[a,b,j] := Wsf_site[h,j]*V[h,a,b]*(site!=j)

#     @tullio mat_ene[a,m] := J[a,Z[j,m],j]

#     partition = sumexp(mat_ene,dims=1)
#     @tullio prob[a,m] := exp(mat_ene[a,m])/partition[m]
#     Z_site = view(Z,site,:)
#     @tullio mat[j,a,b] := (Z[j,m]==b)*((Z_site[m]==a)-prob[a,m,site]) (a in 1:q, b in 1:q)
#     mat[site,:,:] .= 0.0


#     @inbounds for counter = H*N*N+1:L 
#         h,c,d = counter_to_index(counter,N,q,H)
#         Wsf_h = view(Wsf_site,h,:)
#         mat_cd = view(mat,:,:,c,d)
#         @tullio g = Wsf_h[j]*mat_cd[site,j]
#         grad[counter] += -g/M        
#         # grad[counter] += 2*lambdaJ*V[h,c,d]     
#     end 

#     pg == pointer(grad) || error("Different pointer")

#     return 
# end


function JL2pl_and_grad!(x, grad, plmvar::PlmVar)
    
    pg = pointer(grad)
    H = plmvar.H 
    q = plmvar.q
    Z = plmvar.Z
    lambda = plmvar.lambdaJ
    N,M = size(Z)


    L = H*N*N + H*q*q 

    L == length(x) || error("Wrong dimension of parameter vector")
    L == length(grad) || error("Wrong dimension of gradient vector")

    W = reshape(x[1:H*N*N],H,N,N)
    V = reshape(x[H*N*N+1:end],H,q,q)

    pseudologlikelihood = zeros(Float64, N)
    l2 = zeros(Float64, N)
    grad .= 0.0
     
    Threads.@threads for site = 1:N
        pseudologlikelihood[site],l2[site] = update_gradW_siteJL2!(Z,W,V,grad,lambda,site)
    end


    update_gradVL2!(Z,W,V,grad,lambda)
    
    pg == pointer(grad) || error("Different pointer")
    
    total_loglike = sum(pseudologlikelihood)
    L2 = sum(l2)
    println(total_loglike," ",L2)
    return total_loglike + L2

end

function update_gradW_siteJL2!(Z,W,V,grad,lambda,site)
    
    pg = pointer(grad)

    N,M = size(Z)
    H,q,q = size(V)

    L = H*N*N+H*q*q


    W_site = view(W,:,:,site)
    Wsf_site = softmax(W_site,dims=2)
    @tullio J[a,b,j] := Wsf_site[h,j]*V[h,a,b]*(site!=j)
    
    @tullio mat_ene[a,m] := J[a,Z[j,m],j]

    pl = 0.0
    partition = sumexp(mat_ene,dims=1)
    @tullio prob[a,m] := exp(mat_ene[a,m])/partition[m]
    lge = log.(partition)
    Z_site = view(Z,site,:)
    @tullio pl = mat_ene[Z_site[m],m] - lge[m]
    pl /= -M
    

    grad[(site-1)*N*H+1:site*N*H] .= 0.0

    @tullio mat[j,a,b] := (Z[j,m]==b)*((Z_site[m]==a)-prob[a,m]) (a in 1:q, b in 1:q)
    mat[site,:,:] .= 0.0

    @inbounds for counter in (site-1)*N*H+1:site*N*H 
        L2 = 0.0
        h,i,r = counter_to_index(counter,N,q,H)
        if r == site 
            for b = 1:q
                @simd for a = 1:q 
                    for j = 1:N 
                        grad[counter] += mat[j,a,b]*V[h,a,b]*((i==j)*Wsf_site[h,i]-Wsf_site[h,i]*Wsf_site[h,j]) 
                        L2 += J[a,b,j]*V[h,a,b]*((i==j)*Wsf_site[h,j] - Wsf_site[h,i]*Wsf_site[h,j])
                    end
                end
            end
        end
        grad[counter] /= -M
        grad[counter] += lambda*L2/length(W)

    end

    pg == pointer(grad) || error("Different pointer")

    return pl, lambda*L2Tensor(J)
end


function update_gradVL2!(Z,W,V,grad,lambda)
    pg = pointer(grad)


    N,M = size(Z)
    H,q,q = size(V)
    L = H*N*N + H*q*q


    grad[H*N*N+1:end] .= 0.0

    Wsf = softmax(W,dims=2) #Ws[h,i,site]
    @tullio J[a,b,j,site] := Wsf[h,j,site]*V[h,a,b]*(site!=j)

    @tullio mat_ene[a,m,site] := J[a,Z[j,m],j,site]

    partition = sumexp(mat_ene,dims=1)
    part = view(partition,1,:,:)
    @tullio prob[a,m,site] := exp(mat_ene[a,m,site])/part[m,site]
    
    @tullio mat[site,j,a,b] := (Z[j,m]==b)*((Z[site,m]==a)-prob[a,m,site])*(site!=j) (a in 1:q, b in 1:q)
    # mat[site,:,:] .= 0.0


    @inbounds for counter = H*N*N+1:L 
        L2 = 0.0
        h,c,d = counter_to_index(counter,N,q,H)
        Wsf_h = view(Wsf,h,:,:)
        mat_cd = view(mat,:,:,c,d)
        @tullio g = Wsf_h[j,site]*mat_cd[site,j]
        grad[counter] = g
        J_cd = view(J,c,d,:,:)
        @tullio L2 = J_cd[y,x]*Wsf_h[y,x]        
        
        grad[counter] /= -M     
        grad[counter] += 2*lambda*L2
    end 

    pg == pointer(grad) || error("Different pointer")

    return 
end
