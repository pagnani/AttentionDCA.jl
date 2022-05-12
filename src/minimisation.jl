function pslikelihood(x, var::PlmVar,Z,weights,limits)
    N = var.N
    M = var.M
    H = var.H
    q = var.q
    λ = N*q*(var.lambda)/M

    L = H*N*N + H*q*q #total numbero of parameters
    L == length(x) || error("Wrong dimension of parameter vector")
    #L == length(grad) || error("Wrong dimension of gradient vector")

    W = reshape(x[1:H*N*N],H,N,N)
    V = reshape(x[H*N*N+1:end],H,q,q)

    pseudologlikelihood = zeros(Float64, N)
    l2 = zeros(Float64, N)
    grad = zeros(Float64, L)
     
    J = zeros(Float64,q,q,N,N)
    Wsf = zeros(Float64,H,N,N)
    mat = zeros(Float64,N,N,q,q)


    Threads.@threads for site = 1:N
        pseudologlikelihood[site],l2[site],J[:,:,:,site],Wsf[:,:,site],mat[site,:,:,:] = new_update_gradW_site_Jreg!(grad,Z,W,V,weights,λ,site,limits)
    end
    
    
    new_update_gradV_Jreg!(grad,Z,Wsf,V,λ,J,mat)

   
    # pg == pointer(G) || error("Different pointer")
    
    total_loglike = sum(pseudologlikelihood)
    L2 = sum(l2)
    # if dist !== nothing 
    #     score = compute_dcascore(W,V)
    #     roc = compute_referencescore(score, dist)
    #     rocN = roc[N][end]
    #     rocN5 = roc[div(N,5)][end]
        
    #     file !== nothing && write(file, "$total_loglike   "*"$L2   "*"$rocN5   "*"$rocN"*"\n")
    #     verbose && println("$total_loglike   "*"$L2   "*"$rocN5   "*"$rocN")
    # else
    #     file !== nothing && write(file, "$total_loglike   "*"$L2"*"\n")
    #     verbose && println("$total_loglike   "*"$L2 ")
    # end
    println(total_loglike+L2,"  ", L2)
    return total_loglike + L2, grad 
    
end

function new_update_gradW_site_Jreg!(grad,Z,W,V,weights,λ,site,limits)
    pg = pointer(grad)
    N,_ = size(Z)
    H,q,_ = size(V)

    W_site = view(W,:,:,site)
    Wsf_site = softmax(W_site,dims=2)
    
    
    @tullio J[a,b,j] := Wsf_site[h,j]*V[h,a,b]*(site!=j)
    @tullio mat_ene[a,m] := J[a,Z[j,m],j]
    
    
    pl = 0.0
    partition = sumexp(mat_ene,dims=1)
    @tullio prob[a,m] := exp(mat_ene[a,m])/partition[m]
    
    lge = log.(partition)
    Z_site = view(Z,site,:)
    @tullio pl = weights[m]*(mat_ene[Z_site[m],m] - lge[m])
    pl *= -1
    

    grad[(site-1)*N*H+1:site*N*H] .= 0.0

    weights_minibatch = weights[limits]
    Z_minibatch = view(Z,:,limits)
    Z_site_minibatch = view(Z_site,limits)
    prob_minibatch = view(prob,:,limits)

    @tullio mat[j,a,b] := weights_minibatch[m]*(Z_minibatch[j,m]==b)*((Z_site_minibatch[m]==a)-prob_minibatch[a,m]) (a in 1:q, b in 1:q)
    mat[site,:,:] .= 0.0
    @tullio fact[j,h] := mat[j,a,b]*V[h,a,b]
    @tullio fact2[j,h] := 2*J[a,b,j]*V[h,a,b]

    @inbounds for counter in (site-1)*N*H+1:site*N*H 
        gradL2 = 0.0
        h,i,r = counter_to_index(counter,N,q,H)
        if r == site 
            @simd for j = 1:N 
                scra = ((i==j)*Wsf_site[h,i]-Wsf_site[h,i]*Wsf_site[h,j])
                grad[counter] += fact[j,h] *scra
                gradL2 += fact2[j,h]*scra
            end
        end
        grad[counter] *= -1
        grad[counter] += λ*gradL2

    end

    pg == pointer(grad) || error("Different pointer")
    return pl, λ*L2Tensor(J), J, Wsf_site , mat
end


function new_update_gradV_Jreg!(grad,Z,Wsf,V,λ,J,mat)
    pg = pointer(grad)

    N,_ = size(Z)
    H,q,_ = size(V)

    L = H*N*N + H*q*q

    grad[H*N*N+1:end] .= 0.0

    @inbounds for counter = H*N*N+1:L 
        L2 = 0.0
        h,c,d = counter_to_index(counter,N,q,H)
        Wsf_h = view(Wsf,h,:,:)
        mat_cd = view(mat,:,:,c,d)
        @tullio g = Wsf_h[j,site]*mat_cd[site,j]
        grad[counter] = g
        J_cd = view(J,c,d,:,:)
        @tullio L2 = J_cd[y,x]*Wsf_h[y,x]        
        
        grad[counter] *= -1     
        grad[counter] += 2*λ*L2
    end 

    pg == pointer(grad) || error("Different pointer")

    return 
end


# using Flux, Random
# using Flux.Optimise: update!


function my_minimiser(opt, x, var, Z, weights; x_epsconv=1.0e-5, f_epsconv=1.0e-5, maxit=1000, length_minibatches = 100)
    M = length(weights)
    perm = shuffle(1:M)
    Z = Z[:,perm]
    weights = weights[perm]
    number_minibatches = M ÷ length_minibatches
    j = 0
    finitial,grad = pslikelihood(x,var,Z, weights, 1:length_minibatches)
    xinitial = copy(x)
    f1 = 0
    for i in 1:maxit 
        j += 1 
        if j != number_minibatches
            update!(opt, x, grad)
            f1,grad = pslikelihood(x,var,Z, weights, j*length_minibatches + 1 : (j+1)*length_minibatches)
        
            if abs(f1 - finitial) <= f_epsconv 
                println("Maxtol_f reached at ",abs(f1 - finitial))
                return x, f1, "Maxtol_f reached", i
            end  

            if abs(maximum(xinitial-x)) <= x_epsconv
                println("Maxtol_x reached at ",abs(maximum(xinitial-x)))
                return x, f1, "Maxtol_x reached", i
            end
            finitial = copy(f1)
            xinitial = copy(x)
        else
            update!(opt, x, grad)
            f1,grad = pslikelihood(x,var,Z, weights,(number_minibatches-1)*length_minibatches+1:M)
        
            if abs(f1 - finitial) <= f_epsconv 
                println("Maxtol_f reached at ",abs(f1 - finitial))
                return x, f1, "Maxtol_f reached", i
            end  

            if abs(maximum(xinitial-x)) <= x_epsconv
                println("Maxtol_x reached at ",abs(maximum(xinitial-x)))
                return x, f1, "Maxtol_x reached", i
            end
            finitial = copy(f1)
            xinitial = copy(x)
            j = 0
        end
    end

    return x, f1, "Maxeval reached", maxit
end

f(x) = sum(x.^2)
function g!(grad,x)       
    return grad .= 2*x 
end