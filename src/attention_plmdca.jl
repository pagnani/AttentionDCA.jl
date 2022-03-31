function attention_plmdca(Z::Array{T,2},Weights::Vector{Float64}, H::Int, filedist::String;
                filename = "result.txt",
                min_separation::Int=1,
                theta=:auto,
                lambda::Real=0.00005,
                epsconv::Real=1.0e-4,
                maxit::Int=1000,
                verbose::Bool=true,
                method::Symbol=:LD_LBFGS) where T <: Integer

    all(x -> x > 0, Weights) || throw(DomainError("vector W should normalized and with all positive elements"))
    isapprox(sum(Weights), 1) || throw(DomainError("sum(W) ≠ 1. Consider normalizing the vector W"))
    N, M = size(Z)
    M = length(Weights)
    q = Int(maximum(Z))
    plmalg = PlmAlg(method, verbose, epsconv, maxit)
    plmvar = PlmVar(N, M, q, q*q, H, lambda, Z, Weights)
    dist = compute_residue_pair_dist(filedist)
    parameters, pslike = attentionMinimizePL(plmalg, plmvar, dist, filename)
    W = reshape(parameters[1:H*N*N],H,N,N)
    V = reshape(parameters[H*N*N+1:end],H,q,q)
    score = compute_dcascore(W, V)
    return PlmOut(pslike, W, V, score)

end

function attention_plmdca(filename::String,H,filedist::String;
                theta::Union{Symbol,Real}=:auto,
                max_gap_fraction::Real=0.9,
                remove_dups::Bool=true,
                kwds...)
    time = @elapsed Weights, Z, N, M, q = ReadFasta(filename, max_gap_fraction, theta, remove_dups)
    println("preprocessing took $time seconds")
    attention_plmdca(Z, Weights, H, filedist::String; kwds...)
end

plmdca(filename::String, H::Int, filedist; kwds...) = attention_plmdca(filename, H, filedist; kwds...)

function attentionMinimizePL(alg::PlmAlg, var::PlmVar, dist, filename::String; initx0 = nothing, Jreg = false)
    LL = var.H*var.N*var.N + var.H*var.q2
    x0 = if initx0 === nothing 
        rand(Float64, LL)*0.001
    else 
        initx0
    end
    pl = 0.0
    attention_parameters = zeros(LL) |> SharedArray
    
    opt = Opt(alg.method, length(x0))
    ftol_abs!(opt, alg.epsconv)
    xtol_rel!(opt, alg.epsconv)
    xtol_abs!(opt, alg.epsconv)
    ftol_rel!(opt, alg.epsconv)
    maxeval!(opt, alg.maxit)
    file = open(filename, "a")
    Jreg == false && min_objective!(opt, (x, g) -> optimfunwrapper(x, g, var, dist, file))
    Jreg == true && min_objective!(opt, (x, g) -> optimfunwrapperJreg(x, g, var))
    elapstime = @elapsed  (minf, minx, ret) = optimize(opt, x0)
    alg.verbose && @printf("pl = %.4f\t time = %.4f\t", minf, elapstime)
    alg.verbose && println("exit status = $ret")
    pl = minf
    attention_parameters .= minx
    close(file)
    return sdata(attention_parameters), pl
end


function pl_and_grad!(grad, x, plmvar::PlmVar, dist, file)

    pg = pointer(grad)
    N = plmvar.N
    M = plmvar.M 
    H = plmvar.H 
    q = plmvar.q
    q2 = plmvar.q2
    lambda = plmvar.lambda
    weights = plmvar.W
    Z = plmvar.Z
    
    sumweights = sum(weights)
    L = H*N*N + H*q2

    L == length(x) || error("Wrong dimension of parameter vector")
    L == length(grad) || error("Wrong dimension of gradient vector")

    W = reshape(x[1:H*N*N],H,N,N)
    V = reshape(x[H*N*N+1:end],H,q,q)

    pseudologlikelihood = zeros(Float64, N)
    grad .= 0.0
     
    Threads.@threads for site = 1:N #l'upgrade è fatto male non so perchè
        pseudologlikelihood[site] = update_gradW_site!(grad,Z,W,V,site,weights,sumweights,lambda)
    end

    update_gradV!(grad,Z,W,V,weights,sumweights,lambda)
    
    score = compute_dcascore(W,V)
    roc = compute_referencescore(score, dist)
    rocN = roc[N][end]
    rocN5 = roc[div(N,5)][end]

    pg == pointer(grad) || error("Different pointer")
    L2 = lambda*L2Tensor(W) + lambda*L2Tensor(V)
    total_loglike = sum(pseudologlikelihood)
    println(total_loglike," ",L2," ",roc[div(N,5)][end]," ",roc[N][end])
     
    write(file, "\n$total_loglike   $L2   $rocN5   $rocN")

    return total_loglike + L2

end

function update_gradW_site!(grad,Z,W,V,site,weights,sumweights,lambda)
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
    @tullio pl = weights[m]*(mat_ene[Z_site[m],m] - lge[m])
    pl /= -sumweights
    

    grad[(site-1)*N*H+1:site*N*H] .= 0.0

    @tullio mat[j,a,b] := weights[m]*(Z[j,m]==b)*((Z_site[m]==a)-prob[a,m]) (a in 1:q, b in 1:q)
    mat[site,:,:] .= 0.0
    @tullio fact[j,h] := mat[j,a,b]*V[h,a,b]

    @inbounds for counter in (site-1)*N*H+1:site*N*H 
        h,i,r = counter_to_index(counter,N,q,H)
        if r == site 
            @simd for j = 1:N 
                grad[counter] += fact[j,h]*((i==j)*Wsf_site[h,i]-Wsf_site[h,i]*Wsf_site[h,j]) 
            end
            grad[counter] /= -sumweights
        end
        grad[counter] += 2*lambda*W[h,i,r]
    end

    pg == pointer(grad) || error("Different pointer")

    return pl
end


function update_gradV!(grad,Z,W,V,weights,sumweights,lambda)
    pg = pointer(grad)

    N,_ = size(Z)
    H,q,q = size(V)

    L = H*N*N + H*q*q

    grad[H*N*N+1:end] .= 0.0

    Wsf = softmax(W,dims=2) #Ws[h,i,site]
    @tullio J[a,b,j,site] := Wsf[h,j,site]*V[h,a,b]*(site!=j)

    @tullio mat_ene[a,m,site] := J[a,Z[j,m],j,site]

    partition = sumexp(mat_ene,dims=1)
    part = view(partition,1,:,:)
    @tullio prob[a,m,site] := exp(mat_ene[a,m,site])/part[m,site]
    
    @tullio mat[site,j,a,b] := weights[m]*(Z[j,m]==b)*((Z[site,m]==a)-prob[a,m,site])*(site!=j) (a in 1:q, b in 1:q)
    # mat[site,:,:] .= 0.0


    @inbounds for counter = H*N*N+1:L 
        h,c,d = counter_to_index(counter,N,q,H)
        Wsf_h = view(Wsf,h,:,:)
        mat_cd = view(mat,:,:,c,d)
        @tullio g = Wsf_h[j,site]*mat_cd[site,j]
        grad[counter] = g        

        grad[counter] /= -sumweights  
        grad[counter] += 2*lambda*V[h,c,d]
    end 


    pg == pointer(grad) || error("Different pointer")

    return 
end

function pl_and_grad_Jreg!(grad,x,plmvar::PlmVar)
    
    pg = pointer(grad)
    H = plmvar.H 
    q = plmvar.q
    Z = plmvar.Z
    lambda = plmvar.lambda
    N,M = size(Z)
    weights = plmvar.W
    sumweights = sum(weights)

    L = H*N*N + H*q*q 

    L == length(x) || error("Wrong dimension of parameter vector")
    L == length(grad) || error("Wrong dimension of gradient vector")

    W = reshape(x[1:H*N*N],H,N,N)
    V = reshape(x[H*N*N+1:end],H,q,q)

    pseudologlikelihood = zeros(Float64, N)
    l2 = zeros(Float64, N)
    grad .= 0.0
     
    Threads.@threads for site = 1:N
        pseudologlikelihood[site],l2[site] = update_gradW_site_Jreg!(grad,Z,W,V,weights,sumweights,lambda,site)
    end


    update_gradV_Jreg!(grad,Z,W,V,weights,sumweights,lambda)
    
    pg == pointer(grad) || error("Different pointer")
    
    total_loglike = sum(pseudologlikelihood)
    L2 = sum(l2)
    println(total_loglike," ",L2)
    return total_loglike + L2

end

function update_gradW_site_Jreg!(Z,W,V,grad,weights,sumweights,lambda,site)
    
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
    @tullio pl = weights[m]*(mat_ene[Z_site[m],m] - lge[m])
    pl /= -sumweights
    

    grad[(site-1)*N*H+1:site*N*H] .= 0.0

    @tullio mat[j,a,b] := weights[m]*(Z[j,m]==b)*((Z_site[m]==a)-prob[a,m]) (a in 1:q, b in 1:q)
    mat[site,:,:] .= 0.0
    @tullio fact[j,h] := mat[j,a,b]*V[h,a,b]
    @tullio fact2[j,h] := J[a,b,j]*V[h,a,b]

    @inbounds for counter in (site-1)*N*H+1:site*N*H 
        L2 = 0.0
        h,i,r = counter_to_index(counter,N,q,H)
        if r == site 
            @simd for j = 1:N 
                grad[counter] += fact[j,h]*((i==j)*Wsf_site[h,i]-Wsf_site[h,i]*Wsf_site[h,j]) 
                L2 += fact2[j,h]*((i==j)*Wsf_site[h,j] - Wsf_site[h,i]*Wsf_site[h,j])
            end
        end
        grad[counter] /= -sumweights
        grad[counter] += lambda*L2

    end

    pg == pointer(grad) || error("Different pointer")

    return pl, lambda*L2Tensor(J)
end


function update_gradV_Jreg!(grad,Z,W,V,weights,sumweights,lambda)
    pg = pointer(grad)

    N,_ = size(Z)
    H,q,q = size(V)

    L = H*N*N + H*q*q

    grad[H*N*N+1:end] .= 0.0

    Wsf = softmax(W,dims=2) #Ws[h,i,site]
    @tullio J[a,b,j,site] := Wsf[h,j,site]*V[h,a,b]*(site!=j)

    @tullio mat_ene[a,m,site] := J[a,Z[j,m],j,site]

    partition = sumexp(mat_ene,dims=1)
    part = view(partition,1,:,:)
    @tullio prob[a,m,site] := exp(mat_ene[a,m,site])/part[m,site]
    
    @tullio mat[site,j,a,b] := weights[m]*(Z[j,m]==b)*((Z[site,m]==a)-prob[a,m,site])*(site!=j) (a in 1:q, b in 1:q)
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
        
        grad[counter] /= -sumweights     
        grad[counter] += 2*lambda*L2
    end 

    pg == pointer(grad) || error("Different pointer")

    return 
end
