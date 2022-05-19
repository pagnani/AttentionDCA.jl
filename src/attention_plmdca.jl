function attention_plmdca(Z::Array{T,2},Weights::Vector{Float64};
                H::Int = 32,
                structfile::Union{String,Nothing} = nothing, 
                output::Union{String,Nothing} = nothing,
                parallel = false,
                Jreg = false,
                initx0 = nothing,
                min_separation::Int=6,
                theta=:auto,
                lambda::Real=0.00005,
                epsconv::Real=1.0e-5,
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
    dist = if structfile !== nothing 
        compute_residue_pair_dist(structfile)
    else
        nothing 
    end
    parameters, pslike = attentionMinimizePL(plmalg, plmvar, dist=dist, output=output,verbose=verbose,parallel=parallel,Jreg=Jreg, initx0=initx0)
    W = reshape(parameters[1:H*N*N],H,N,N)
    if parallel 
        V = reshape(parameters[H*N*N+1:end],H,q,q,N)
        score = parallel_compute_dcascore(W, V)
    else
        V = reshape(parameters[H*N*N+1:end],H,q,q)
        score = compute_dcascore(W, V)
    end
    return PlmOut(pslike, W, V, score)
end

function attention_plmdca(filename::String;
                theta::Union{Symbol,Real}=:auto,
                max_gap_fraction::Real=0.9,
                remove_dups::Bool=true,
                kwds...)
    time = @elapsed Weights, Z, N, M, q = ReadFasta(filename, max_gap_fraction, theta, remove_dups)
    println("preprocessing took $time seconds")
    attention_plmdca(Z, Weights; kwds...)
end

function attentionMinimizePL(alg::PlmAlg, var::PlmVar;
                initx0 = nothing, 
                dist = nothing, 
                output::Union{Nothing, String} = nothing, 
                Jreg = false,
                parallel = false,
                verbose = true)
    LL = if parallel
        var.H*var.N*var.N + var.H*var.q2*var.N
    else
        var.H*var.N*var.N + var.H*var.q2
    end
    x0 = if initx0 === nothing 
        rand(Float64, LL)*0.0001
    else 
        initx0
    end
    pl = 0.0
    attention_parameters = zeros(LL)
 
    opt = Opt(alg.method, length(x0))
    ftol_abs!(opt, alg.epsconv)
    xtol_rel!(opt, alg.epsconv)
    xtol_abs!(opt, alg.epsconv)
    ftol_rel!(opt, alg.epsconv)
    maxeval!(opt, alg.maxit)
    file = if output !== nothing 
        open(output, "a")
    else
        nothing
    end
    if Jreg==false
        min_objective!(opt, (x, g) -> optimfunwrapper(x, g, var, file, dist=dist, verbose=verbose, parallel=parallel))
    else
        min_objective!(opt, (x, g) -> optimfunwrapperJreg(g, x, var, file, dist=dist, verbose=verbose))
    end
    elapstime = @elapsed  (minf, minx, ret) = optimize(opt, x0)
    alg.verbose && @printf("pl = %.4f\t time = %.4f\t", minf, elapstime)
    alg.verbose && println("exit status = $ret")
    pl = minf
    attention_parameters .= minx
    if output !== nothing 
        close(file)
    end
    return attention_parameters, pl
end

function pl_and_grad_Jreg!(grad, x, plmvar, file; dist = nothing, verbose = true)
    pg = pointer(grad)
    H = plmvar.H 
    N = plmvar.N
    M = plmvar.M
    q = plmvar.q
    Z = plmvar.Z
    λ = N*q*(plmvar.lambda)/M
    N,M = size(Z)
    weights = plmvar.W
    

    L = H*N*N + H*q*q 
    L == length(x) || error("Wrong dimension of parameter vector")
    L == length(grad) || error("Wrong dimension of gradient vector")

    W = reshape(x[1:H*N*N],H,N,N)
    V = reshape(x[H*N*N+1:end],H,q,q)

    pseudologlikelihood = zeros(Float64, N)
    l2 = zeros(Float64, N)
    grad .= 0.0
     
    J = zeros(Float64,q,q,N,N)
    Wsf = zeros(Float64,H,N,N)
    mat = zeros(Float64,N,N,q,q)
    
    Threads.@threads for site = 1:N
        pseudologlikelihood[site],l2[site],J[:,:,:,site],Wsf[:,:,site],mat[site,:,:,:] = update_gradW_site_Jreg!(grad,Z,W,V,weights,λ,site)
    end
    
    
    update_gradV_Jreg!(grad,Z,Wsf,V,λ,J,mat)

    pg == pointer(grad) || error("Different pointer")
    
    total_loglike = sum(pseudologlikelihood)
    L2 = sum(l2)
    if dist !== nothing 
        score = compute_dcascore(W,V)
        roc = compute_referencescore(score, dist)
        rocN = roc[N][end]
        rocN5 = roc[div(N,5)][end]
        
        file !== nothing && write(file, "$total_loglike   "*"$L2   "*"$rocN5   "*"$rocN"*"\n")
        verbose && println("$total_loglike   "*"$L2   "*"$rocN5   "*"$rocN")
    else
        file !== nothing && write(file, "$total_loglike   "*"$L2"*"\n")
        verbose && println("$total_loglike   "*"$L2 ")
    end
    return total_loglike + L2

end

function update_gradW_site_Jreg!(grad,Z,W,V,weights,λ,site)
    
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

    @tullio mat[j,a,b] := weights[m]*(Z[j,m]==b)*((Z_site[m]==a)-prob[a,m]) (a in 1:q, b in 1:q)
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


function update_gradV_Jreg!(grad,Z,Wsf,V,λ,J,mat)
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


function pl_and_grad!(grad, x, plmvar::PlmVar, file; dist = nothing, verbose = true, parallel = false)
    pg = pointer(grad)
    N = plmvar.N
    M = plmvar.M 
    H = plmvar.H 
    q = plmvar.q
    q2 = plmvar.q2
    lambda = plmvar.lambda
    weights = plmvar.W
    Z = plmvar.Z
   

    L = if parallel  
        H*N*N + H*q2*N
    else
        H*N*N + H*q2
    end

     

    L == length(x) || error("Wrong dimension of parameter vector")
    L == length(grad) || error("Wrong dimension of gradient vector")
    
    if parallel
        W = reshape(x[1:H*N*N],H,N,N)
        V = reshape(x[H*N*N+1:end],H,q,q,N)
        pseudologlikelihood = zeros(Float64, N)
        grad .= 0.0
        Threads.@threads for site = 1:N 
                pseudologlikelihood[site] = parallel_update_grad_site!(grad,Z,W,V,site,weights,lambda)
        end
        L2 = lambda*L2Tensor(W) + lambda*L2Tensor(V)
        total_loglike = sum(pseudologlikelihood) 
    else 
        W = reshape(x[1:H*N*N],H,N,N)
        V = reshape(x[H*N*N+1:end],H,q,q)

        pseudologlikelihood = zeros(Float64, N)
        grad .= 0.0
        
        W_sf = zeros(Float64,H,N,N)
        # prob = zeros(Float64,q,M,N)
        mat = zeros(Float64,N,N,q,q)
        Threads.@threads for site = 1:N #l'upgrade è fatto male non so perchè
            pseudologlikelihood[site],mat[site,:,:,:],W_sf[:,:,site] = update_gradW_site!(grad,Z,W,V,site,weights,lambda)
        end
        
        
        update_gradV!(grad,Z,W_sf,V,lambda,mat)
        
        L2 = lambda*L2Tensor(W) + lambda*L2Tensor(V)
        total_loglike = sum(pseudologlikelihood)
    end
    if dist !== nothing 
        score = if parallel 
            parallel_compute_dcascore(W,V)
        else
            compute_dcascore(W,V)
        end
        roc = compute_referencescore(score, dist)
        rocN = roc[N][end]
        rocN5 = roc[div(N,5)][end]
        
        file !== nothing && write(file, "$total_loglike   "*"$L2   "*"$rocN5   "*"$rocN"*"\n")
        verbose && println("$total_loglike   "*"$L2   "*"$rocN5   "*"$rocN")
    else
        file !== nothing && write(file, "$total_loglike   "*"$L2"*"\n")
        verbose && println("$total_loglike   "*"$L2 ")
    end
    
    pg == pointer(grad) || error("Different pointer")
    return total_loglike + L2
end

function update_gradW_site!(grad,Z,W,V,site,weights,lambda)
    pg = pointer(grad)

    N,M = size(Z)
    H,q,q = size(V)

    L = H*N*N+H*q*q
    
    W_site = view(W,:,:,site)
    Wsf_site = softmax(W_site,dims=2)
    

    @tullio threads=true fastmath=true avx=true grad=false J[a,b,j] := Wsf_site[h,j]*V[h,a,b]*(site!=j)
    
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
    @tullio threads=true fastmath=true avx=true grad=false fact[j,h] := mat[j,a,b]*V[h,a,b]

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
    #
    return pl, mat, Wsf_site
    #
end

function update_gradV!(grad,Z,Wsf,V,lambda,mat)
    pg = pointer(grad)

    N,_ = size(Z)
    H,q,q = size(V)

    L = H*N*N + H*q*q

    grad[H*N*N+1:end] .= 0.0

    # Wsf = softmax(W,dims=2) #Ws[h,i,site]
    # @tullio threads=true fastmath=true avx=true grad=false J[a,b,j,site] := Wsf[h,j,site]*V[h,a,b]*(site!=j)

    # @tullio threads=true fastmath=true avx=true grad=false mat_ene[a,m,site] := J[a,Z[j,m],j,site]

    # partition = sumexp(mat_ene,dims=1)
    # part = view(partition,1,:,:)
    # @tullio threads=true fastmath=true avx=true grad=false prob[a,m,site] := exp(mat_ene[a,m,site])/part[m,site]
    
    # @tullio threads=true fastmath=true avx=true grad=false mat[site,j,a,b] := weights[m]*(Z[j,m]==b)*((Z[site,m]==a)-prob[a,m,site])*(site!=j) (a in 1:q, b in 1:q)
    


    @inbounds for counter = H*N*N+1:L 
        h,c,d = counter_to_index(counter,N,q,H)
        Wsf_h = view(Wsf,h,:,:)
        mat_cd = view(mat,:,:,c,d)
        @tullio threads=true fastmath=true avx=true grad=false g = Wsf_h[j,site]*mat_cd[site,j]
        grad[counter] = g        

        grad[counter] *= -1 
        grad[counter] += 2*lambda*V[h,c,d]
    end 


    pg == pointer(grad) || error("Different pointer")

    return 
end

function parallel_update_grad_site!(grad,Z,W,V,site,weights,lambda)
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


    

    @tullio threads=true fastmath=true avx=true grad=false mat[j,a,b] := weights[m]*(Z[j,m]==b)*((Z_site[m]==a)-prob[a,m]) (a in 1:q, b in 1:q)
    mat[site,:,:] .= 0.0
    @tullio threads=true fastmath=true avx=true grad=false fact[j,h] := mat[j,a,b]*V_site[h,a,b]

    grad[(site-1)*N*H+1:site*N*H] .= 0.0
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


    grad[(site-1)*H*q*q+H*N*N+1:site*H*q*q+H*N*N] .= 0.0
    @inbounds for counter = (site-1)*H*q*q+H*N*N+1:site*H*q*q+H*N*N 
        h,c,d,i = counter_to_index_V_parallel(counter,N,q,H)
        if i==site
            mat_cd = view(mat,:,c,d)
            Wsf_h = view(Wsf_site,h,:)
            # @tullio threads=true fastmath=true avx=true grad=false g = mat_cd[j]*Wsf_h[j]

            grad[counter] = dot(mat_cd,Wsf_h)
            grad[counter] *= -1 
            grad[counter] += 2*lambda*V[h,c,d,i]
        end
    end 

    pg == pointer(grad) || error("Different pointer")
    return pl


end



