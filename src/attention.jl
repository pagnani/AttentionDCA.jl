function attdca(Z::Array{T,2},Weights::Vector{Float64};
    H::Int = 32,
    d::Int = 20,
    dd = d,
    normalise_lambda = true,
    initx0 = nothing,
    min_separation::Int=6,
    theta=:auto,
    lambda::Real=0.01,
    epsconv::Real=1.0e-5,
    maxit::Int=1000,
    verbose::Bool=true,
    method::Symbol=:LD_LBFGS) where T <: Integer

    all(x -> x > 0, Weights) || throw(DomainError("vector W should normalized and with all positive elements"))
    isapprox(sum(Weights), 1) || throw(DomainError("sum(W) ≠ 1. Consider normalizing the vector W"))
    
    

    N, M = size(Z)
    M = length(Weights)
    q = Int(maximum(Z))
    
    λ = if normalise_lambda
        lambda/(N*(N-1)*q*q)
    else
        lambda 
    end
    
    plmalg = PlmAlg(method, verbose, epsconv, maxit)
    plmvar = AttPlmVar(N, M, d, q, H, λ, Z, Weights, dd = dd) 
    
    parameters, pslike, elapstime, numevals, ret = minimize_pl(plmalg, plmvar,initx0=initx0)
    Q = reshape(parameters[1:H*d*N],H,d,N)
    K = reshape(parameters[H*d*N+1:2*H*d*N],H,d,N) 
    V = reshape(parameters[2*H*d*N+1:end],H,q,q)
    return AttPlmOut(Q, K, V, pslike), elapstime, numevals, ret

end

function attdca(filename::String;
    theta::Union{Symbol,Real}=:auto,
    max_gap_fraction::Real=0.9,
    remove_dups::Bool=true,
    kwds...)
    
    time = @elapsed Weights, Z, N, M, q = ReadFasta(filename, max_gap_fraction, theta, remove_dups)
    println("preprocessing took $time seconds")
    attdca(Z, Weights; kwds...)
end

function minimize_pl(alg::PlmAlg, var::AttPlmVar;
    initx0::Union{Nothing, Vector{Float64}} = nothing)
    
    @extract var : N H d q2 LL = 2*H*N*d + H*q2

    x0 = if initx0 === nothing 
        rand(Float64, LL)
    else 
        initx0
    end
    pl = 0.0
    parameters = zeros(LL)

    opt = Opt(alg.method, length(x0))
    ftol_abs!(opt, alg.epsconv)
    xtol_rel!(opt, alg.epsconv)
    xtol_abs!(opt, alg.epsconv)
    ftol_rel!(opt, alg.epsconv)
    maxeval!(opt, alg.maxit)
    min_objective!(opt, (x, g) -> optimfunwrapper(g,x,var))
    elapstime = @elapsed  (minf, minx, ret) = optimize(opt, x0)
    numevals = opt.numevals
    alg.verbose && @printf("pl = %.4f\t time = %.4f\t", minf, elapstime)
    alg.verbose && println("exit status = $ret")
    pl = minf
    parameters .= minx

    return parameters, pl, elapstime, numevals, ret
end

function pl_and_grad!(grad::Vector{Float64}, x::Vector{Float64}, plmvar::AttPlmVar)
    
    @extract plmvar : H N M d dd q Z λ = lambda weights = W delta wdelta

    L = 2*H*N*d + H*q*q 
    L == length(x) || error("Wrong dimension of parameter vector")
    L == length(grad) || error("Wrong dimension of gradient vector")

    Q = reshape(x[1:H*N*d],H,d,N)
    K = reshape(x[H*N*d+1 : 2*H*N*d],H,d,N)
    V = reshape(x[2*H*N*d+1:end],H,q,q)

    pseudologlikelihood = zeros(Float64, N)
    reg = zeros(Float64, N)

    data = AttComputationQuantities(N,H,q)

    big_scra_grad = zeros(N,H*N*d)

    grad .= 0.0
     
    Threads.@threads for site in 1:N 
        pseudologlikelihood[site], reg[site], big_scra_grad[site,:] = update_QK_site!(grad, Z, view(Q,:,:,site), K, V, site, weights, λ, data,view(delta,site,:,:),wdelta,dd)
    end
    
    grad[H*N*d+1 : 2*H*N*d] = sum(big_scra_grad, dims=1) 

    update_V!(grad, Q, V, λ, data)
    
    regularisation = sum(reg)
    total_pslikelihood = sum(pseudologlikelihood) + regularisation
    
    println(total_pslikelihood," ",regularisation)
    return total_pslikelihood
end

function update_QK_site!(grad::Vector{Float64}, Z::Array{Int,2}, Q::AbstractArray{Float64,2}, K::Array{Float64,3}, V::Array{Float64,3}, site::Int, weights::Vector{Float64}, lambda::Float64, data::AttComputationQuantities, delta, wdelta, dd)
    H,d = size(Q)
    H,q,_ = size(V)
    N,_ = size(Z) 
    @tullio sf[j, h] := Q[h,d]*K[h,d,j]
    sf = softmax(sf./sqrt(dd),dims=1) 
    view(data.sf,:,site,:) .= sf

    @tullio J_site[j,a,b] := sf[j,h]*V[h,a,b]
    view(J_site,site,:,:) .= 0.0
    view(data.J,site,:,:,:) .= J_site 

    @tullio mat_ene[a,m] := J_site[j,a,Z[j,m]] #order NMq
    partition = sumexp(mat_ene,dims=1) #partition function for each m ∈ 1:M 

    @tullio probnew[m,a] := delta[m,a] - exp(mat_ene[a,m])/partition[m]
    lge = log.(partition) 

    Z_site = view(Z,site,:) 
    @tullio pl = weights[m]*(mat_ene[Z_site[m],m] - lge[m]) #order M
    pl *= -1

    @tullio mat[a,b,j] := wdelta[j,m,b]*probnew[m,a] (a in 1:q, b in 1:q)
    view(mat,:,:,site) .= 0.0 
    view(data.mat,site,:,:,:) .= mat

    @tullio fact[j,h] := mat[a,b,j]*V[h,a,b] #order HNq^2
    outersum = zeros(Float64, N)

    @inbounds for counter in (site-1)*H*d + 1 : site*H*d 
        h,y,_ = counter_to_index(counter, N, d, q, H)
        @tullio  innersum = K[$h,$y,j]*sf[j,$h] #order N
        @tullio  outersum[j] = (K[$h,$y,j]*sf[j,$h] - sf[j,$h]*innersum)/sqrt(dd) #order N
        @tullio  scra = fact[j,$h]*outersum[j] #order N
        @tullio  ∇reg =  J_site[j,a,b]*V[$h,a,b]*outersum[j] #order Nq^2
        grad[counter] = -scra + 2*lambda*∇reg 
    end
    reg = lambda*L2Tensor(J_site)

    
    scra = zeros(Float64,N)
    scra1 = zeros(Float64,q,q)
    scra_grad = []
    @inbounds for counter in H*N*d+1:2*H*N*d
        h,y,x = counter_to_index(counter, N, d, q, H) #h, lower dim, position
        @tullio scra[j] = Q[$h,$y]*(sf[j,$h]*(x==j) - sf[j,$h]*sf[$x,$h])/sqrt(dd) #order N^2
        @tullio scra2 = scra[j]*fact[j,$h] #order N^2
        @tullio scra1[a,b] = scra[j]*J_site[j,a,b] #order N^2q^2
        @tullio ∇reg = scra1[a,b]*V[$h,a,b] #order q^2
        push!(scra_grad, - scra2 + 2*lambda*∇reg)
    end


    return pl, reg, scra_grad

end


function update_V!(grad::Vector{Float64}, Q::Array{Float64,3}, V::Array{Float64,3}, lambda::Float64, data::AttComputationQuantities)

    H,d,N = size(Q)
    H,q,_ = size(V)


    grad[2*N*d*H+1:2*N*H*d + H*q*q] .= 0.0
    scra = zeros(Float64, N)
    @inbounds for counter in 2*N*d*H+1:2*N*H*d + H*q*q
        h,a,b = counter_to_index(counter, N,d,q, H)
        @tullio scra[site] = data.mat[site,$a,$b,j]*data.sf[j,site,$h] #order N^2
        @tullio ∇reg = data.J[i,j,$a,$b]*data.sf[j,i,$h] #order N^2
        
        grad[counter] = -sum(scra) + 2*lambda*∇reg
        
    end
    return
end