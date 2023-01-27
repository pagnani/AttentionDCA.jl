function arold_attention(Z::Array{T,2},Weights::Vector{Float64};
    msample::Union{Int64, Nothing} = nothing,
    H::Int = 32,
    d::Int = 20,
    structfile::Union{String,Nothing} = nothing, 
    output::Union{String,Nothing} = nothing,
    initx0 = nothing,
    min_separation::Int=6,
    theta=:auto,
    lambda::Real=0.01,
    epsconv::Real=1.0e-5,
    maxit::Int=1000,
    verbose::Bool=true,
    permorder::Union{Symbol,Vector{Int}}=:NATURAL,
    method::Symbol=:LD_LBFGS) where T <: Integer

    all(x -> x > 0, Weights) || throw(DomainError("vector W should normalized and with all positive elements"))
    isapprox(sum(Weights), 1) || throw(DomainError("sum(W) ≠ 1. Consider normalizing the vector W"))
    N, M = size(Z)
    M = length(Weights)
    q = Int(maximum(Z))
    plmalg = PlmAlg(method, verbose, epsconv, maxit)
    plmvar = AttPlmVar(N, M, d, q, H, lambda, Z, Weights)
    arvar = ArVar(N,M,q,lambda,0.0,Z,Weights,permorder)
    dist = if structfile !== nothing 
        compute_residue_pair_dist(structfile)
    else
        nothing 
    end
    parameters, pslike = arold_minimizepl(plmalg, plmvar, dist=dist, output=output,verbose=verbose, initx0=initx0)
    
    mask = zeros(N,N,H)
    @tullio mask[i,j,h] := -10000*(j>=i) (i in 1:N, j in 1:N, h in 1:H)

    
    Q = reshape(parameters[1:H*d*N],H,d,N)
    K = reshape(parameters[H*d*N+1:2*H*d*N],H,d,N) 
    V = reshape(parameters[2*H*d*N+1:end],H,q,q)
    @tullio W[h, i, j] := Q[h,d,i]*K[h,d,j] + mask[i,j,h]
    sf = softmax(W./sqrt(d),dims=3) 
    @tullio J[i,j,a,b] := sf[h,i,j]*V[h,a,b]*(i!=1)
    
    J_reshaped = reshapetensor(J,N,q)
    p0 = computep0(plmvar)
    H = [zeros(q) for i in 1:N-1]


    return ArNet(arvar.idxperm, p0, J_reshaped,H), arvar, parameters, pslike, J
end


function arold_attention(filename::String;
    theta::Union{Symbol,Real}=:auto,
    max_gap_fraction::Real=0.9,
    remove_dups::Bool=true,
    kwds...)
time = @elapsed Weights, Z, N, M, q = ReadFasta(filename, max_gap_fraction, theta, remove_dups)
println("preprocessing took $time seconds")
arold_attention(Z, Weights; kwds...)
end

function arold_minimizepl(alg::PlmAlg, var::AttPlmVar;
    initx0 = nothing, 
    dist = nothing, 
    output::Union{Nothing, String} = nothing,
    verbose = true)

    @extract var : N H d q2 LL = 2*H*N*d + H*q2

    # LL = 2*var.H*var.N*var.d + var.H*var.q2
    x0 = if initx0 === nothing 
        rand(Float64, LL)
    else 
        initx0
    end
    pl = 0.0
    arold_attention_parameters = zeros(LL)

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
    min_objective!(opt, (x, g) -> arold_optimfunwrapperfactored(g,x, var))
    elapstime = @elapsed  (minf, minx, ret) = optimize(opt, x0)
    alg.verbose && @printf("pl = %.4f\t time = %.4f\t", minf, elapstime)
    alg.verbose && println("exit status = $ret")
    pl = minf
    arold_attention_parameters .= minx
    if output !== nothing 
        close(file)
    end
    return arold_attention_parameters, pl

end

function arold_optimfunwrapperfactored(g::Vector{Float64},x::Vector{Float64}, var::AttPlmVar)
    g === nothing && (g = zeros(Float64, length(x)))
    return arold_pl_and_grad!(g, x, var)
end


function arold_pl_and_grad!(grad::Vector{Float64}, x::Vector{Float64}, plmvar::AttPlmVar)
    @extract plmvar : N M H d q Z lambda weights = W   
    
    numpar = N*(N-1)*q*q
    λ = lambda/numpar

    L = 2*H*N*d + H*q*q 
    L == length(x) || error("Wrong dimension of parameter vector")
    L == length(grad) || error("Wrong dimension of gradient vector")

    Q = reshape(x[1:H*N*d],H,d,N)
    K = reshape(x[H*N*d+1 : 2*H*N*d],H,d,N)
    V = reshape(x[2*H*N*d+1:end],H,q,q)

    pseudologlikelihood = zeros(Float64, N)
    reg = zeros(Float64, N)

    data = OldAttComputationQuantities(N,H,q)

     
    Threads.@threads for site in 1:N 
        pseudologlikelihood[site], reg[site] = arold_update_Q_site!(grad, Z, Q, K, V, site, weights, λ, data)
    end
    
    Threads.@threads for site in 1:N 
        old_update_K_site!(grad, Q, V, site, λ, data.sf, data.J, data.fact)
    end

    old_update_V!(grad, Q, V, λ, data)
    
    regularisation = sum(reg)
    total_pslikelihood = sum(pseudologlikelihood) + regularisation
    
    
    
    println(total_pslikelihood," ",regularisation)
    return total_pslikelihood

end


function arold_update_Q_site!(grad::Vector{Float64}, Z::Array{Int64,2}, Q::Array{Float64, 3}, K::Array{Float64, 3}, V::Array{Float64, 3}, site::Int64, weights::Vector{Float64}, lambda::Float64, data::OldAttComputationQuantities)
    pg = pointer(grad)
    size(Q) == size(K) || error("Wrong dimensionality for Q and K")
    H,d,N = size(Q)
    H,q,_ = size(V)
    
    N,M = size(Z)
    # mask = UpperTriangular(fill(-10000,N,N))
    # @tullio W[h, j] := Q[h,d,$site]*K[h,d,j]
    @tullio W[h, j] := Q[h,d,$site]*K[h,d,j] #order HNd
    @tullio W[h,j] += -10000*(j>=site) 
    sf = softmax(W./sqrt(d),dims=2) 
    data.sf[:,site,:] .= sf 

    # @tullio J_site[j,a,b] := sf[h,j]*V[h,a,b]*(j<site) #order HNq^2
    if site == 1
        J_site = zeros(N,q,q)
    else
        @tullio J_site[j,a,b] := sf[h,j]*V[h,a,b]*(site!=1)
    end
    data.J[site,:,:,:] .= J_site 

    @tullio mat_ene[a,m] := data.J[$site,j,a,Z[j,m]] #order NMq
    partition = sumexp(mat_ene,dims=1) #partition function for each m ∈ 1:M 

    @tullio prob[a,m] := exp(mat_ene[a,m])/partition[m] #order Mq
    lge = log.(partition) 

    Z_site = view(Z,site,:)
    @tullio pl = weights[m]*(mat_ene[Z_site[m],m] - lge[m]) #order M
    pl *= -1


    @tullio mat[a,b,j] := weights[m]*(Z[j,m]==b)*((Z_site[m]==a)-prob[a,m]) (a in 1:q, b in 1:q) #order NMq^2
    @tullio mat[a,b,j] *= (j<site)
    # mat[:,:,site] .= 0.0 
    data.mat[site,:,:,:] .= mat

    @tullio fact[h,j] := mat[a,b,j]*V[h,a,b] #order HNq^2
    data.fact[site,:,:] .= fact 
    outersum = zeros(Float64, N)

    @inbounds for counter in (site-1)*H*d + 1 : site*H*d 
        h,y,_ = counter_to_index(counter, N, d, q, H)
        @tullio  innersum = K[$h,$y,j]*sf[$h,j] #order N
        @tullio  outersum[j] = (K[$h,$y,j]*sf[$h,j] - sf[$h,j]*innersum)/sqrt(d) #order N
        @tullio  scra = fact[$h,j]*outersum[j] #order N
        @tullio  ∇reg =  J_site[j,a,b]*V[$h,a,b]*outersum[j] #order Nq^2
        grad[counter] = -scra + 2*lambda*∇reg 
    end
    reg = lambda*L2Tensor(J_site) 
        pg == pointer(grad) || error("Different pointer")
    return pl, reg
end


function old_update_K_site!(grad::Vector{Float64}, Q::Array{Float64,3}, V::Array{Float64,3}, site::Int, lambda::Float64, sf::Array{Float64,3}, J::Array{Float64,4}, fact::Array{Float64,3}) 
    H,d,N = size(Q)
    _,q,_ = size(V)
    
    scra = zeros(Float64,N,N)
    scra1 = zeros(Float64,q,q)

    @inbounds for counter in H*N*d+(site-1)*H*d+1:H*N*d + site*H*d
        h,y,x = counter_to_index(counter, N, d, q, H) #h, lower dim, position
        @tullio scra[i,j] = Q[$h,$y,i]*(sf[$h,i,j]*(x==j) - sf[$h,i,j]*sf[$h,i,$x])/sqrt(d) #order N^2
        @tullio scra2 = scra[i,j]*fact[i,$h,j] #order N^2
        @tullio scra1[a,b] = scra[i,j]*J[i,j,a,b] #order N^2q^2
        @tullio ∇reg = scra1[a,b]*V[$h,a,b] #order q^2
        grad[counter] = - scra2 + 2*lambda*∇reg
    end
    return
end

function old_update_V!(grad::Vector{Float64}, Q::Array{Float64,3}, V::Array{Float64,3}, lambda::Float64, data::OldAttComputationQuantities)

    H,d,N = size(Q)
    H,q,_ = size(V)


    grad[2*N*d*H+1:2*N*H*d + H*q*q] .= 0.0
    scra = zeros(Float64, N)
    @inbounds for counter in 2*N*d*H+1:2*N*H*d + H*q*q
        h,a,b = counter_to_index(counter, N,d,q, H)
        @tullio scra[site] = data.mat[site,$a,$b,j]*data.sf[$h,site,j] #order N^2
        @tullio ∇reg = data.J[i,j,$a,$b]*data.sf[$h,i,j] #order N^2
        
        grad[counter] = -sum(scra) + 2*lambda*∇reg
        
    end
    return
end


function computeJtens(parameters::Vector, plmvar)
    
    @extract plmvar : N H d q
    
    mask = zeros(N,N,H)
    @tullio mask[i,j,h] := -10000*(j>=i) (i in 1:N, j in 1:N, h in 1:H)


    Q = reshape(parameters[1:H*d*N],H,d,N)
    K = reshape(parameters[H*d*N+1:2*H*d*N],H,d,N) 
    V = reshape(parameters[2*H*d*N+1:end],H,q,q)
    @tullio W[h,i, j] := Q[h,d,i]*K[h,d,j] + mask[i,j,h]
    sf = softmax(W,dims=3) 
    @tullio J[a,i,b,j] := sf[h,i,j]*V[h,a,b]*(i!=1)

    return J

end