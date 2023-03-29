function my_attentiondca(Z::Array{T,2}, Weights::Vector{Float64};
    version::Symbol = :NOFIELD,
    H::Int = 32,
    d::Int = 20, 
    dd::Real = d, 
    initx0::Union{Nothing, Vector{Float64}} = nothing,
    lambdaQ::Real = version == :NOFIELD ? 0.01 : 0.01,
    lambdaV::Real = version == :NOFIELD ? 0.01 : 0.01, 
    lambdaF::Union{Nothing,Real} = version == :NOFIELD ? nothing : 0.01,
    normalize_lambda::Bool = false,
    epsconv::Real = 1.0e-5, 
    maxit::Int = 1000, 
    verbose::Bool=true,
    method::Symbol=:LD_LBFGS
    ) where T <: Integer

    knownversions = [:NOFIELD, :FIELD]
    version ∈ knownversions || error("Version $version is unknown, only $knownversions are supported")

    all(x -> x > 0, Weights) || Error("Only positive weights are supported")
    isapprox(sum(Weights), 1) || Error("Weights are not normalized")

    version = lambdaF !== nothing ? :FIELD : :NOFIELD

    N, M = size(Z)
    q = Int(maximum(Z))

    λQ = normalize_lambda ? lambdaQ/(N*d*H) : lambdaQ
    λV = normalize_lambda ? lambdaV/(q*q*H) : lambdaV
    if lambdaF !== nothing
        λF = normalize_lambda ? lambdaF/(N*q) : lambdaF
    else 
        λF = nothing
    end

    plmalg = PlmAlg(method, verbose, epsconv, maxit)

    plmvar = λF === nothing ? myAttPlmVar(N, M, d, q, H, λQ, λV, Z, Weights, dd = dd) : myFieldAttPlmVar(N, M, d, q, H, λQ, λV, λF, Z, Weights, dd = dd)
    parameters, pslike, elapstime, numevals, ret = my_minimize_pl(plmalg, plmvar,initx0=initx0)

    Q = reshape(parameters[1:H*d*N],H,d,N)
    K = reshape(parameters[H*d*N+1:2*H*d*N],H,d,N) 
    V = reshape(parameters[2*H*d*N+1:2*H*N*d + H*q*q],H,q,q)
    F = λF !== nothing ? reshape(parameters[2*H*N*d + H*q*q + 1 : end], q,N) : nothing
    
    info = Info(λQ,λV,λF, numevals, elapstime, ret)
    
    return AttOut(Q,K,V,F,pslike), info
end

function my_attentiondca(filename::String;
    theta::Union{Symbol,Real}=:auto,
    max_gap_fraction::Real=0.9,
    remove_dups::Bool=true,
    kwds...)
    
    time = @elapsed Weights, Z, N, M, q = ReadFasta(filename, max_gap_fraction, theta, remove_dups)
    println("preprocessing took $time seconds")
    my_attentiondca(Z, Weights; kwds...)
end

function my_minimize_pl(alg::PlmAlg, var::myFieldAttPlmVar;
    initx0::Union{Nothing, Vector{Float64}} = nothing)
    
    @extract var : N H d q2 LL = 2*H*N*d + H*q2 + N*q

    x0 = initx0 === nothing ? rand(Float64, LL) : initx0

    pl = 0.0
    parameters = zeros(LL)

    opt = Opt(alg.method, length(x0))
    ftol_abs!(opt, alg.epsconv)
    xtol_rel!(opt, alg.epsconv)
    xtol_abs!(opt, alg.epsconv)
    ftol_rel!(opt, alg.epsconv)
    maxeval!(opt, alg.maxit)
    min_objective!(opt, (x, g) -> my_optimfunwrapper(g,x, var))
    elapstime = @elapsed  (minf, minx, ret) = optimize(opt, x0)
    numevals = opt.numevals
    alg.verbose && @printf("pl = %.4f\t numevals = %i\t time = %.4f\t", minf,numevals,elapstime)
    alg.verbose && println("exit status = $ret")
    pl = minf
    parameters .= minx
    
    return parameters, pl, elapstime, numevals, ret
end

function my_minimize_pl(alg::PlmAlg, var::myAttPlmVar;
    initx0::Union{Nothing, Vector{Float64}} = nothing)
    
    @extract var : N H d q2 LL = 2*H*N*d + H*q2

    x0 = initx0 === nothing ? rand(Float64, LL) : initx0 

    pl = 0.0
    parameters = zeros(LL)

    opt = Opt(alg.method, length(x0))
    ftol_abs!(opt, alg.epsconv)
    xtol_rel!(opt, alg.epsconv)
    xtol_abs!(opt, alg.epsconv)
    ftol_rel!(opt, alg.epsconv)
    maxeval!(opt, alg.maxit)
    min_objective!(opt, (x, g) -> my_optimfunwrapper(g,x,var))
    elapstime = @elapsed  (minf, minx, ret) = optimize(opt, x0)
    numevals = opt.numevals
    alg.verbose && @printf("pl = %.4f\t numevals = %i \t time = %.4f\t", minf,numevals,elapstime)
    alg.verbose && println("exit status = $ret")
    pl = minf
    parameters .= minx

    return parameters, pl, elapstime, numevals, ret
end

function my_pl_and_grad!(grad::Vector{Float64}, x::Vector{Float64}, plmvar::myFieldAttPlmVar)

    @extract plmvar : H N M d dd q Z λQ = lambdaQ λV = lambdaV λF = lambdaF weights = W delta wdelta
    L = 2*H*N*d + H*q*q + N*q
    L == length(x) || error("Wrong dimension of parameter vector")

    L == length(grad) || error("Wrong dimension of gradient vector")
    
    Q = reshape(x[1:H*N*d],H,d,N)
    K = reshape(x[H*N*d+1 : 2*H*N*d],H,d,N)
    V = reshape(x[2*H*N*d+1:2*H*N*d + H*q*q],H,q,q)
    F = reshape(x[2*H*N*d + H*q*q + 1 : end],q,N)

    pseudologlikelihood = zeros(Float64, N)

    data = AttComputationQuantities(N,H,q)

    big_scra_grad = zeros(N,H*N*d)

    grad .= 0.0
    
    Threads.@threads for site in 1:N 
        pseudologlikelihood[site], big_scra_grad[site,:]= my_update_QK_site!(grad, Z, view(Q,:,:,site), K, V, view(F,:,site),site, weights, data,view(delta,site,:,:),wdelta,dd)
    end
    
    grad[H*N*d+1 : 2*H*N*d] = sum(big_scra_grad, dims=1)
    
    my_update_V!(grad, Q, V, data)

    grad[1:H*N*d] .+= 2*λQ*x[1:H*N*d]
    grad[H*N*d+1 : 2*H*N*d] .+= 2*λQ*x[H*N*d+1 : 2*H*N*d]
    grad[2*H*N*d+1:2*H*N*d + H*q*q] .+= 2*λV*x[2*H*N*d+1:2*H*N*d + H*q*q]
    grad[2*H*N*d + H*q*q + 1 : end] .+= 2*λF*x[2*H*N*d + H*q*q + 1 : end]
    
    regularisation = λQ*L2Tensor(Q) + λQ*L2Tensor(K) + λV*L2Tensor(V) + λF*L2Tensor(F)
    total_pslikelihood = sum(pseudologlikelihood) + regularisation
    println(total_pslikelihood," ",regularisation)
    return total_pslikelihood
end

function my_pl_and_grad!(grad::Vector{Float64}, x::Vector{Float64}, plmvar::myAttPlmVar)
    
    @extract plmvar : H N M d dd q Z λQ = lambdaQ λV = lambdaV weights = W delta wdelta
    L = 2*H*N*d + H*q*q 
    L == length(x) || error("Wrong dimension of parameter vector")
    L == length(grad) || error("Wrong dimension of gradient vector")

    Q = reshape(x[1:H*N*d],H,d,N)
    K = reshape(x[H*N*d+1 : 2*H*N*d],H,d,N)
    V = reshape(x[2*H*N*d+1:end],H,q,q)

    pseudologlikelihood = zeros(Float64, N)
    
    data = AttComputationQuantities(N,H,q)

    big_scra_grad = zeros(N,H*N*d)

    grad .= 0.0
    Threads.@threads for site in 1:N 
        pseudologlikelihood[site], big_scra_grad[site,:] = my_update_QK_site!(grad, Z, view(Q,:,:,site), K, V, site, weights,data,view(delta,site,:,:),wdelta,dd)
    end
    
    grad[H*N*d+1 : 2*H*N*d] = sum(big_scra_grad, dims=1) 

    my_update_V!(grad, Q, V, data)
    
    grad[1:H*N*d] .+= 2*λQ*x[1:H*N*d]
    grad[H*N*d+1 : 2*H*N*d] .+= 2*λQ*x[H*N*d+1 : 2*H*N*d]
    grad[2*H*N*d+1:end] .+= 2*λV*x[2*H*N*d+1:end]

    regularisation = λQ*L2Tensor(Q) + λQ*L2Tensor(K) + λV*L2Tensor(V)
    total_pslikelihood = sum(pseudologlikelihood) + regularisation
    
    println(total_pslikelihood," ",regularisation)
    return total_pslikelihood
end

function my_update_QK_site!(grad::Vector{Float64}, Z::Array{Int,2}, Q::AbstractArray{Float64,2}, K::Array{Float64,3}, V::Array{Float64,3}, site::Int, weights::Vector{Float64}, data::AttComputationQuantities, delta::AbstractArray{Int, 2}, wdelta::Array{Float64,3}, dd::Real)
    
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
        h,y,i = counter_to_index(counter, N, d, q, H)
        @tullio  innersum = K[$h,$y,j]*sf[j,$h] #order N
        @tullio  outersum[j] = (K[$h,$y,j]*sf[j,$h] - sf[j,$h]*innersum)/sqrt(dd) #order N
        @tullio  scra = fact[j,$h]*outersum[j] #order N
        grad[counter] = -scra
    end
    
    scra = zeros(Float64,N)
    scra_grad = []
    @inbounds for counter in H*N*d+1:2*H*N*d
        h,y,x = counter_to_index(counter, N, d, q, H) #h, lower dim, position
        @tullio scra[j] = Q[$h,$y]*(sf[j,$h]*(x==j) - sf[j,$h]*sf[$x,$h])/sqrt(dd) #order N^2
        @tullio scra2 = scra[j]*fact[j,$h] #order N^2
        push!(scra_grad, - scra2)
    end

    return pl, scra_grad
    
end

function my_update_QK_site!(grad::Vector{Float64}, Z::Array{Int,2}, Q::AbstractArray{Float64,2}, K::Array{Float64,3}, V::Array{Float64,3}, F, site::Int, weights::Vector{Float64}, data::AttComputationQuantities, delta::AbstractArray{Int, 2}, wdelta::Array{Float64,3}, dd::Real)
    H,d = size(Q)
    H,q,_ = size(V)
    N,M = size(Z) 
    @tullio sf[j, h] := Q[h,d]*K[h,d,j]
    sf = softmax(sf./sqrt(dd),dims=1) 
    view(data.sf,:,site,:) .= sf

    @tullio J_site[j,a,b] := sf[j,h]*V[h,a,b]
    view(J_site,site,:,:) .= 0.0
    view(data.J,site,:,:,:) .= J_site 

    Z_site = view(Z,site,:)

    @tullio _mat_ene[a,m] := J_site[j,a,Z[j,m]]
    @tullio mat_ene[a,m] := _mat_ene[a,m] + F[a]
    partition = sumexp(mat_ene,dims=1) #partition function for each m ∈ 1:M 

    @tullio probnew[m,a] := delta[m,a] - exp(mat_ene[a,m])/partition[m]
    lge = log.(partition) 

     
    @tullio pl = weights[m]*(mat_ene[Z_site[m],m] - lge[m]) #order M
    pl *= -1

    @tullio mat[a,b,j] := wdelta[j,m,b]*probnew[m,a] (a in 1:q, b in 1:q)
    view(mat,:,:,site) .= 0.0 
    view(data.mat,site,:,:,:) .= mat

    @tullio fact[j,h] := mat[a,b,j]*V[h,a,b] #order HNq^2
    outersum = zeros(Float64, N)

    @inbounds for counter in (site-1)*H*d + 1 : site*H*d 
        h,y,i = counter_to_index(counter, N, d, q, H)
        @tullio  innersum = K[$h,$y,j]*sf[j,$h] #order N
        @tullio  outersum[j] = (K[$h,$y,j]*sf[j,$h] - sf[j,$h]*innersum)/sqrt(dd) #order N
        @tullio  scra = fact[j,$h]*outersum[j] #order N
        grad[counter] = -scra
    end
    
    scra = zeros(Float64,N)
    scra_grad = []
    @inbounds for counter in H*N*d+1:2*H*N*d
        h,y,x = counter_to_index(counter, N, d, q, H) #h, lower dim, position
        @tullio scra[j] = Q[$h,$y]*(sf[j,$h]*(x==j) - sf[j,$h]*sf[$x,$h])/sqrt(dd) #order N^2
        @tullio scra2 = scra[j]*fact[j,$h] #order N^2
        push!(scra_grad, - scra2)
    end   

    
    # @tullio ∇field[l] := -weights[m]*probnew[m,l]
    ∇field = zeros(Float64,q)
    for l in 1:q 
        for m in 1:M 
            ∇field[l] += -weights[m]*probnew[m,l]
        end
    end


    grad[2*H*N*d + H*q*q + q*(site-1) + 1 : 2*H*N*d + H*q*q + q*site] .= ∇field


    return pl, scra_grad

end

function my_update_V!(grad::Vector{Float64}, Q::Array{Float64,3}, V::Array{Float64,3}, data::AttComputationQuantities)
    
    H,d,N = size(Q)
    H,q,_ = size(V)


    grad[2*N*d*H+1:2*N*H*d + H*q*q] .= 0.0
    scra = zeros(Float64, N)
    @inbounds for counter in 2*N*d*H+1:2*N*H*d + H*q*q
        h,a,b = counter_to_index(counter, N,d,q, H)
        @tullio scra[site] = data.mat[site,$a,$b,j]*data.sf[j,site,$h] #order N^2
        
        grad[counter] = -sum(scra)
        
    end
    return
end

function my_optimfunwrapper(g::Vector,x::Vector, var::Union{myAttPlmVar,myFieldAttPlmVar})
    g === nothing && (g = zeros(Float64, length(x)))
    return my_pl_and_grad!(g, x, var)
end
