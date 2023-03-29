#Forse dovrei non imparare il primo field, metterlo a zero sempre e poi quando creo H da mettere 
#dentro arnet scarto il primo

function arattentiondca(Z::Array{T,2},Weights::Vector{Float64};
    version::Symbol = :NOFIELD,
    H::Int = 32,
    d::Int = 20,
    dd::Real = d, 
    initx0::Union{Nothing, Vector{Float64}} = nothing,
    lambdaJ::Real=0.01,
    lambdaF::Union{Nothing,Real} = version == :NOFIELD ? nothing : 0.01,
    normalize_lambda::Bool = false,
    epsconv::Real=1.0e-5,
    maxit::Int=1000,
    verbose::Bool=true,
    permorder::Union{Symbol,Vector{Int}}=:NATURAL,
    method::Symbol=:LD_LBFGS) where T <: Integer

    knownversions = [:NOFIELD, :FIELD]
    version ∈ knownversions || error("Version $version is unknown, only $knownversions are supported")

    all(x -> x > 0, Weights) || error("vector W should normalized and with all positive elements")
    isapprox(sum(Weights), 1) || error("sum(W) ≠ 1. Consider normalizing the vector W")
    
    version = lambdaF !== nothing ? :FIELD : :NOFIELD

    N, M = size(Z)
    q = Int(maximum(Z))

    λJ = normalize_lambda ? lambdaJ/(N*(N-1)*q*q) : lambdaJ
    if lambdaF !== nothing
        λF = normalize_lambda ? lambdaF/(N*q) : lambdaF
    else 
        λF = nothing
    end

    plmalg = PlmAlg(method, verbose, epsconv, maxit)
    
    plmvar = λF === nothing ? AttPlmVar(N, M, d, q, H, λJ, Z, Weights, dd = dd) : FieldAttPlmVar(N, M, d, q, H, λJ, λF, Z, Weights, dd = dd)
    arvar = λF === nothing ? ArVar(N,M,q,lambdaJ,0.0,Z,Weights,permorder) : ArVar(N,M,q,lambdaJ,lambdaF,Z,Weights,permorder)

    parameters, pslike = ar_minimizepl(plmalg, plmvar, initx0=initx0)

    Q = reshape(parameters[1:H*d*N],H,d,N)
    K = reshape(parameters[H*d*N+1:2*H*d*N],H,d,N) 
    V = reshape(parameters[2*H*d*N+1:2*H*N*d + H*q*q],H,q,q)
    F = λF !== nothing ? reshape(parameters[2*H*N*d + H*q*q + 1 : end], q,N) : nothing

    @tullio W[h, i, j] := Q[h,d,i]*K[h,d,j]
    sf = softmax(W./sqrt(dd),dims=3) 
    @tullio J[i,j,a,b] := sf[h,i,j]*V[h,a,b]*(j<i)
    
    J_reshaped = reshapetensor(J,N,q)
    p0 = computep0(plmvar)
    H = λF !== nothing ? [F[:,i] for i in 2:N] : [zeros(q) for _ in 1:N-1]

    return ArNet(arvar.idxperm, p0, J_reshaped,H), arvar, AttOut(Q,K,V,F,pslike)

end

function arattentiondca(filename::String;
    theta::Union{Symbol,Real}=:auto,
    max_gap_fraction::Real=0.9,
    remove_dups::Bool=true,
    kwds...)

    time = @elapsed Weights, Z, N, M, q = ReadFasta(filename, max_gap_fraction, theta, remove_dups)
    println("preprocessing took $time seconds")
    arattentiondca(Z, Weights; kwds...)
end

function ar_minimizepl(alg::PlmAlg, var::AttPlmVar;
    initx0::Union{Nothing, Vector{Float64}} = nothing)

    @extract var : N H d q2 LL = 2*H*N*d + H*q2

    x0 = initx0 === nothing ? rand(Float64, LL) : initx0
    
    ar_attention_parameters = zeros(LL)

    opt = Opt(alg.method, LL)
    ftol_abs!(opt, alg.epsconv)
    xtol_rel!(opt, alg.epsconv)
    xtol_abs!(opt, alg.epsconv)
    ftol_rel!(opt, alg.epsconv)
    maxeval!(opt, alg.maxit)
    
    min_objective!(opt, (x, g) -> ar_optimfunwrapperfactored(g,x, var))
    elapstime = @elapsed  (minf, minx, ret) = optimize(opt, x0)
    alg.verbose && @printf("pl = %.4f\t time = %.4f\t", minf, elapstime)
    alg.verbose && println("exit status = $ret")
    
    pl = minf
    ar_attention_parameters .= minx
    
    return ar_attention_parameters, pl

end

function ar_minimizepl(alg::PlmAlg, var::FieldAttPlmVar;
    initx0::Union{Nothing, Vector{Float64}} = nothing)
    @extract var : N H d q2 LL = 2*H*N*d + H*q2 + N*q 

    x0 = initx0 === nothing ? rand(Float64, LL) : initx0
    
    ar_attention_parameters = zeros(LL)

    opt = Opt(alg.method, LL)
    ftol_abs!(opt, alg.epsconv)
    xtol_rel!(opt, alg.epsconv)
    xtol_abs!(opt, alg.epsconv)
    ftol_rel!(opt, alg.epsconv)
    maxeval!(opt, alg.maxit)
    
    min_objective!(opt, (x, g) -> ar_optimfunwrapperfactored(g,x, var))
    elapstime = @elapsed  (minf, minx, ret) = optimize(opt, x0)
    alg.verbose && @printf("pl = %.4f\t time = %.4f\t", minf, elapstime)
    alg.verbose && println("exit status = $ret")
    
    pl = minf
    ar_attention_parameters .= minx
    
    return ar_attention_parameters, pl

end

function ar_pl_and_grad!(grad::Vector{Float64}, x::Vector{Float64}, plmvar::FieldAttPlmVar)
    @extract plmvar : N M H d dd q Z λJ = lambdaJ λF = lambdaF weights = W delta wdelta
    L = 2*H*N*d + H*q*q + N*q
    L == length(x) || error("Wrong dimension of parameter vector")
    L == length(grad) || error("Wrong dimension of gradient vector")

    Q = reshape(x[1:H*N*d],H,d,N)
    K = reshape(x[H*N*d+1 : 2*H*N*d],H,d,N)
    V = reshape(x[2*H*N*d+1:2*H*N*d + H*q*q],H,q,q)
    F = reshape(x[2*H*N*d + H*q*q + 1 : end],q,N)

    pseudologlikelihood = zeros(Float64, N)
    reg = zeros(Float64, N)

    data = AttComputationQuantities(N,H,q)

    big_scra_grad = zeros(N,H*N*d)
     
    grad .= 0.0

    Threads.@threads for site in 1:N 
        pseudologlikelihood[site], reg[site], big_scra_grad[site,:] = ar_update_QK_site!(grad, Z, view(Q,:,:,site), K, V, view(F,:,site), site, weights, λJ, λF, data,view(delta,site,:,:),wdelta,dd)
    end
    grad[H*N*d+1 : 2*H*N*d] = sum(big_scra_grad, dims=1)
    grad[2*H*N*d + H*q*q + 1:end] .+= 2*λF*x[2*H*N*d + H*q*q + 1 : end] 

    ar_update_V!(grad, Q, V, λJ, data)
    
    regularisation = sum(reg)
    total_pslikelihood = sum(pseudologlikelihood) + regularisation
    
    println(total_pslikelihood," ",regularisation)
    return total_pslikelihood

end

function ar_pl_and_grad!(grad::Vector{Float64}, x::Vector{Float64}, plmvar::AttPlmVar)
    @extract plmvar : N M H d dd q Z λ = lambda weights = W delta wdelta
 
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
     
    Threads.@threads for site in 1:N 
        pseudologlikelihood[site], reg[site], big_scra_grad[site,:] = ar_update_QK_site!(grad, Z, view(Q,:,:,site), K, V, site, weights, λ, data, view(delta,site,:,:),wdelta,dd)
    end

    grad[H*N*d+1 : 2*H*N*d] = sum(big_scra_grad, dims=1)

    ar_update_V!(grad, Q, V, λ, data)
    
    regularisation = sum(reg)
    total_pslikelihood = sum(pseudologlikelihood) + regularisation
    
    println(total_pslikelihood," ",regularisation)
    return total_pslikelihood

end

function ar_update_QK_site!(grad::Vector{Float64}, Z::Array{Int,2}, Q::AbstractArray{Float64,2}, K::Array{Float64,3}, V::Array{Float64,3}, site::Int, weights::Vector{Float64}, lambda::Float64, data::AttComputationQuantities, delta::AbstractArray{Int,2}, wdelta::Array{Float64, 3}, dd::Real)
    H,d = size(Q)
    H,q,_ = size(V)
    N,_ = size(Z) 
    @tullio sf[j,h] := Q[h,d]*K[h,d,j]
    sf = softmax(sf./sqrt(dd),dims=1) 
    view(data.sf,site,:,:) .= sf

    @tullio J_site[j,a,b] := sf[j,h]*V[h,a,b]*(j<site)
    data.J[site,:,:,:] .= J_site 

    @tullio mat_ene[a,m] := J_site[j,a,Z[j,m]] 
    partition = sumexp(mat_ene,dims=1) 

    @tullio probnew[m,a] := delta[m,a] - exp(mat_ene[a,m])/partition[m]
    lge = log.(partition) 

    Z_site = view(Z,site,:) 
    @tullio pl = weights[m]*(mat_ene[Z_site[m],m] - lge[m]) 
    pl *= -1

    @tullio mat[a,b,j] := wdelta[j,m,b]*probnew[m,a] (a in 1:q, b in 1:q)
    @tullio mat[a,b,j] *= (j<site) 
    data.mat[site,:,:,:] .= mat

    @tullio fact[j,h] := mat[a,b,j]*V[h,a,b]
    outersum = zeros(Float64, N)

    @inbounds for counter in (site-1)*H*d + 1 : site*H*d 
        h,y,_ = counter_to_index(counter, N, d, q, H)
        @tullio  innersum = K[$h,$y,j]*sf[j,$h] 
        @tullio  outersum[j] = (K[$h,$y,j]*sf[j,$h] - sf[j,$h]*innersum)/sqrt(dd) 
        @tullio  scra = fact[j,$h]*outersum[j] 
        @tullio  ∇reg =  J_site[j,a,b]*V[$h,a,b]*outersum[j] 
        grad[counter] = -scra + 2*lambda*∇reg 
    end
    reg = lambda*L2Tensor(J_site)

    
    scra = zeros(Float64,N)
    scra1 = zeros(Float64,q,q)
    scra_grad = []
    @inbounds for counter in H*N*d+1:2*H*N*d
        h,y,x = counter_to_index(counter, N, d, q, H) #h, lower dim, position
        @tullio scra[j] = Q[$h,$y]*(sf[j,$h]*(x==j) - sf[j,$h]*sf[$x,$h])/sqrt(dd) 
        @tullio scra2 = scra[j]*fact[j,$h] 
        @tullio scra1[a,b] = scra[j]*J_site[j,a,b] 
        @tullio ∇reg = scra1[a,b]*V[$h,a,b]
        push!(scra_grad, - scra2 + 2*lambda*∇reg)
    end


    return pl, reg, scra_grad
     

end

function ar_update_QK_site!(grad::Vector{Float64}, Z::Array{Int,2}, Q::AbstractArray{Float64,2}, K::Array{Float64,3}, V::Array{Float64,3}, F::AbstractArray{Float64,1}, site::Int, weights::Vector{Float64}, lambdaJ::Float64, lambdaF::Float64, data::AttComputationQuantities, delta::AbstractArray{Int, 2}, wdelta::Array{Float64,3}, dd::Real)
    H,d = size(Q)
    H,q,_ = size(V)
    N,M = size(Z) 
    @tullio sf[j,h] := Q[h,d]*K[h,d,j]
    sf = softmax(sf./sqrt(dd),dims=1) 
    view(data.sf,site,:,:) .= sf

    @tullio J_site[j,a,b] := sf[j,h]*V[h,a,b]*(j<site)
    data.J[site,:,:,:] .= J_site 

    @tullio _mat_ene[a,m] := J_site[j,a,Z[j,m]]
    @tullio mat_ene[a,m] := _mat_ene[a,m] + F[a]
    partition = sumexp(mat_ene,dims=1) 

    @tullio probnew[m,a] := delta[m,a] - exp(mat_ene[a,m])/partition[m]
    lge = log.(partition) 

    Z_site = view(Z,site,:) 
    @tullio pl = weights[m]*(mat_ene[Z_site[m],m] - lge[m]) 
    pl *= -1

    @tullio mat[a,b,j] := wdelta[j,m,b]*probnew[m,a] (a in 1:q, b in 1:q)
    @tullio mat[a,b,j] *= (j<site) 
    data.mat[site,:,:,:] .= mat

    @tullio fact[j,h] := mat[a,b,j]*V[h,a,b] 
    outersum = zeros(Float64, N)

    @inbounds for counter in (site-1)*H*d + 1 : site*H*d 
        h,y,_ = counter_to_index(counter, N, d, q, H)
        @tullio  innersum = K[$h,$y,j]*sf[j,$h] 
        @tullio  outersum[j] = (K[$h,$y,j]*sf[j,$h] - sf[j,$h]*innersum)/sqrt(dd) 
        @tullio  scra = fact[j,$h]*outersum[j]
        @tullio  ∇reg =  J_site[j,a,b]*V[$h,a,b]*outersum[j]
        grad[counter] = -scra + 2*lambdaJ*∇reg 
    end
    reg = lambdaJ*L2Tensor(J_site) + lambdaF*L2Tensor(F)

    scra = zeros(Float64,N)
    scra1 = zeros(Float64,q,q)
    scra_grad = []
    @inbounds for counter in H*N*d+1:2*H*N*d
        h,y,x = counter_to_index(counter, N, d, q, H) #h, lower dim, position
        @tullio scra[j] = Q[$h,$y]*(sf[j,$h]*(x==j) - sf[j,$h]*sf[$x,$h])/sqrt(dd) 
        @tullio scra2 = scra[j]*fact[j,$h] 
        @tullio scra1[a,b] = scra[j]*J_site[j,a,b] 
        @tullio ∇reg = scra1[a,b]*V[$h,a,b] 
        push!(scra_grad, - scra2 + 2*lambdaJ*∇reg)
    end

    ∇field = zeros(Float64,q)
    for l in 1:q 
        for m in 1:M 
            ∇field[l] += -weights[m]*probnew[m,l]
        end
    end

    grad[2*H*N*d + H*q*q + q*(site-1) + 1 : 2*H*N*d + H*q*q + q*site] .= ∇field


    return pl, reg, scra_grad
     

end

function ar_update_V!(grad::Vector{Float64}, Q::Array{Float64,3}, V::Array{Float64,3}, lambda::Float64, data::AttComputationQuantities)

    H,d,N = size(Q)
    H,q,_ = size(V)


    grad[2*N*d*H+1:2*N*H*d + H*q*q] .= 0.0
    scra = zeros(Float64, N)
    @inbounds for counter in 2*N*d*H+1:2*N*H*d + H*q*q
        h,a,b = counter_to_index(counter, N,d,q, H)
        @tullio scra[site] = data.mat[site,$a,$b,j]*data.sf[site,j,$h] 
        @tullio ∇reg = data.J[i,j,$a,$b]*data.sf[i,j,$h,] 
        
        grad[counter] = -sum(scra) + 2*lambda*∇reg
        
    end
    return
end