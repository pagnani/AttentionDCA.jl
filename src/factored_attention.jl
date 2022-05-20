function factored_attention_plmdca(Z::Array{T,2},Weights::Vector{Float64};
    H::Int = 32,
    d::Int = 10,
    structfile::Union{String,Nothing} = nothing, 
    output::Union{String,Nothing} = nothing,
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
    plmalg = PlmAlg(method, verbose, epsconv, maxit)
    plmvar = FAPlmVar(N, M, d, q, q*q, H, lambda, Z, Weights)
    dist = if structfile !== nothing 
        compute_residue_pair_dist(structfile)
    else
        nothing 
    end
    parameters, pslike = attentionMinimizePL(plmalg, plmvar, dist=dist, output=output,verbose=verbose, initx0=initx0)
    Q = reshape(parameters[1:H*d*N],H,d,N)
    K = reshape(parameters[H*d*N+1:2*H*d*N],H,d,N) 
    V = reshape(parameters[2*H*d*N+1:end],H,q,q)
    score = compute_dcascore_fa(Q, K, V)
    roc = if structfile !== nothing
        map(x->x[4],compute_referencescore(score, compute_residue_pair_dist(structfile)))
    else
        nothing
    end
    return FAPlmOut(pslike, Q, K, V, score,roc)
end

function factored_attention_plmdca(filename::String;
    theta::Union{Symbol,Real}=:auto,
    max_gap_fraction::Real=0.9,
    remove_dups::Bool=true,
    kwds...)
time = @elapsed Weights, Z, N, M, q = ReadFasta(filename, max_gap_fraction, theta, remove_dups)
println("preprocessing took $time seconds")
factored_attention_plmdca(Z, Weights; kwds...)
end

function attentionMinimizePL(alg::PlmAlg, var::FAPlmVar;
    initx0 = nothing, 
    dist = nothing, 
    output::Union{Nothing, String} = nothing,
    verbose = true)
    LL = 2*var.H*var.N*var.d + var.H*var.q2
    x0 = if initx0 === nothing 
        rand(Float64, LL)*0.0001
    else 
        initx0
    end
    pl = 0.0
    factored_attention_parameters = zeros(LL)

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
    min_objective!(opt, (x, g) -> optimfunwrapperfactored(g,x, var))
    elapstime = @elapsed  (minf, minx, ret) = optimize(opt, x0)
    alg.verbose && @printf("pl = %.4f\t time = %.4f\t", minf, elapstime)
    alg.verbose && println("exit status = $ret")
    pl = minf
    factored_attention_parameters .= minx
    if output !== nothing 
        close(file)
    end
    return factored_attention_parameters, pl

end




function fa_pl_and_grad!(grad, x, plmvar)
    pg = pointer(grad)
    H = plmvar.H 
    N = plmvar.N
    M = plmvar.M
    d = plmvar.d
    q = plmvar.q
    Z = plmvar.Z
    λ = N*q*(plmvar.lambda)/M
    weights = plmvar.W
    
    
    L = 2*H*N*d + H*q*q 
    L == length(x) || error("Wrong dimension of parameter vector")
    L == length(grad) || error("Wrong dimension of gradient vector")

    Q = reshape(x[1:H*N*d],H,d,N)
    K = reshape(x[H*N*d+1 : 2*H*N*d],H,d,N)
    V = reshape(x[2*H*N*d+1:end],H,q,q)

    pseudologlikelihood = zeros(Float64, N)
    reg = zeros(Float64, N)

    data = FAComputationQuantities(N,H,q)

     
    Threads.@threads for site in 1:N 
        pseudologlikelihood[site], reg[site] = update_Q_site!(grad, Z, Q, K, V, site, weights,λ,data)
    end
    
    Threads.@threads for site in 1:N 
        update_K_site!(grad, Q, V, site, λ, data)
    end

    update_V!(grad, Q, V, λ, data)
    
    regularisation = sum(reg)
    total_pslikelihood = sum(pseudologlikelihood) + regularisation
    
    
    
    pg == pointer(grad) || error("Different pointer")
    println(total_pslikelihood," ",regularisation)
    return total_pslikelihood

end


function update_Q_site!(grad, Z, Q, K, V, site, weights, lambda, data)
    pg = pointer(grad)
    size(Q) == size(K) || error("Wrong dimensionality for Q and K")
    H,d,N = size(Q)
    H,q,_ = size(V)
    

    @tullio W[h, j] := Q[h,d,$site]*K[h,d,j]
    sf = softmax(W,dims=2)
    data.sf[:,site,:] = softmax(W,dims=2)

    @tullio J_site[j,a,b] := sf[h,j]*V[h,a,b]*(site!=j)
    data.J[site,:,:,:] = J_site

    @tullio mat_ene[a,m] := data.J[$site,j,a,Z[j,m]] 
    partition = sumexp(mat_ene,dims=1) #partition function for each m ∈ 1:M

    @tullio prob[a,m] := exp(mat_ene[a,m])/partition[m]
    lge = log.(partition)

    Z_site = view(Z,site,:)
    @tullio pl = weights[m]*(mat_ene[Z_site[m],m] - lge[m])
    pl *= -1



    @tullio mat[a,b,j] := weights[m]*(Z[j,m]==b)*((Z_site[m]==a)-prob[a,m]) (a in 1:q, b in 1:q)
    mat[:,:,site] .= 0.0
    data.mat[site,:,:,:] = mat

    @tullio fact[h,j] := mat[a,b,j]*V[h,a,b]
    data.fact[site,:,:] = fact

    for counter in (site-1)*H*d + 1 : site*H*d
        h,y,x = new_counter_to_index(counter, N, d, q, H)
        @tullio innersum = K[$h,$y,j]*sf[$h,j]
        @tullio outersum[j] := (K[$h,$y,j]*sf[$h,j] - sf[$h,j]*innersum) 
        @tullio scra := fact[$h,j]*outersum[j]
        @tullio ∇reg :=  J_site[j,a,b]*V[$h,a,b]*outersum[j]
        grad[counter] = -scra + 2*lambda*∇reg 
    end
    
    pg == pointer(grad) || error("Different pointer")
    reg = lambda*L2Tensor(J_site)
    return pl, reg

end

function update_K_site!(grad, Q, V, site, lambda, data) #sf HNN, fact NHN
    pg = pointer(grad)

    H,d,N = size(Q)
    H,q,_ = size(V)


    for counter in H*N*d+(site-1)*H*d+1:H*N*d + site*H*d
        h,y,x = new_counter_to_index(counter, N, d, q, H) #h, lower dim, position
        @tullio scra[i,j] := Q[$h,$y,i]*(data.sf[$h,i,j]*(x==j) - data.sf[$h,i,j]*data.sf[$h,i,$x])
        @tullio scra2 := scra[i,j]*data.fact[i,$h,j]
        @tullio scra1[a,b] := scra[i,j]*data.J[i,j,a,b]
        @tullio ∇reg := scra1[a,b]*V[$h,a,b]
        grad[counter] = - scra2 + 2*lambda*∇reg
    end
    pg == pointer(grad) || error("Different pointer")
    return
end

function update_V!(grad, Q, V, lambda, data)

    pg = pointer(grad)

    H,d,N = size(Q)
    H,q,_ = size(V)


    grad[2*N*d*H+1:2*N*H*d + H*q*q] .= 0.0
    
    for counter in 2*N*d*H+1:2*N*H*d + H*q*q
        h,a,b = new_counter_to_index(counter, N,d,q, H)
        
        @tullio scra[site] := data.mat[site,$a,$b,j]*data.sf[$h,site,j]
        @tullio ∇reg = data.J[i,j,$a,$b]*data.sf[$h,i,j]
        
        grad[counter] = -sum(scra) + 2*lambda*∇reg
        
    end
    pg == pointer(grad) || error("Different pointer")
    return
end


function new_counter_to_index(l::Int, N::Int, d:: Int, Q::Int, H::Int; verbose::Bool=false)
    h::Int = 0
    if l <= H*N*d
        i::Int = ceil(l/(d*H))
        l = l-(d*H)*(i-1)
        m::Int = ceil(l/H)
        h = l-H*(m-1)
        verbose && println("h = $h \nm = $m \ni = $i")
        return h,m,i
    elseif H*N*d < l <= 2*H*N*d 
        l-=d*N*H
        j::Int = ceil(l/(d*H))
        l-=(d*H)*(j-1)
        n::Int = ceil(l/H)
        h = l-H*(n-1)
        verbose && println("h = $h \nn = $n \nj = $j")
        return h, n, j
    else
        l-=2*N*d*H
        b::Int = ceil(l/(Q*H))
        l-=(Q*H)*(b-1)
        a::Int = ceil(l/H)
        h = l-H*(a-1)
        verbose && println("h = $h \na = $a \nb = $b \n")
        return h, a, b
    end
end

