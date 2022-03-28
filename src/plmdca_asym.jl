function plmdca_asym(Z::Array{T,2},W::Vector{Float64}, H::Int;
                decimation::Bool=false,
                fracmax::Real=0.3,
                fracdec::Real=0.1,
                remove_dups::Bool=true,
                min_separation::Int=1,
                theta=:auto,
                lambdaJ::Real=0.01,
                epsconv::Real=1.0e-5,
                maxit::Int=1000,
                verbose::Bool=true,
                method::Symbol=:LD_LBFGS) where T <: Integer

    all(x -> x > 0, W) || throw(DomainError("vector W should normalized and with all positive elements"))
    isapprox(sum(W), 1) || throw(DomainError("sum(W) ≠ 1. Consider normalizing the vector W"))
    N, M = size(Z)
    M = length(W)
    q = Int(maximum(Z))
    plmalg = PlmAlg(method, verbose, epsconv, maxit)
    plmvar = PlmVar(N, M, q, q * q,H,lambdaJ, Z, W)
    Jmat, pslike = MinimizePLAsym(plmalg, plmvar)
    score, FN, Jtensor =  ComputeScore(Jmat, plmvar, min_separation)
    return PlmOut(sdata(pslike), Jtensor, score)

end
plmdca(Z,W;kwds...) = plmdca_asym(Z, W, H;kwds...)

function plmdca_asym(filename::String,H;
                theta::Union{Symbol,Real}=:auto,
                max_gap_fraction::Real=0.9,
                remove_dups::Bool=true,
                kwds...)
    time = @elapsed W, Z, N, M, q = ReadFasta(filename, max_gap_fraction, theta, remove_dups)
    println("preprocessing took $time seconds")
    plmdca_asym(Z, W, H; kwds...)
end

plmdca(filename::String,H::Int; kwds...) = plmdca_asym(filename,H; kwds...)


function attentionMinimizePLAsym(alg::PlmAlg, var::PlmVar; initx0 = nothing)
    LL = var.H*var.N*var.N + var.H*var.q2
    x0 = if initx0 == nothing 
        rand(Float64, LL)
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
    min_objective!(opt, (x, g) -> optimfunwrapper2(x, g, var))
    elapstime = @elapsed  (minf, minx, ret) = optimize(opt, x0)
    alg.verbose && @printf("pl = %.4f\t time = %.4f\t", minf, elapstime)
    alg.verbose && println("exit status = $ret")
    pl = minf
    attention_parameters .= minx
    
    return sdata(attention_parameters), pl
end


function L2norm_asym(vec::Array{Float64,1}, plmvar::PlmVar)
    q = plmvar.q
    N = plmvar.N
    lambdaJ = plmvar.lambdaJ

    LL = length(vec)

    mysum = 0.0
    @inbounds @avx for i = 1:LL 
        mysum += vec[i] * vec[i]
    end
    mysum *= lambdaJ
    return mysum
end



function ComputeScore(Jmat::Array{Float64,2}, var::PlmVar, min_separation::Int)

    q = var.q
    N = var.N
    JJ = reshape(Jmat[1:end,:], q, q, N - 1, N)
    Jtemp1 = zeros(q, q, Int(N * (N - 1) / 2))
    Jtemp2 = zeros(q, q, Int(N * (N - 1) / 2))
    l = 1
    for i = 1:(N - 1)
        for j = (i + 1):N
            Jtemp1[:,:,l] = JJ[:,:,j - 1,i] # J_ij as estimated from from g_i.
            Jtemp2[:,:,l] = JJ[:,:,i,j]' # J_ij as estimated from from g_j.
            l = l + 1
        end
    end



    Jtensor1 = inflate_matrix(Jtemp1, N)
    Jtensor2 = inflate_matrix(Jtemp2, N)
    Jplm = (Jtensor1 + Jtensor2) / 2 # for the energy I do not want to gauge

    ctr = 0
    for i in 1:N - 1
        for j in i + 1:N
            ctr += 1
            Jtensor1[:,:,i,j] = Jtemp1[:,:,ctr] - repeat(mean(Jtemp1[:,:,ctr], dims=1), q, 1) - repeat(mean(Jtemp1[:,:,ctr], dims=2), 1, q) .+ mean(Jtemp1[:,:,ctr])
            Jtensor1[:,:,j,i] = Jtensor1[:,:,i,j]'
            Jtensor2[:,:,i,j] = Jtemp2[:,:,ctr] - repeat(mean(Jtemp2[:,:,ctr], dims=1), q, 1) - repeat(mean(Jtemp2[:,:,ctr], dims=2), 1, q) .+ mean(Jtemp2[:,:,ctr])
            Jtensor2[:,:,j,i] = Jtensor2[:,:,i,j]'
        end
    end # zerosumgauge the different tensors

    Jtensor = (Jtensor1 + Jtensor2) / 2

    FN = compute_APC(Jtensor, N, q)
    score = compute_ranking(FN, min_separation)
    return score, FN, Jplm
end



function L2Tensor(matrix)
    L2 = 0.0
    for x in matrix
        L2 += x*x 
    end
    return L2
end