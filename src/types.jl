struct PlmAlg
    method::Symbol
    verbose::Bool
    epsconv::Float64
    maxit::Int
end

struct PlmOut
    pslike::Union{Vector{Float64},Float64}
    Jtensor::Array{Float64,4}
    score::Array{Tuple{Int, Int, Float64},1}  
end

struct Attention
    N::Int
    H::Int
    q::Int
    #W::Array{Float64,3}
    #V::Array{Float64,3}
    # Wsf::Array{Float64,3}
    # function Attention(N,H,q,W,V)
    #     Wsf = zeros(Float64, H,N,N)
    #     softmax!(Wsf, W, dims=3) 
    #     new(N,H,q,W,V,Wsf)
    # end
end

struct PlmVar
    N::Int
    M::Int
    q::Int  
    q2::Int
    H::Int
    lambdaJ::Float64
    Z::SharedArray{Int,2} #MSA
    W::SharedArray{Float64,1} #weigths 
    IdxZ::SharedArray{Int,2} #partial index computation for speed up energy calculation
    function PlmVar(N,M,q,q2,H,lambdaJ, Z,W)
        sZ = SharedArray{Int}(size(Z))
        sZ[:] = Z
        sW = SharedArray{Float64}(size(W))
        sW[:] = W
 
        IdxZ = Array{Int}(undef, N, M)
        q2=q*q
        for i in 1:M
            for j in 1:N
                IdxZ[j,i] = (j-1) * q2 + q * (Z[j,i] - 1)
            end
        end
        sIdxZ = SharedArray{Int}(size(IdxZ))
        sIdxZ[:] = IdxZ
        new(N,M,q,q2,H,lambdaJ, sZ, sW, sIdxZ)
    end
end


struct DecVar{N} 
    fracdec::Float64
    fracmax::Float64
    blockdecimate::Bool
    dmask::SharedArray{Bool,N}
    function DecVar{N}(fracdec, fracmax, blockdecimate, dmask) where N
        sdmask = SharedArray{Bool}(size(dmask))
        sdmask[:] = dmask 
        new(fracdec, fracmax, blockdecimate, sdmask)
    end
end