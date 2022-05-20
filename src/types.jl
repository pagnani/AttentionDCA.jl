struct PlmAlg
    method::Symbol
    verbose::Bool
    epsconv::Float64
    maxit::Int
end

struct PlmOut
    pslike::Union{Vector{Float64},Float64}
    Wtensor::Array{Float64,3}
    Vtensor::Union{Array{Float64,3},Array{Float64,4}}
    score::Array{Tuple{Int, Int, Float64},1}  
end

# struct PlmOutParallel
#     pslike::Union{Vector{Float64},Float64}
#     Wtensor::Array{Float64,3}
#     Vtensor::Array{Float64,4}
#     score::Array{Tuple{Int, Int, Float64},1}  
# end

struct PlmVar
    N::Int
    M::Int
    q::Int  
    q2::Int
    H::Int
    lambda::Float64
    Z::Array{Int,2} #MSA
    W::Array{Float64,1} #weigths
end


struct FAPlmVar
    N::Int
    M::Int
    d::Int
    q::Int  
    q2::Int
    H::Int
    lambda::Float64
    Z::Array{Int,2} #MSA
    W::Array{Float64,1} #weigths
end

struct FAPlmOut
    pslike::Union{Vector{Float64},Float64}
    Qtensor::Array{Float64,3}
    Ktensor::Array{Float64,3}
    Vtensor::Array{Float64,3}
    score::Array{Tuple{Int, Int, Float64},1}  
    roc::Union{Vector{Float64},Nothing}
end

struct FAComputationQuantities 
    sf::Array{Float64,3}
    J::Array{Float64,4}
    mat::Array{Float64,4}
    fact::Array{Float64,3}
    function FAComputationQuantities(N,H,q)
        sf = zeros(Float64, H, N, N)
        J = zeros(Float64, N, N, q, q)
        mat = zeros(Float64, N, q, q, N)
        fact = zeros(Float64, N, H, N)
        new(sf,J,mat,fact)
    end
end
        
        

