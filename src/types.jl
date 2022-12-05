struct PlmAlg
    method::Symbol
    verbose::Bool
    epsconv::Float64
    maxit::Int
end

# struct PlmOut
#     pslike::Union{Vector{Float64},Float64}
#     Wtensor::Array{Float64,3}
#     Vtensor::Union{Array{Float64,3},Array{Float64,4}}
#     score::Array{Tuple{Int, Int, Float64},1}  
# end



struct AttPlmVar
    N::Int
    M::Int
    d::Int
    q::Int  
    q2::Int
    H::Int
    lambda::Float64
    Z::Array{Int,2} #MSA
    W::Array{Float64,1} #weigths
    delta::Array{Int,3}
    # idx::Dict{Any,Any}
    wdelta::Array{Float64,3}
    function AttPlmVar(N,M,d,q,H,lambda,Z,Weigths)
        idx = Dict()
        @tullio delta[j,m,a] := Int(Z[j,m]==a) (a in 1:q)
        @tullio wdelta[j,m,a] := Weigths[m]*delta[j,m,a]
        # for a in 1:q
        #     for j in 1:N
        #         push!(idx, [j,a] => findall(x->x==a,Z[j,:]))
        #     end
        # end
        
        new(N,M,d,q,q*q,H,lambda,Z,Weigths,delta,wdelta)
    end
end

function Base.show(io::IO, AttPlmVar::AttPlmVar)
    @extract AttPlmVar: N M d q H lambda
    print(io,"AttPlmVar: \nN=$N\nM=$M\nq=$q\nH=$H\nd=$d\nλ=$(lambda)")
end


struct AttPlmOut
    Q::Array{Float64,3}
    K::Array{Float64,3}
    V::Array{Float64,3}
    pslike::Union{Vector{Float64},Float64}
end

function Base.show(io::IO, AttPlmOut::AttPlmOut)
    @extract AttPlmOut: Q K V pslike
    H,d,N = size(Q)
    H,q,q = size(V) 
    print(io,"AttPlmOut: \nsize(Q)=[$H,$d,$N]\nsize(K)=[$H,$d,$N]\nsize(V)=[$H,$q,$q]\npslike=$(pslike)")
end


struct AttComputationQuantities 
    sf::Array{Float64,3}
    J::Array{Float64,4}
    mat::Array{Float64,4}
    fact::Array{Float64,3}
    function AttComputationQuantities(N,H,q)
        sf = zeros(Float64, H, N, N)
        J = zeros(Float64, N, N, q, q)
        mat = zeros(Float64, N, q, q, N)
        fact = zeros(Float64, N, H, N)
        new(sf,J,mat,fact)
    end
end

        

