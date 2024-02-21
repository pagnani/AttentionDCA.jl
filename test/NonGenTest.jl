module NonGenTest
using AttentionDCA, Test
using AttentionDCA: softmax, logsumexp, loss

function test_non_generative_version()
    @test typeof(AttentionDCA.trainer("../precompilation_data/PF00014.fasta", 1, H = 2, d = 2, structfile = "../precompilation_data/PF00014_struct.dat", verbose = false)) == @NamedTuple{Q::Array{Float64, 3}, K::Array{Float64, 3}, V::Array{Float64, 3}}
end

function test_stat_non_generative_version()
    @test typeof(AttentionDCA.stat_trainer("../precompilation_data/PF00014.fasta", 2, n_epochs=2, H = 2, d = 2, structfile = "../precompilation_data/PF00014_struct.dat", verbose = false)) == Vector{Tuple{Int64, Int64, Float64}}
end



function test_loss()

    Z = rand(1:2, 2, 2)
    W = fill(1/2, 2)
    Q = rand(2, 2, 2)
    K = rand(2, 2, 2)
    V = rand(2, 2, 2)

    function _loss(Q::Array{T1, 3},
        K::Array{T1, 3},
        V::Array{T1, 3}, 
        Z::Matrix{T2},
        W::Vector{T3};
        λ::Float64 = 0.001) where {T1<:Real, T2<: Integer, T3<:Real}
    
        H,d,L = size(Q)
        H,d,L = size(K)
        H,q,q = size(V)
        L,M = size(Z)
        
    
        sf = zeros(H,L,L)
        for h in 1:H
            for i in 1:L
                for j in 1:L
                    for n in 1:d
                        sf[h,i,j] += Q[h,n,i]*K[h,n,j]
                    end
                end
            end
        end 
        
        sf = softmax(sf, dims=3)
        J = zeros(L,L,q,q)
        for i in 1:L
            for j in 1:L
                if j != i
                    for h in 1:H
                        for a in 1:q
                            for b in 1:q
                                J[i, j, a, b] += sf[h, i, j]*V[h, a, b]
                            end
                        end
                    end    
                end
            end
        end
    
        mat_ene = zeros(q, L, M)
        
        for m in 1:M
            for a in 1:q
                for i in 1:L
                    for j in 1:L
                        mat_ene[a, i, m] += J[i, j, a, Z[j, m]]
                    end
                end
            end
        end
    
        lge = logsumexp(mat_ene,dims=1)[1,:,:]
    
        pl = 0.0
    
    
        for i in 1:L
            for m in 1:M
                pl -= W[m]*(mat_ene[Z[i, m], i, m] - lge[i, m])
            end
        end
    
        pl += λ*(sum(abs2, J))
    
        return pl
    end


    @test _loss(Q, K, V, Z, W) ≈ loss(Q, K, V, Z, W)
end



test_non_generative_version()
test_stat_non_generative_version()
test_loss()

printstyled("All Non_Generative_Version tests passed\n", bold=true, color=:light_green)

end