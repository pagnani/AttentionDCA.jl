# function loglike(Zm::Matrix{Int}, W, V, site)
#     Jtens = computeJ(W, V)
#     M = size(Zm,2)
#     @tullio vec_ene[α,l] := Jtens[α, Zm[j,l], j, site]
#     @tullio ret := vec_ene[Zm[site,l],l]
#     return ret/M-sum(lsexp_mat(vec_ene))/M
# end

# function lsexp_mat(mat ; dims = 1)
#     max_ = maximum(mat, dims = dims)
#     exp_mat = exp.(mat .- max_)
#     sum_exp_ = sum(exp_mat, dims = dims)
#     log.(sum_exp_) .+ max_
# end

# function computeJ(a::Attention)
#     @extract a Wsf W V
#     Wsf .= softmax(W, dims = 3)
#     @tullio Jtens[a, b, j, i] := Wsf[h, i, j] * V[h, a, b] * (i != j)
# end

# function loglike(Zm::Matrix{Int}, a::Attention, site)
#     Jtens = computeJ(a)
#     N,M = size(Zm,)
#     Jtens_site = Jtens[:,:,:,site]
#     @tullio vec_ene[α,l] := Jtens_site[α, Zm[j,l], j]
#     @tullio ret := vec_ene[Zm[site,l],l]
#     return ret/M-sum(lsexp_mat(vec_ene))/M
    
# end



# Ex. my_fastloglike
function loglike(Z::Matrix{Int},W,V,site)
    M = size(Z,2)
    Wsf_site = softmax(W[:,site,:],dims=2)
    @tullio J[a,b,j] := Wsf_site[h,j]*V[h,a,b]*(site!=j)
    
    @tullio mat_ene[a,m] := J[a,Z[j,m],j]

    pseudologlikelihood = 0.0 
    lge = logsumexp(mat_ene,dims=1)
    Z_site = view(Z,site,:)
    @tullio pseudologlikelihood = mat_ene[Z_site[m],m] - lge[m]
    pseudologlikelihood /= M
    
    return pseudologlikelihood
end



# function lsexp_mat(mat ; dims = 1)
#     max_ = maximum(mat, dims = dims)
#     exp_mat = exp.(mat .- max_)
#     sum_exp_ = sum(exp_mat, dims = dims)
#     log.(sum_exp_) .+ max_
# end


# function lsexp_vec(vec)
#     max_ = maximum(vec)
#     exp_vec = exp.(vec .- max_)
#     sum_exp_ = sum(exp_vec)
#     log.(sum_exp_) .+ max_
# end

# function computeJ(a::Attention)
#     #@extract a Wsf W V
#     Wsf = a.Wsf
#     W = a.W
#     V = a.V
#     Wsf .= softmax(W, dims = 3)
#     @tullio Jtens[a, b, j, i] := Wsf[h, i, j] * V[h, a, b] * (i != j)
# end

# function loglikeslow(Zm::Matrix{Int}, a::Attention, site)
#     Jtens = computeJ(a)
#     M = size(Zm, 2)
#     q,q,N,N=size(Jtens)
#     lf = zeros(q,M)
#     ll = 0.0
#     for l in 1:M
#         #fill!(lf,0.0)
#         for j in 1:N
#             for a in 1:q
#                  lf[a,l] += Jtens[a,Zm[j,l],j,site]
#             end
#         end
#         ll += lf[Zm[site,l],l] -lsexp_vec(lf[:,l])
#     end
#     # return ll / M, lf
#     return ll/M
# end



# function my_slowloglike(Z::Matrix{Int},a::Attention,site)
#     W = a.W 
#     V = a.V 
#     q = a.q
#     N,M = size(Z) 
#     Wsf = softmax(W,dims=3) 
#     J = zeros(Float64, q,q,N,N)
#     @tullio J[a,b,j,i] = Wsf[h,i,j]*V[h,a,b]*(i!=j)
#     mat_ene = zeros(q,M)
#     pseudolikelihood = 0.0
#     for m = 1:M
#         for j = 1:N
#             for a = 1:q
#                 mat_ene[a,m] += J[a,Z[j,m],j,site] 
#             end
#         end
#         pseudolikelihood += mat_ene[Z[site,m],m] - logsumexp(mat_ene[:,m])
#         # pseudolikelihood += mat_ene[Z[site,m],m]
#     end
#     pseudolikelihood /= M
#     return pseudolikelihood
# end