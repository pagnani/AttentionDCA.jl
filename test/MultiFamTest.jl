module MultiFamTest
using AttentionDCA, ArDCA, Test

fams = ["../precompilation_data/PF00014.fasta", "../precompilation_data/PF00014.fasta"]

function test_multifam()
    @test typeof(AttentionDCA.multi_trainer(fams, 1, 2, [2,2], verbose = false)) == @NamedTuple{Qs::Vector{Array{Float64, 3}}, Ks::Vector{Array{Float64, 3}}, V::Array{Float64, 3}}
end
function test_stat_multifam()
    @test typeof(AttentionDCA.stat_multi_trainer(fams, 2, 2, [2,2], n_epochs=2, verbose = false)) == Vector{Vector{Tuple{Int64, Int64, Float64}}}
end

function test_multi_artrainer()
    @test typeof(AttentionDCA.multi_artrainer(fams, 1, 2, [2,2], verbose = false)) == Tuple{Vector{ArDCA.ArNet}, Vector{Any}, @NamedTuple{Qs::Vector{Array{Float64, 3}}, Ks::Vector{Array{Float64, 3}}, V::Array{Float64, 3}}}
end


test_multifam()
test_stat_multifam()
test_multi_artrainer()

printstyled("All MultiFam tests passed\n", bold=true, color=:light_green)

end