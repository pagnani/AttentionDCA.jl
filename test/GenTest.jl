module GenTest
using AttentionDCA, ArDCA, Test

function test_generative_version()
    @test typeof(AttentionDCA.artrainer("../precompilation_data/PF00014.fasta", 1, H = 2, d = 2, verbose = false)) == Tuple{ArDCA.ArNet, ArDCA.ArVar{Int64}, @NamedTuple{Q::Array{Float64, 3}, K::Array{Float64, 3}, V::Array{Float64, 3}}}
end

function test_stat_generative_version()
    @test typeof(AttentionDCA.stat_artrainer("../precompilation_data/PF00014.fasta", 2, n_epochs=2, H = 2, d = 2, verbose = false)) == Vector{Tuple{Int64, Int64, Float64}}
end

test_generative_version()
test_stat_generative_version()

printstyled("All Generative_Version tests passed\n", bold=true, color=:light_green)

end