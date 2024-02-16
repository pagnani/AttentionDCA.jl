module NonGenTest
using AttentionDCA, Test

function test_non_generative_version()
    @test typeof(AttentionDCA.trainer("../precompilation_data/PF00014.fasta", 1, H = 2, d = 2, structfile = "../precompilation_data/PF00014_struct.dat", verbose = false)) == @NamedTuple{Q::Array{Float64, 3}, K::Array{Float64, 3}, V::Array{Float64, 3}}
end

function test_stat_non_generative_version()
    @test typeof(AttentionDCA.stat_trainer("../precompilation_data/PF00014.fasta", 2, n_epochs=2, H = 2, d = 2, structfile = "../precompilation_data/PF00014_struct.dat", verbose = false)) == Vector{Tuple{Int64, Int64, Float64}}
end

test_non_generative_version()
test_stat_non_generative_version()

printstyled("All Non_Generative_Version tests passed\n", bold=true, color=:light_green)

end