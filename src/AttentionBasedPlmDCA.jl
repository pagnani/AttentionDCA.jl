module AttentionBasedPlmDCA
using SharedArrays, Distributed, Printf, LinearAlgebra, Statistics, Tullio, Flux, Zygote, PottsGauge
using NLopt
#using ExtractMacro,KernelAbstractions
import DCAUtils: read_fasta_alignment, remove_duplicate_sequences, compute_weights, add_pseudocount, compute_weighted_frequencies
using LoopVectorization
using DelimitedFiles: readdlm

export PlmOut, attention_plmdca, compute_dcascore, compute_PPV

include("types.jl")
include("utils.jl")
include("attention_plmdca.jl")
include("dcascore.jl")
include("gradient_test.jl")
end
