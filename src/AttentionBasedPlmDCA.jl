module AttentionBasedPlmDCA
using SharedArrays, Distributed, Printf, LinearAlgebra, Statistics
using NLopt
import DCAUtils: read_fasta_alignment, remove_duplicate_sequences, compute_weights, add_pseudocount, compute_weighted_frequencies
using LoopVectorization
using DelimitedFiles: readdlm

export PlmOut, plmdca, plmdca_asym, plmdca_asym, computescore 

include("types.jl")
include("utils.jl")
include("plmdca_asym.jl")
include("dcascore.jl")
end
