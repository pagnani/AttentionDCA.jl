module AttentionBasedPlmDCA
using Printf, LinearAlgebra, Statistics, Tullio, Zygote, PottsGauge, Random
using NLopt
#using KernelAbstractions
import DCAUtils: read_fasta_alignment, remove_duplicate_sequences, compute_weights, add_pseudocount, compute_weighted_frequencies
using LoopVectorization
using DelimitedFiles: readdlm
using Flux: softmax
using Flux.Optimise: update!
using Distributions: wsample
using ExtractMacro

export PlmOut, attention_plmdca, ar_attention_plmdca,compute_dcascore, compute_PPV

include("types.jl")
include("utils.jl")
include("dcascore.jl")
include("gradient_test.jl")
include("attention.jl")
include("ar.jl")
end
