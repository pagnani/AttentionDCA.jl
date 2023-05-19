module AttentionBasedPlmDCA
using Printf, LinearAlgebra, Statistics, Tullio, PottsGauge, Random
using NLopt
#using KernelAbstractions
import DCAUtils: read_fasta_alignment, remove_duplicate_sequences, compute_weights, add_pseudocount, compute_weighted_frequencies
using LoopVectorization
using PrecompileTools
using DelimitedFiles: readdlm
using Flux: DataLoader, Adam, gradient
using Flux.Optimise: update! 
using Flux.Optimisers: setup
using Distributions: wsample
using ExtractMacro
using ArDCA

export AttOut, AttPlmVar, FieldAttPlmVar, attentiondca, arattentiondca, score, compute_PPV, L2reg, sample, my_attentiondca, mytrainer

include("types.jl")
include("utils.jl")
include("dcascore.jl")
include("gradient_test.jl")
#include("precompile.jl")
include("attention.jl")
include("stochastic.jl")
include("autoregressive.jl")
include("newfile.jl")
end
