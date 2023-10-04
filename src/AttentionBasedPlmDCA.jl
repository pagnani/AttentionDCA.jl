module AttentionBasedPlmDCA
using Printf, LinearAlgebra, Statistics, Tullio, PottsGauge, Random
#using NLopt
#using KernelAbstractions
import DCAUtils: read_fasta_alignment, remove_duplicate_sequences, compute_weights, add_pseudocount, compute_weighted_frequencies
using LoopVectorization
using PrecompileTools
using DelimitedFiles: readdlm
import Flux
import Flux: DataLoader, Adam, gradient
import Flux.Optimise: update! 
import Flux.Optimisers: setup
import Distributions: wsample
using ExtractMacro
using ArDCA


export AttOut, AttPlmVar, score, compute_freq, compute_PPV, L2reg, sample, trainer, artrainer, artrainer2, attention_matrix_asym, attention_matrix_sym

include("types.jl")
include("gauge.jl")
include("utils.jl")
include("dcascore.jl")
include("gradient_test.jl")
include("attention.jl")
include("attention_analysis_utils.jl")
include("multifam.jl")
#include("stochastic.jl")
include("autoregressive.jl")
#include("newfile.jl")
#include("precompile.jl")
end
