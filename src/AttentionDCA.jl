module AttentionDCA

import DCAUtils: compute_weighted_frequencies, compute_weights, read_fasta_alignment, remove_duplicate_sequences
import Flux
import Flux: Adam, DataLoader, gradient, softmax
import Flux.Optimise: update!
import Flux.Optimisers: setup

using LoopVectorization
using Tullio 
using ArDCA, PottsGauge
using DelimitedFiles: readdlm
using ExtractMacro, Printf
using LinearAlgebra
using Random, Statistics




export trainer, stat_trainer, artrainer, stat_artrainer, multi_trainer, multi_artrainer, stat_multi_trainer
export score, compute_freq, compute_PPV, sample, quickread

include("types.jl")
include("utils.jl")
include("score.jl")
include("attention.jl")
include("multifam.jl")
include("autoregressive.jl")
end
