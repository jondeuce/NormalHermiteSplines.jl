if normpath(@__DIR__) ∉ LOAD_PATH
    pushfirst!(LOAD_PATH, normpath(@__DIR__, ".."))
    pushfirst!(LOAD_PATH, normpath(@__DIR__))
end

using NormalHermiteSplines
const nhs = NormalHermiteSplines

using BenchmarkTools
using LinearAlgebra
using Random
using StaticArrays

function random_nodes(::Val{n} = 2, ::Type{T} = Float64; max_size) where {n, T}
    min_bound   = -rand(SVector{n, T})
    max_bound   = rand(SVector{n, T})
    rand_node() = min_bound .+ rand(SVector{n, T}) .* (max_bound .- min_bound)
    rand_dir()  = (x = rand_node(); x / norm(x))

    nodes    = [min_bound, max_bound, (rand_node() for i in 3:max_size)...]
    values   = rand(T, max_size)
    d_nodes  = [rand_node() for _ in 1:n*max_size]
    d_dirs   = [rand_dir() for _ in 1:n*max_size]
    d_values = rand(T, n * max_size)
    return (; min_bound, max_bound, nodes, values, d_nodes, d_dirs, d_values)
end

function bench_insertion(::Val{n} = Val(2), ::Type{T} = Float64) where {n, T}
    for max_size in [8, 16, 32]
        (; min_bound, max_bound, nodes, values, d_nodes, d_dirs, d_values) = random_nodes(Val(n), T; max_size = max_size)
        rk_H0 = RK_H0(T(0.5 + rand()))
        rk_H1 = RK_H1(T(0.5 + rand()))
        espl_H0 = ElasticNormalSpline(min_bound, max_bound, max_size, rk_H0)
        espl_H1_0 = ElasticNormalSpline(min_bound, max_bound, max_size, rk_H1)
        espl_H1_1 = ElasticNormalSpline(min_bound, max_bound, max_size, rk_H1)

        print((dim = n, eltype = T, spline = :elastic, kernel = :RK_H0, nodes = max_size, d_nodes = 0))
        @btime (empty!($espl_H0); insert!($espl_H0, $nodes, $values))

        print((dim = n, eltype = T, spline = :elastic, kernel = :RK_H1, nodes = max_size, d_nodes = 0))
        @btime (empty!($espl_H1_0); insert!($espl_H1_0, $nodes, $values))

        print((dim = n, eltype = T, spline = :elastic, kernel = :RK_H1, nodes = max_size, d_nodes = n * max_size))
        @btime (empty!($espl_H1_1); insert!($espl_H1_1, $nodes, $values, $d_nodes, $d_dirs, $d_values))

        print((dim = n, eltype = T, spline = :normal, kernel = :RK_H0, nodes = max_size, d_nodes = 0))
        @btime for i in 2:$max_size # i=1 errors; requires either 2+ nodes or 1+ node and 1+ derivative node
            interpolate(view($nodes, 1:i), view($values, 1:i), $rk_H0)
        end

        print((dim = n, eltype = T, spline = :normal, kernel = :RK_H1, nodes = max_size, d_nodes = 0))
        @btime for i in 2:$max_size # i=1 errors; requires either 2+ nodes or 1+ node and 1+ derivative node
            interpolate(view($nodes, 1:i), view($values, 1:i), $rk_H1)
        end

        print((dim = n, eltype = T, spline = :normal, kernel = :RK_H1, nodes = max_size, d_nodes = n * max_size))
        @btime for i in 1:$max_size
            interpolate(view($nodes, 1:i), view($values, 1:i), view($d_nodes, 1:$n*i), view($d_dirs, 1:$n*i), view($d_values, 1:$n*i), $rk_H1)
        end

        println("")
    end
end

bench_insertion(Val(1));
bench_insertion(Val(2));
bench_insertion(Val(3));
