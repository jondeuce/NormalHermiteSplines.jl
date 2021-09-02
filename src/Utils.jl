@inbounds function _normalize(point::SVector{n}, min_bound::SVector{n}, max_bound::SVector{n}, compression::Real) where {n}
    return (point .- min_bound) ./ compression
    # return clamp.((point .- min_bound) ./ compression, 0, 1) #TODO: clamp nodes? roughly equivalent to nearest neighbour extrapolation
end
@inbounds function _normalize(spline::NormalSpline{n}, point::SVector{n}) where {n}
    return _normalize(point, spline._min_bound, spline._max_bound, spline._compression)
end

@inbounds function _unnormalize(point::SVector{n}, min_bound::SVector{n}, max_bound::SVector{n}, compression::Real) where {n}
    return min_bound .+ compression .* point
    # return clamp.(min_bound .+ compression .* point, min_bound, max_bound) #TODO: clamp nodes? roughly equivalent to nearest neighbour extrapolation
end
@inbounds function _unnormalize(spline::NormalSpline{n}, point::SVector{n}) where {n}
    return _unnormalize(point, spline._min_bound, spline._max_bound, spline._compression)
end

function _normalization_scaling(nodes::AbstractVecOfSVecs)
    min_bound = reduce((x, y) -> min.(x, y), nodes)
    max_bound = reduce((x, y) -> max.(x, y), nodes)
    compression = maximum(max_bound .- min_bound)
    return min_bound, max_bound, compression
end

function _normalization_scaling(nodes::AbstractVecOfSVecs, d_nodes::AbstractVecOfSVecs)
    min_bound = min.(reduce((x, y) -> min.(x, y), nodes), reduce((x, y) -> min.(x, y), d_nodes))
    max_bound = max.(reduce((x, y) -> max.(x, y), nodes), reduce((x, y) -> max.(x, y), d_nodes))
    compression = maximum(max_bound .- min_bound)
    return min_bound, max_bound, compression
end

function _estimate_accuracy(spline::NormalSpline{n,T,RK}) where {n,T,RK <: ReproducingKernel_0}
    vmax = maximum(abs, spline._values)
    rmae = zero(T)
    @inbounds for i in 1:length(spline._nodes)
        point = _unnormalize(spline, spline._nodes[i])
        σ = _evaluate(spline, point)
        rmae = max(rmae, abs(spline._values[i] - σ))
    end
    if vmax > 0
        rmae /= vmax
    end
    rmae = max(rmae, eps(T))
    res  = -floor(log10(rmae)) - 1
    res  = max(res, 0)
    return trunc(Int, res)
end

function _pairwise_sum_norms(nodes::AbstractVecOfSVecs{n,T}) where {n,T}
    ℓ = zero(T)
    @inbounds for i in 1:length(nodes), j in i:length(nodes)
        ℓ += norm(nodes[i] .- nodes[j])
    end
    return ℓ
end

function _pairwise_sum_norms_weighted(nodes::AbstractVecOfSVecs{n,T}, d_nodes::AbstractVecOfSVecs{n,T}, w_d_nodes::T) where {n,T}
    ℓ = zero(T)
    @inbounds for i in 1:length(nodes), j in 1:length(d_nodes)
        ℓ += norm(nodes[i] .- w_d_nodes .* d_nodes[j])
    end
    return ℓ
end

@inline _ε_factor(::RK_H0, ε::T) where {T} = one(T)
@inline _ε_factor(::RK_H1, ε::T) where {T} = T(3)/2
@inline _ε_factor(::RK_H2, ε::T) where {T} = T(2)

@inline _ε_factor_d(::RK_H0, ε::T) where {T} = one(T)
@inline _ε_factor_d(::RK_H1, ε::T) where {T} = T(2)
@inline _ε_factor_d(::RK_H2, ε::T) where {T} = T(5)/2

function _estimate_ε(nodes::AbstractVecOfSVecs{n,T}) where {n,T}
    n_1 = length(nodes)
    ε   = _pairwise_sum_norms(nodes)
    return ε > 0 ? ε * T(n)^T(inv(n)) / T(n_1)^(T(5) / 3) : one(T)
end

function _estimate_ε(nodes::AbstractVecOfSVecs{n,T}, d_nodes::AbstractVecOfSVecs{n,T}, w_d_nodes::T=T(0.1)) where {n,T}
    n_1 = length(nodes)
    n_2 = length(d_nodes)
    ε   = _pairwise_sum_norms(nodes) + _pairwise_sum_norms_weighted(nodes, d_nodes, w_d_nodes) + w_d_nodes * _pairwise_sum_norms(d_nodes)
    return ε > 0 ? ε * T(n)^T(inv(n)) / T(n_1 + n_2)^(T(5) / 3) : one(T)
end

function _estimate_epsilon(nodes::AbstractVecOfSVecs, kernel::ReproducingKernel_0)
    min_bound, max_bound, compression = _normalization_scaling(nodes)
    nodes = _normalize.(nodes, (min_bound,), (max_bound,), compression)
    ε     = _estimate_ε(nodes)
    ε    *= _ε_factor(kernel, ε)
    return ε
end

function _estimate_epsilon(nodes::AbstractVecOfSVecs, d_nodes::AbstractVecOfSVecs, kernel::ReproducingKernel_1)
    min_bound, max_bound, compression = _normalization_scaling(nodes, d_nodes)
    nodes   = _normalize.(nodes, (min_bound,), (max_bound,), compression)
    d_nodes = _normalize.(d_nodes, (min_bound,), (max_bound,), compression)
    ε       = _estimate_ε(nodes, d_nodes)
    ε      *= _ε_factor_d(kernel, ε)
    return ε
end

function _get_gram(nodes::AbstractVecOfSVecs, kernel::ReproducingKernel_0)
    min_bound, max_bound, compression = _normalization_scaling(nodes)
    nodes  = _normalize.(nodes, (min_bound,), (max_bound,), compression)
    if kernel.ε == 0
        ε  = _estimate_ε(nodes)
        ε *= _ε_factor(kernel, ε)
        kernel = typeof(kernel)(ε)
    end
    return _gram!(zeros(T, n_1, n_1), nodes, kernel)
end

function _get_gram(nodes::AbstractVecOfSVecs, d_nodes::AbstractVecOfSVecs, es::AbstractVecOfSVecs, kernel::ReproducingKernel_1)
    min_bound, max_bound, compression = _normalization_scaling(nodes, d_nodes)
    nodes   = _normalize.(nodes, (min_bound,), (max_bound,), compression)
    d_nodes = _normalize.(d_nodes, (min_bound,), (max_bound,), compression)
    es      = es ./ norm.(es)
    if kernel.ε == 0
        ε   = _estimate_ε(nodes, d_nodes)
        ε  *= _ε_factor_d(kernel, ε)
        kernel = typeof(kernel)(ε)
    end
    return _gram!(zeros(T, n_1 + n_2, n_1 + n_2), nodes, d_nodes, es, kernel)
end

function _get_cond(nodes::AbstractVecOfSVecs, kernel::ReproducingKernel_0)
    T = promote_type(typeof(kernel.ε), eltype(eltype(nodes)))
    gram = _get_gram(nodes, kernel)
    cond = zero(T)
    try
        evs = svdvals!(gram)
        maxevs = maximum(evs)
        minevs = minimum(evs)
        if minevs > 0
            cond = maxevs / minevs
            cond = T(10)^floor(log10(cond))
        end
    catch
    end
    return cond
end

function _get_cond(nodes::AbstractVecOfSVecs, d_nodes::AbstractVecOfSVecs, es::AbstractVecOfSVecs, kernel::ReproducingKernel_1)
    _get_cond(nodes, kernel)
end

# ```
# Get estimation of the Gram matrix condition number
# Brás, C.P., Hager, W.W. & Júdice, J.J. An investigation of feasible descent algorithms for estimating the condition number of a matrix. TOP 20, 791–809 (2012).
# https://link.springer.com/article/10.1007/s11750-010-0161-9
# ```
function _estimate_cond(
        gram::AbstractMatrix{T},
        chol::LinearAlgebra.Cholesky{T,Array{T,2}},
        nit = 3,
    ) where {T}
    normgram = norm(gram, 1)
    n = size(gram, 1)
    x = fill(inv(T(n)), n)
    z = Vector{T}(undef, n)
    gamma = zero(T)
    for _ in 1:nit
        z = ldiv!(z, chol, x)
        gamma = zero(T)
        @inbounds for i in 1:n
            gamma += abs(z[i])
            z[i] = sign(z[i])
        end
        z = ldiv!(z, chol, copy(z))
        zx = z ⋅ x
        idx = 1
        @inbounds for i in 1:n
            z[i] = abs(z[i])
            if z[i] > z[idx]
                idx = i
            end
        end
        @inbounds begin
            if z[idx] <= zx
                break
            end
            x .= 0
            x[idx] = 1
        end
    end
    cond = T(10)^floor(log10(normgram * gamma))
    return cond
end
