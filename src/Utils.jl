@inbounds function _normalize(point::SVector{n}, min_bound::SVector{n}, max_bound::SVector{n}, scale::Real) where {n}
    return (point .- min_bound) ./ scale
end
@inbounds function _normalize(spl::AbstractNormalSpline{n}, point::SVector{n}) where {n}
    return _normalize(point, _get_min_bound(spl), _get_max_bound(spl), _get_scale(spl))
end

@inbounds function _unnormalize(point::SVector{n}, min_bound::SVector{n}, max_bound::SVector{n}, scale::Real) where {n}
    return min_bound .+ scale .* point
end
@inbounds function _unnormalize(spl::AbstractNormalSpline{n}, point::SVector{n}) where {n}
    return _unnormalize(point, _get_min_bound(spl), _get_max_bound(spl), _get_scale(spl))
end

function _normalization_scaling(nodes::AbstractVecOfSVecs)
    min_bound = reduce((x, y) -> min.(x, y), nodes; init = fill(+Inf, eltype(nodes)))
    max_bound = reduce((x, y) -> max.(x, y), nodes; init = fill(-Inf, eltype(nodes)))
    scale = maximum(max_bound .- min_bound)
    return min_bound, max_bound, scale
end

function _normalization_scaling(nodes::AbstractVecOfSVecs, d_nodes::AbstractVecOfSVecs)
    min_bound = min.(reduce((x, y) -> min.(x, y), nodes; init = fill(+Inf, eltype(nodes))), reduce((x, y) -> min.(x, y), d_nodes; init = fill(+Inf, eltype(d_nodes))))
    max_bound = max.(reduce((x, y) -> max.(x, y), nodes; init = fill(-Inf, eltype(nodes))), reduce((x, y) -> max.(x, y), d_nodes; init = fill(-Inf, eltype(d_nodes))))
    scale = maximum(max_bound .- min_bound)
    return min_bound, max_bound, scale
end

function _estimate_accuracy(spl::AbstractNormalSpline{n, T, RK}) where {n, T, RK <: ReproducingKernel_0}
    vmax = maximum(abs, _get_values(spl))
    rmae = zero(T)
    @inbounds for i in 1:length(_get_nodes(spl))
        point = _unnormalize(spl, _get_nodes(spl)[i])
        σ     = _evaluate(spl, point)
        rmae  = max(rmae, abs(_get_values(spl)[i] - σ))
    end
    if vmax > 0
        rmae /= vmax
    end
    rmae = max(rmae, eps(T))
    res  = -floor(log10(rmae)) - 1
    res  = max(res, 0)
    return trunc(Int, res)
end

function _pairwise_sum_norms(nodes::AbstractVecOfSVecs{n, T}) where {n, T}
    ℓ = zero(T)
    @inbounds for i in 1:length(nodes), j in i:length(nodes)
        ℓ += _norm(nodes[i] - nodes[j])
    end
    return ℓ
end

function _pairwise_sum_norms_weighted(nodes::AbstractVecOfSVecs{n, T}, d_nodes::AbstractVecOfSVecs{n, T}, w_d_nodes::T) where {n, T}
    ℓ = zero(T)
    @inbounds for i in 1:length(nodes), j in 1:length(d_nodes)
        ℓ += _norm(nodes[i] - w_d_nodes * d_nodes[j])
    end
    return ℓ
end

@inline _ε_factor(::RK_H0, ε::T) where {T} = one(T)
@inline _ε_factor(::RK_H1, ε::T) where {T} = T(3) / 2
@inline _ε_factor(::RK_H2, ε::T) where {T} = T(2)

@inline _ε_factor_d(::RK_H0, ε::T) where {T} = one(T)
@inline _ε_factor_d(::RK_H1, ε::T) where {T} = T(2)
@inline _ε_factor_d(::RK_H2, ε::T) where {T} = T(5) / 2

function _estimate_ε(k::ReproducingKernel_0, nodes)
    ε = _estimate_ε(nodes)
    ε *= _ε_factor(k, ε)
    k = typeof(k)(ε)
    return k
end

function _estimate_ε(k::ReproducingKernel_0, nodes, d_nodes)
    ε = _estimate_ε(nodes, d_nodes)
    ε *= _ε_factor_d(k, ε)
    k = typeof(k)(ε)
    return k
end

function _estimate_ε(nodes::AbstractVecOfSVecs{n, T}) where {n, T}
    n₁ = length(nodes)
    ε  = _pairwise_sum_norms(nodes)
    return ε > 0 ? ε * T(n)^T(inv(n)) / T(n₁)^(T(5) / 3) : one(T)
end

function _estimate_ε(nodes::AbstractVecOfSVecs{n, T}, d_nodes::AbstractVecOfSVecs{n, T}, w_d_nodes::T = T(0.1)) where {n, T}
    n₁ = length(nodes)
    n₂ = length(d_nodes)
    ε  = _pairwise_sum_norms(nodes) + _pairwise_sum_norms_weighted(nodes, d_nodes, w_d_nodes) + w_d_nodes * _pairwise_sum_norms(d_nodes)
    return ε > 0 ? ε * T(n)^T(inv(n)) / T(n₁ + n₂)^(T(5) / 3) : one(T)
end

function _estimate_epsilon(nodes::AbstractVecOfSVecs, kernel::ReproducingKernel_0)
    min_bound, max_bound, scale = _normalization_scaling(nodes)
    nodes = _normalize.(nodes, (min_bound,), (max_bound,), scale)
    ε = _estimate_ε(nodes)
    ε *= _ε_factor(kernel, ε)
    return ε
end

function _estimate_epsilon(nodes::AbstractVecOfSVecs, d_nodes::AbstractVecOfSVecs, kernel::ReproducingKernel_1)
    min_bound, max_bound, scale = _normalization_scaling(nodes, d_nodes)
    nodes = _normalize.(nodes, (min_bound,), (max_bound,), scale)
    d_nodes = _normalize.(d_nodes, (min_bound,), (max_bound,), scale)
    ε = _estimate_ε(nodes, d_nodes)
    ε *= _ε_factor_d(kernel, ε)
    return ε
end

function _get_gram(nodes::AbstractVecOfSVecs, kernel::ReproducingKernel_0)
    min_bound, max_bound, scale = _normalization_scaling(nodes)
    nodes = _normalize.(nodes, (min_bound,), (max_bound,), scale)
    if kernel.ε == 0
        kernel = _estimate_ε(kernel, nodes)
    end
    return _gram(nodes, kernel)
end

function _get_gram(nodes::AbstractVecOfSVecs, d_nodes::AbstractVecOfSVecs, d_dirs::AbstractVecOfSVecs, kernel::ReproducingKernel_1)
    min_bound, max_bound, scale = _normalization_scaling(nodes, d_nodes)
    nodes = _normalize.(nodes, (min_bound,), (max_bound,), scale)
    d_nodes = _normalize.(d_nodes, (min_bound,), (max_bound,), scale)
    d_dirs = d_dirs ./ _norm.(d_dirs)
    if kernel.ε == 0
        kernel = _estimate_ε(kernel, nodes, d_nodes)
    end
    return _gram(nodes, d_nodes, d_dirs, kernel)
end

function _get_cond(nodes::AbstractVecOfSVecs, kernel::ReproducingKernel_0)
    T = promote_type(eltype(kernel), eltype(eltype(nodes)))
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

function _get_cond(nodes::AbstractVecOfSVecs, d_nodes::AbstractVecOfSVecs, d_dirs::AbstractVecOfSVecs, kernel::ReproducingKernel_1)
    return _get_cond(nodes, kernel)
end

"""
Get estimation of the Gram matrix condition number
Brás, C.P., Hager, W.W. & Júdice, J.J. An investigation of feasible descent algorithms for estimating the condition number of a matrix. TOP 20, 791–809 (2012).
https://link.springer.com/article/10.1007/s11750-010-0161-9
"""
function _estimate_cond(A::AbstractMatrix{T}, F::Factorization, nit = 3) where {T}
    Anorm = norm(A, 1)
    n = size(A, 1)
    x = fill(inv(T(n)), n)
    z = zeros(T, n)
    z′ = zeros(T, n)
    γ = zero(T)
    @inbounds for _ in 1:nit
        ldiv!(z′, F, x)
        γ = zero(T)
        for i in 1:n
            γ += abs(z′[i])
            z′[i] = sign(z′[i])
        end
        ldiv!(z, F, z′)
        zdotx = z ⋅ x
        zmax, imax = T(-Inf), 1
        for i in 1:n
            zᵢ = z[i] = abs(z[i])
            if zᵢ > zmax
                zmax, imax = zᵢ, i
            end
        end
        (zmax <= zdotx) && break
        x .= 0
        x[imax] = 1
    end
    cond = prevpow(T(10), Anorm * γ)
    return cond
end
