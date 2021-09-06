function _prepare(nodes::AbstractVecOfSVecs{n,T}, kernel::ReproducingKernel_0) where {n,T}
    # Normalize nodes out-of-place to avoid aliasing
    min_bound, max_bound, scale = _normalization_scaling(nodes)
    nodes = _normalize.(nodes, (min_bound,), (max_bound,), scale)

    if kernel.ε == 0
        kernel = _estimate_ε(kernel, nodes)
    end

    values   = zeros(T, 0)
    d_nodes  = zeros(SVector{n,T}, 0)
    d_dirs   = zeros(SVector{n,T}, 0)
    d_values = zeros(T, 0)
    mu       = zeros(T, 0)
    gram     = _gram(nodes, kernel)
    chol     = cholesky(gram)
    cond     = _estimate_cond(gram, chol)

    return NormalSpline(kernel, nodes, values, d_nodes, d_dirs, d_values, mu, gram, chol, cond, min_bound, max_bound, scale)
end

function _construct!(
        spline::NormalSpline{n,T,RK},
        values::AbstractVector{T},
    ) where {n, T, RK <: ReproducingKernel_0}
    n₁ = length(values)
    length(spline._nodes) != n₁ && error("Number of data values ($n₁) does not correspond to the number of nodes $(length(spline._nodes)).")
    size(spline._chol) != (n₁, n₁) && error("Number of data values ($n₁) does not correspond to the size of the Gram matrix ($(size(spline._chol))).")

    # Resize buffers
    resize!(spline._values, n₁)
    empty!(spline._d_nodes)
    empty!(spline._d_dirs)
    empty!(spline._d_values)
    resize!(spline._mu, n₁)

    # Copy values to avoid aliasing
    spline._values .= values

    # Compute spline coefficients
    ldiv!(spline._mu, spline._chol, spline._values)

    return spline
end

###################

function _prepare(nodes::AbstractVecOfSVecs{n,T}, d_nodes::AbstractVecOfSVecs{n,T}, d_dirs::AbstractVecOfSVecs{n,T}, kernel::ReproducingKernel_1) where {n,T}
    # Normalize inputs out-of-place to avoid aliasing
    min_bound, max_bound, scale = _normalization_scaling(nodes, d_nodes)
    nodes   = _normalize.(nodes, (min_bound,), (max_bound,), scale)
    d_nodes = _normalize.(d_nodes, (min_bound,), (max_bound,), scale)
    d_dirs  = d_dirs ./ norm.(d_dirs)

    if kernel.ε == 0
        kernel = _estimate_ε(kernel, nodes, d_nodes)
    end

    values   = zeros(T, 0)
    d_values = zeros(T, 0)
    mu       = zeros(T, 0)
    gram     = _gram(nodes, d_nodes, d_dirs, kernel)
    chol     = cholesky(gram)
    cond     = _estimate_cond(gram, chol)

    return NormalSpline(kernel, nodes, values, d_nodes, d_dirs, d_values, mu, gram, chol, cond, min_bound, max_bound, scale)
end

function _construct!(
        spline::NormalSpline{n,T,RK},
        values::AbstractVector{T},
        d_values::AbstractVector{T},
    ) where {n, T, RK <: ReproducingKernel_0}
    n₁ = length(values)
    n₂ = length(d_values)
    length(spline._nodes) != n₁ && error("Number of data values ($n₁) does not correspond to the number of nodes $(length(spline._nodes)).")
    length(spline._d_nodes) != n₂ && error("Number of derivative values ($n₂) does not correspond to the number of derivative nodes.")
    size(spline._chol) != (n₁+n₂, n₁+n₂) && error("Number of data and derivative values ($(n₁+n₂)) do not correspond to the size of the Gram matrix ($(size(spline._chol))).")

    # Resize buffers
    resize!(spline._values, n₁)
    resize!(spline._d_nodes, n₂)
    resize!(spline._d_dirs, n₂)
    resize!(spline._d_values, n₂)
    resize!(spline._mu, n₁+n₂)

    # Copy values to avoid aliasing
    spline._values .= values

    # Nodes scaled down by `_scale` -> directional derivative scaled up by `_scale`; allocate new array to avoid aliasing
    spline._d_values .= spline._scale .* d_values

    # Compute spline coefficients and construct spline
    ldiv!(spline._mu, spline._chol, [spline._values; spline._d_values])

    return spline
end

@inline function _evaluate!(
        spline_values::AbstractArray{<:Any,N},
        spline::NormalSpline{n,<:Any,<:ReproducingKernel_0},
        points::AbstractArrOfSVecs{n,<:Any,N},
    ) where {n, N}
    @inbounds for i in 1:length(points)
        spline_values[i] = _evaluate(spline, points[i])
    end
    return spline_values
end

@inline function _evaluate(
        spline::NormalSpline{n,<:Any,RK},
        x::SVector{n},
    ) where {n, RK <: ReproducingKernel_0}
    @unpack _kernel, _nodes, _d_nodes, _d_dirs, _mu = spline
    n₁ = length(_nodes)
    n₂ = length(_d_nodes)
    x  = _normalize(spline, x)
    v  = zero(promote_type(eltype(spline), eltype(_kernel), eltype(x)))
    @inbounds for i in 1:n₁
        v += _mu[i] * _rk(_kernel, x, _nodes[i])
    end
    if RK <: ReproducingKernel_1
        @inbounds for i in 1:n₂
            v += _mu[i+n₁] * _∂rk_∂e(_kernel, x, _d_nodes[i], _d_dirs[i])
        end
    end
    return v
end

@inline function _evaluate_gradient(
        spline::NormalSpline{n,<:Any,RK},
        x::SVector{n},
    ) where {n, RK <: ReproducingKernel_0}
    @unpack _kernel, _nodes, _d_nodes, _d_dirs, _mu, _scale = spline
    n₁ = length(_nodes)
    n₂ = length(_d_nodes)
    x  = _normalize(spline, x)
    ∇  = zero(SVector{n,promote_type(eltype(spline), eltype(_kernel), eltype(x))})
    @inbounds for i in 1:n₁
        ∇ += _mu[i] * _∂rk_∂η(_kernel, x, _nodes[i])
    end
    if RK <: ReproducingKernel_1
        @inbounds for i in 1:n₂
            ∇² = _∂²rk_∂η∂ξ(_kernel, x, _d_nodes[i])
            ∇ += _mu[i+n₁] * (∇² * _d_dirs[i])
        end
    end
    return ∇ ./ _scale
end
