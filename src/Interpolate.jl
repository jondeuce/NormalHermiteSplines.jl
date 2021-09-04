function _prepare(nodes::AbstractVecOfSVecs, kernel::ReproducingKernel_0)
    min_bound, max_bound, scale = _normalization_scaling(nodes)
    nodes = _normalize.(nodes, (min_bound,), (max_bound,), scale)

    if kernel.ε == 0
        ε  = _estimate_ε(nodes)
        ε *= _ε_factor(kernel, ε)
        kernel = typeof(kernel)(ε)
    end

    gram = _gram(nodes, kernel)
    chol = cholesky(gram)
    cond = _estimate_cond(gram, chol)

    return NormalSpline(
        kernel,
        nodes,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        gram,
        chol,
        cond,
        min_bound,
        max_bound,
        scale,
    )
end

function _construct(
        spline::NormalSpline{n,T,RK},
        values::AbstractVector{T},
    ) where {n,T,RK <: ReproducingKernel_0}
    length(values) != length(spline._nodes) && error("Number of data values does not correspond to the number of nodes.")
    isnothing(spline._chol) && error("Gram matrix was not factorized.")

    # Copy values to avoid aliasing
    values = copy(values)

    # Compute spline coefficients and construct spline
    mu = spline._chol \ values

    return NormalSpline(
        spline._kernel,
        spline._nodes,
        values,
        nothing,
        nothing,
        nothing,
        mu,
        spline._gram,
        spline._chol,
        spline._cond,
        spline._min_bound,
        spline._max_bound,
        spline._scale,
    )
end

###################

function _prepare(nodes::AbstractVecOfSVecs, d_nodes::AbstractVecOfSVecs, es::AbstractVecOfSVecs, kernel::ReproducingKernel_1)
    # Normalize inputs, making copies in the process to avoid aliasing
    min_bound, max_bound, scale = _normalization_scaling(nodes, d_nodes)
    nodes = _normalize.(nodes, (min_bound,), (max_bound,), scale)
    d_nodes = _normalize.(d_nodes, (min_bound,), (max_bound,), scale)
    es = es ./ norm.(es)

    if kernel.ε == 0
        ε = _estimate_ε(nodes, d_nodes)
        ε *= _ε_factor_d(kernel, ε)
        kernel = typeof(kernel)(ε)
    end

    gram = _gram(nodes, d_nodes, es, kernel)
    chol = cholesky(gram)
    cond = _estimate_cond(gram, chol)

    return NormalSpline(
        kernel,
        nodes,
        nothing,
        d_nodes,
        es,
        nothing,
        nothing,
        gram,
        chol,
        cond,
        min_bound,
        max_bound,
        scale,
    )
end

function _construct(
        spline::NormalSpline{n,T,RK},
        values::AbstractVector{T},
        d_values::AbstractVector{T},
    ) where {n,T,RK <: ReproducingKernel_0}
    length(values) != length(spline._nodes) && error("Number of data values does not correspond to the number of nodes.")
    length(d_values) != length(spline._d_nodes) && error("Number of derivative values does not correspond to the number of derivative nodes.")
    isnothing(spline._chol) && error("Gram matrix was not factorized.")

    # Copy values to avoid aliasing
    values = copy(values)

    # Nodes scaled down by `_scale` -> directional derivative scaled up by `_scale`; allocate new array to avoid aliasing
    d_values = spline._scale .* d_values

    # Compute spline coefficients and construct spline
    mu = spline._chol \ [values; d_values]

    return NormalSpline(
        spline._kernel,
        spline._nodes,
        values,
        spline._d_nodes,
        spline._es,
        d_values,
        mu,
        spline._gram,
        spline._chol,
        spline._cond,
        spline._min_bound,
        spline._max_bound,
        spline._scale,
    )
end

@inline function _evaluate!(
        spline_values::AbstractVector,
        spline::NormalSpline{n, <:Any, <:ReproducingKernel_0},
        points::AbstractVecOfSVecs{n},
    ) where {n}
    @inbounds for i in 1:length(points)
        spline_values[i] = _evaluate(spline, points[i])
    end
    return spline_values
end

@inline function _evaluate(
        spline::NormalSpline{n, <:Any, <:ReproducingKernel_0},
        point::SVector{n},
    ) where {n}
    point = _normalize(spline, point)
    if isnothing(spline._d_nodes)
        _evaluate(point, spline._nodes, spline._mu, spline._kernel)
    else
        _evaluate(point, spline._nodes, spline._d_nodes, spline._es, spline._mu, spline._kernel)
    end
end

@inline function _evaluate(
        point::SVector,
        nodes::AbstractVecOfSVecs,
        mu::AbstractVector,
        kernel::ReproducingKernel_0,
    )
    T = promote_type(typeof(kernel.ε), eltype(point), eltype(eltype(nodes)), eltype(mu))
    v = zero(T)
    @inbounds for i in 1:length(nodes)
        v += mu[i] * _rk(kernel, point, nodes[i])
    end
    return v
end

@inline function _evaluate(
        point::SVector,
        nodes::AbstractVecOfSVecs,
        d_nodes::AbstractVecOfSVecs,
        es::AbstractVecOfSVecs,
        mu::AbstractVector,
        kernel::ReproducingKernel_1,
    )
    n_1 = length(nodes)
    v = _evaluate(point, nodes, mu, kernel)
    @inbounds for i in 1:length(d_nodes)
        v += mu[i+n_1] * _∂rk_∂e(kernel, point, d_nodes[i], es[i])
    end
    return v
end

@inline function _evaluate_gradient(
        spline::NormalSpline{n, <:Any, <:ReproducingKernel_0},
        point::SVector{n},
    ) where {n}
    point = _normalize(spline, point)
    if isnothing(spline._d_nodes)
        _evaluate_gradient(point, spline._nodes, spline._mu, spline._kernel) ./ spline._scale
    else
        _evaluate_gradient(point, spline._nodes, spline._d_nodes, spline._es, spline._mu, spline._kernel) ./ spline._scale
    end
end

@inline function _evaluate_gradient(
        point::SVector{n},
        nodes::AbstractVecOfSVecs{n},
        mu::AbstractVector,
        kernel::ReproducingKernel_0,
    ) where {n}
    T = promote_type(typeof(kernel.ε), eltype(point), eltype(eltype(nodes)), eltype(mu))
    ∇ = zero(SVector{n,T})
    @inbounds for i in 1:length(nodes)
        μ    = mu[i]
        node = nodes[i]
        ∇   += μ * _∂rk_∂η(kernel, point, node)
    end
    return ∇
end

@inline function _evaluate_gradient(
        point::SVector{n},
        nodes::AbstractVecOfSVecs{n},
        d_nodes::AbstractVecOfSVecs{n},
        es::AbstractVecOfSVecs{n},
        mu::AbstractVector,
        kernel::ReproducingKernel_1,
    ) where {n}
    n_1 = length(nodes)
    ∇ = _evaluate_gradient(point, nodes, mu, kernel)
    @inbounds for i in 1:length(d_nodes)
        μi    = mu[n_1 + i]
        ∂node = d_nodes[i]
        êi    = es[i]
        ∂²rk  = _∂²rk_∂η∂ξ(kernel, point, ∂node)
        ∇    += μi * (∂²rk * êi)
    end
    return ∇
end
