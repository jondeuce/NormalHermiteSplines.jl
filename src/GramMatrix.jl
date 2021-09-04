function _gram!(A::AbstractMatrix, nodes::AbstractVecOfSVecs, kernel::ReproducingKernel_0)
    n_1 = length(nodes)
    @inbounds for j in 1:n_1
        for i in j:n_1
            A[i,j] = _rk(kernel, nodes[i], nodes[j])
        end
    end
    return Hermitian(A, :L)
end
function _gram(nodes::AbstractVecOfSVecs, kernel::ReproducingKernel_0)
    n_1 = length(nodes)
    T = eltype(eltype(nodes))
    _gram!(zeros(T, n_1, n_1), nodes, kernel)
end

function _gram!(
        A::AbstractMatrix,
        nodes::AbstractVecOfSVecs{n},
        d_nodes::AbstractVecOfSVecs{n},
        es::AbstractVecOfSVecs{n},
        kernel::ReproducingKernel_1,
    ) where {n}
    n_1 = length(nodes)
    n_2 = length(d_nodes)
    A11 = A
    A21 = uview(A, n_1 + 1 : n_1 + n_2, 1 : n_1)
    A22 = uview(A, n_1 + 1 : n_1 + n_2, n_1 + 1 : n_1 + n_2)

    @inbounds for j in 1:n_1
        # Top-left block (n_1 x n_1)
        for i in j:n_1
            A11[i,j] = _rk(kernel, nodes[i], nodes[j])
        end
        # Bottom-left block (n_2 x n_1)
        for i in 1:n_2
            A21[i, j] = _∂rk_∂e(kernel, nodes[j], d_nodes[i], es[i])
        end
    end

    # Bottom-right block (n_2 x n_2)
    ε² = kernel.ε^2
    @inbounds for j in 1:n_2
        A22[j, j] = ε²
        for i in j+1:n_2
            A22[i, j] = es[i] ⋅ (_∂²rk_∂η∂ξ(kernel, d_nodes[j], d_nodes[i]) * es[j])
        end
    end

    return Hermitian(A, :L)
end
function _gram(
        nodes::AbstractVecOfSVecs{n},
        d_nodes::AbstractVecOfSVecs{n},
        es::AbstractVecOfSVecs{n},
        kernel::ReproducingKernel_1,
    ) where {n}
    n_1 = length(nodes)
    n_2 = length(d_nodes)
    T = promote_type(eltype(eltype(nodes)), eltype(eltype(d_nodes)), eltype(eltype(es)))
    _gram!(zeros(T, n_1 + n_2, n_1 + n_2), nodes, d_nodes, es, kernel)
end

"""
    _insert_factor_column(C::LinearAlgebra.Cholesky{T,Matrix{T}}, v::AbstractVector{T}) where {T}

Update the Cholesky factorization `C` as if a row and column were inserted
into the underlying matrix `A`. Specifically, let `C = cholesky(A)` and

    Ã = [A  d]
        [dᵀ γ]

where `v = [d; γ]`.

The corresponding updated cholesky factorization is:

    L̃ = [L   ]
        [eᵀ α]

where `e = L⁻¹d`, `α = √τ`, and `τ = γ - e⋅e > 0`.
If `τ ≤ 0` then `Ã` is not positive definite.

See:
    https://igorkohan.github.io/NormalHermiteSplines.jl/dev/Normal-Splines-Method/#Algorithms-for-updating-Cholesky-factorization
"""
function _insert_factor_column(
        C::LinearAlgebra.Cholesky{T,Matrix{T}},
        v::AbstractVector{T},
    ) where {T}
    @assert size(C,1) == size(C,2) == length(v)-1
    m = size(C,1)
    d = @views v[1:m]
    γ = @inbounds v[m+1]
    e = C.L\d
    τ = max(γ - e⋅e, zero(T))
    α = √τ
    U = UpperTriangular([C.U e; e' α])
    return Cholesky(U, :U, 0)
end
