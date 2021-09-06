function _gram!(
        A::AbstractMatrix,
        nodes::AbstractVecOfSVecs,
        kernel::ReproducingKernel_0,
    )
    n₁ = length(nodes)
    @inbounds for j in 1:n₁
        for i in j:n₁
            A[i,j] = _rk(kernel, nodes[i], nodes[j])
        end
    end
    return Hermitian(A, :L)
end
function _gram(
        nodes::AbstractVecOfSVecs,
        kernel::ReproducingKernel_0,
    )
    n₁ = length(nodes)
    T  = eltype(eltype(nodes))
    _gram!(zeros(T, n₁, n₁), nodes, kernel)
end

function _gram!(
        A::AbstractMatrix,
        nodes::AbstractVecOfSVecs{n},
        d_nodes::AbstractVecOfSVecs{n},
        d_dirs::AbstractVecOfSVecs{n},
        kernel::ReproducingKernel_1,
    ) where {n}
    n₁  = length(nodes)
    n₂  = length(d_nodes)
    A11 = A
    A21 = uview(A, n₁+1 : n₁+n₂, 1 : n₁)
    A22 = uview(A, n₁+1 : n₁+n₂, n₁+1 : n₁+n₂)

    @inbounds for j in 1:n₁
        # Top-left block (n₁ x n₁)
        for i in j:n₁
            A11[i,j] = _rk(kernel, nodes[i], nodes[j])
        end
        # Bottom-left block (n₂ x n₁)
        for i in 1:n₂
            A21[i,j] = _∂rk_∂e(kernel, nodes[j], d_nodes[i], d_dirs[i])
        end
    end

    # Bottom-right block (n₂ x n₂)
    ε² = kernel.ε^2
    @inbounds for j in 1:n₂
        A22[j, j] = ε²
        for i in j+1:n₂
            A22[i,j] = _∂²rk_∂²e(kernel, d_nodes[j], d_nodes[i], d_dirs[j], d_dirs[i])
        end
    end

    return Hermitian(A, :L)
end
function _gram(
        nodes::AbstractVecOfSVecs{n},
        d_nodes::AbstractVecOfSVecs{n},
        d_dirs::AbstractVecOfSVecs{n},
        kernel::ReproducingKernel_1,
    ) where {n}
    n₁ = length(nodes)
    n₂ = length(d_nodes)
    T  = promote_type(eltype(eltype(nodes)), eltype(eltype(d_nodes)), eltype(eltype(d_dirs)))
    _gram!(zeros(T, n₁+n₂, n₁+n₂), nodes, d_nodes, d_dirs, kernel)
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
