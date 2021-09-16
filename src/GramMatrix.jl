#### Build full Gram matrix (ReproducingKernel_0)

function _gram!(
        A::AbstractMatrix,
        nodes::AbstractVecOfSVecs,
        kernel::ReproducingKernel_0,
    )
    n₁ = length(nodes)
    @inbounds for j in 1:n₁
        for i in 1:j
            A[i,j] = _rk(kernel, nodes[i], nodes[j])
        end
    end
    return Hermitian(A, :U)
end

function _gram(
        nodes::AbstractVecOfSVecs,
        kernel::ReproducingKernel_0,
    )
    n₁ = length(nodes)
    T  = eltype(eltype(nodes))
    _gram!(zeros(T, n₁, n₁), nodes, kernel)
end

#### Incrementally add to Gram matrix (ReproducingKernel_0)

function _gram!(
        A::AbstractMatrix,
        new_node::SVector,
        curr_nodes::AbstractVecOfSVecs,
        kernel::ReproducingKernel_0,
    )
    n₁ = length(curr_nodes)
    @inbounds for i in 1:n₁
        A[i,end] = _rk(kernel, curr_nodes[i], new_node)
    end
    @inbounds A[end,end] = _rk(kernel, new_node, new_node)
    return Hermitian(A, :U)
end

#### Build full Gram matrix (ReproducingKernel_1)

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

#### Elastic Cholesky

Base.@kwdef struct ElasticCholesky{T} <: AbstractMatrix{T}
    maxsize::Int
    A::Matrix{T} = zeros(T, maxsize, maxsize)
    U::Matrix{T} = zeros(T, maxsize, maxsize)
    buf::Vector{T} = zeros(T, maxsize)
    perms::Vector{Int} = zeros(Int, maxsize)
    ncols::Base.RefValue{Int} = Ref(0)
end
Base.eltype(::ElasticCholesky{T}) where {T} = T
Base.parent(C::ElasticCholesky) = C.A
Base.size(C::ElasticCholesky) = (C.ncols[], C.ncols[])

function LinearAlgebra.ldiv!(y::AbstractVector{T}, C::ElasticCholesky{T}, b::AbstractVector{T}) where {T}
    @unpack U, buf, perms, ncols = C
    J = uview(perms, 1:ncols[])
    U = UpperTriangular(uview(U, J, J))
    tmp = uview(buf, 1:ncols[])
    ldiv!(tmp, U', b)
    ldiv!(y, U, tmp)
    return y
end

"""
    _insert_column!(C::ElasticCholesky, v::AbstractVector{T}) where {T}

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
function _insert_column!(f!, C::ElasticCholesky{T}, j::Int) where {T}
    @unpack maxsize, A, U, perms, ncols = C
    @assert j ∉ perms
    @inbounds if ncols[] == 0
        # Initialize first entry of A
        perms[1] = j
        f!(uview(A, j:j, j:j))
        U[j,j] = sqrt(A[j,j])
        ncols[] = 1
    else
        # Update A with new column
        perms[ncols[]+1] = j
        J = uview(perms, 1:ncols[]+1)
        f!(uview(A, J, J))

        # Update U with new column
        J = uview(perms, 1:ncols[])
        d = uview(A, J, j)
        γ = A[j,j]
        e = uview(U, J, j)
        Uᵀ = UpperTriangular(uview(U, J, J))'
        ldiv!(e, Uᵀ, d)
        α = √max(γ - e⋅e, zero(T))
        U[j,j] = max(α, eps(T))

        # Update indices
        ncols[] += 1
    end
    return C
end

# Insert `v` into column `j` of `C.A` and update the corresponding factorization `C.U`
function _insert_column!(C::ElasticCholesky{T}, j::Int, v::AbstractVector{T}) where {T}
    @unpack maxsize, ncols = C
    @assert length(v) == ncols[]+1 <= maxsize
    _insert_column!(C, j) do A
        @inbounds for i in 1:size(A, 1)
            A[i,end] = v[i]
        end
    end
end

# Update column `j` of the factorization `C.U`, assuming the corresponding column `j` of `C.A` has already been updated
function _factorize_column!(C::ElasticCholesky, j::Int)
    _insert_column!(C, j) do _
        nothing
    end
end

using Random
function test_elastic_cholesky(; maxsize = 3)
    A = rand(maxsize, maxsize)
    A = A'A
    C = ElasticCholesky{Float64}(; maxsize = maxsize)
    perms = randperm(maxsize) #1:maxsize #
    for j in 1:maxsize
        jperm = perms[j]
        v = A[1:j,j]
        @time _insert_column!(C, jperm, v)

        J = perms[1:C.ncols[]]
        @assert Hermitian(C.A[J, J], :U) ≈ A[1:j, 1:j]
        @assert UpperTriangular(C.U[J, J]) ≈ cholesky(A[1:j, 1:j]).U
        @assert C.perms[1:C.ncols[]] == J
        @assert C.ncols[] == j

        b = rand(j)
        y = similar(b)
        @time ldiv!(y, C, b)
        @assert y ≈ cholesky(A[1:j, 1:j]) \ b
    end
    return C, A
end
