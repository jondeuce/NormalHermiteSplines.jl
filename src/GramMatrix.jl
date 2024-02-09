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

#### Incrementally add column to Gram matrix (ReproducingKernel_0)

function _gram!(
        A::AbstractMatrix,
        new_node::SVector,
        curr_nodes::AbstractVecOfSVecs,
        kernel::ReproducingKernel_0,
    )
    n₁ = length(curr_nodes)
    @inbounds for i in 1:n₁
        A[i, n₁+1] = _rk(kernel, curr_nodes[i], new_node)
    end
    @inbounds A[n₁+1, n₁+1] = _rk(kernel, new_node, new_node)
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
    A12 = uview(A, 1    : n₁,    n₁+1 : n₁+n₂)
    A22 = uview(A, n₁+1 : n₁+n₂, n₁+1 : n₁+n₂)

    @inbounds for j in 1:n₁
        # Top-left block (n₁ × n₁)
        for i in 1:j
            A11[i,j] = _rk(kernel, nodes[i], nodes[j])
        end
    end

    ε² = kernel.ε^2
    @inbounds for j in 1:n₂
        # Top-right block (n₁ × n₂)
        for i in 1:n₁
            A12[i,j] = _∂rk_∂e(kernel, nodes[i], d_nodes[j], d_dirs[j])
        end

        # Bottom-right block (n₂ × n₂)
        for i in 1:j-1
            A22[i,j] = _∂²rk_∂²e(kernel, d_nodes[j], d_nodes[i], d_dirs[j], d_dirs[i])
        end
        A22[j,j] = ε²
    end

    return Hermitian(A, :U)
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

#### Incrementally add column to Gram matrix (ReproducingKernel_1)

function _gram!(
        A::AbstractMatrix,
        new_node::SVector{n},
        curr_nodes::AbstractVecOfSVecs{n},
        curr_d_nodes::AbstractVecOfSVecs{n},
        curr_d_dirs::AbstractVecOfSVecs{n},
        kernel::ReproducingKernel_1,
    ) where {n}
    n₁ = length(curr_nodes)
    n₂ = length(curr_d_nodes)
    @assert size(A) == (n₁+1+n₂, n₁+1+n₂)

    # Top-left block (n₁+1 × n₁+1), right column (n₁+1 terms)
    @inbounds for i in 1:n₁
        A[i, n₁+1] = _rk(kernel, new_node, curr_nodes[i])
    end
    @inbounds A[n₁+1, n₁+1] = _rk(kernel, new_node, new_node)

    # Top-right block (n₁+1 × n₂), bottom row (n₂ terms)
    @inbounds for j in 1:n₂
        A[n₁+1, n₁+1+j] = _∂rk_∂e(kernel, new_node, curr_d_nodes[j], curr_d_dirs[j])
    end

    return Hermitian(A, :U)
end

function _gram!(
        A::AbstractMatrix,
        d_node::SVector{n},
        d_dir::SVector{n},
        curr_nodes::AbstractVecOfSVecs{n},
        curr_d_nodes::AbstractVecOfSVecs{n},
        curr_d_dirs::AbstractVecOfSVecs{n},
        kernel::ReproducingKernel_1,
    ) where {n}
    n₁ = length(curr_nodes)
    n₂ = length(curr_d_nodes)
    @assert size(A) == (n₁+n₂+1, n₁+n₂+1)

    # Top-right block, (n₁ × n₂+1), right column (n₁ terms)
    @inbounds for i in 1:n₁
        A[i, n₁+n₂+1] = _∂rk_∂e(kernel, curr_nodes[i], d_node, d_dir)
    end

    # Bottom-right block (n₂+1 × n₂+1), right column (n₂+1 terms)
    ε² = kernel.ε^2
    @inbounds for i in 1:n₂
        A[n₁+i, n₁+n₂+1] = _∂²rk_∂²e(kernel, d_node, curr_d_nodes[i], d_dir, curr_d_dirs[i])
    end
    @inbounds A[n₁+n₂+1, n₁+n₂+1] = ε²

    return Hermitian(A, :U)
end

#### Elastic Cholesky

Base.@kwdef struct ElasticCholesky{T, AType <: AbstractMatrix{T}} <: Factorization{T}
    maxcols::Int
    ncols::Base.RefValue{Int} = Ref(0)
    colperms::Vector{Int}     = zeros(Int, maxcols)
    A::AType                  = zeros(T, maxcols, maxcols)
    U::Matrix{T}              = zeros(T, maxcols, maxcols)
    U⁻ᵀb::Vector{T}           = zeros(T, maxcols)
end
ElasticCholesky{T}(maxcols::Int) where {T} = ElasticCholesky{T,Matrix{T}}(; maxcols = maxcols)
ElasticCholesky(A::AbstractMatrix{T}) where {T} = ElasticCholesky{T,typeof(A)}(; maxcols = size(A,2), A = A)

Base.eltype(::ElasticCholesky{T}) where {T} = T
Base.size(C::ElasticCholesky) = (C.ncols[], C.ncols[])
Base.parent(C::ElasticCholesky) = C.A
Base.empty!(C::ElasticCholesky) = (C.ncols[] = 0; C)
Base.show(io::IO, mime::MIME"text/plain", C::ElasticCholesky{T}) where {T} = (print(io, "ElasticCholesky{T}\nU factor:\n"); show(io, mime, UpperTriangular(C.U[C.colperms[1:C.ncols[]], C.colperms[1:C.ncols[]]])))

function LinearAlgebra.ldiv!(x::AbstractVector{T}, C::ElasticCholesky{T}, b::AbstractVector{T}, ::Val{permview} = Val(false)) where {T, permview}
    (; U, U⁻ᵀb, colperms, ncols) = C
    J = uview(colperms, 1:ncols[])
    U = UpperTriangular(uview(U, J, J))
    U⁻ᵀb = uview(U⁻ᵀb, 1:ncols[])
    if permview
        x = uview(x, J)
        b = uview(b, J)
    end
    ldiv!(U⁻ᵀb, U', b)
    ldiv!(x, U, U⁻ᵀb)
    return x
end

function Base.insert!(C::ElasticCholesky{T}, j::Int, B::AbstractMatrix{T}) where {T}
    (; A, colperms, ncols) = C
    @inbounds colperms[ncols[] + 1] = j
    rows = uview(colperms, 1 : ncols[] + 1)
    @inbounds for i in rows
        A[i,j] = B[i,j]
    end
    return C
end

"""
    LinearAlgebra.cholesky!(C::ElasticCholesky, v::AbstractVector{T}) where {T}

Update the Cholesky factorization `C` as if the column `v` (and by symmetry, the corresponding row `vᵀ`)
were inserted into the underlying matrix `A`. Specifically, let `L` be the lower-triangular cholesky factor
of `A` such that `A = LLᵀ`, and let `v = [d; γ]` such that the new matrix `A⁺` is given by

```
A⁺ = [A  d]
     [dᵀ γ].
```

Then, the corresponding updated cholesky factor `L⁺` of `⁺` is:

```
L⁺ = [L  e]
     [eᵀ α]
```

where `e = L⁻¹d`, `α = √τ`, and `τ = γ - e⋅e > 0`. If `τ ≤ 0`, then `A⁺` is not positive definite.

See:
    https://igorkohan.github.io/NormalHermiteSplines.jl/dev/Normal-Splines-Method/#Algorithms-for-updating-Cholesky-factorization
"""
function LinearAlgebra.cholesky!(
        C::ElasticCholesky{T},
        j::Int,
        v::AbstractVector{T},
        ::Val{fill_parent},
    ) where {T, fill_parent}
    (; maxcols, A, U, colperms, ncols) = C
    @assert length(v) == ncols[] + 1 <= maxcols

    @inbounds if ncols[] == 0
        # Initialize first entry of `A`
        colperms[1] = j
        if fill_parent
            A[j,j] = v[1]
        end
        U[j,j] = sqrt(v[1])
        ncols[] = 1
    else
        # Fill `A` with new column
        colperms[ncols[] + 1] = j
        if fill_parent
            rows = uview(colperms, 1:ncols[] + 1)
            copyto!(uview(A, rows, j), v)
        end

        # Update `U` with new column
        J = uview(colperms, 1:ncols[])
        d = uview(A, J, j)
        γ = A[j,j]
        e = uview(U, J, j)
        Uᵀ = UpperTriangular(uview(U, J, J))'
        ldiv!(e, Uᵀ, d)
        τ = γ - e⋅e
        α = √max(τ, 0) # `τ` should be positive by construction
        U[j,j] = max(α, eps(T)) # if `α < ϵ` you have bigger problems...

        # Increment column counter
        ncols[] += 1
    end

    return C
end

# Update the `j`th column of the factorization `C.U`, assuming the corresponding column `j` of `C.A` has been filled
function LinearAlgebra.cholesky!(C::ElasticCholesky{T}, j::Int) where {T}
    (; maxcols, A, colperms, ncols) = C
    @assert ncols[] + 1 <= maxcols

    @inbounds colperms[ncols[] + 1] = j
    rows = uview(colperms, 1:ncols[] + 1)
    v = uview(A, rows, j)
    cholesky!(C, j, v, Val(false))

    return C
end

# Update columns `J` of the factorization `C.U`, assuming the corresponding columns `J` of `C.A` have been filled
function LinearAlgebra.cholesky!(C::ElasticCholesky, J = axes(C.A, 2))
    for j in J
        cholesky!(C, j)
    end
    return C
end
