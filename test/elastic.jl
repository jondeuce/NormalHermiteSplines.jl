@testset "Elastic Cholesky" begin
    T = Float64
    max_size = 4
    A = rand(MersenneTwister(0), max_size, max_size)
    A = A'A

    # Test incrementally adding columns of `A` to `ElasticCholesky`
    for colperms in [[1,2,3,4], [4,3,2,1], [1,3,2,4], [4,1,3,2]]
        C_copy_into_A = ElasticCholesky{T}(max_size)
        C_wrap_A = ElasticCholesky(A)
        for j in 1:max_size
            # Add (permuted) column `colperms[j]` of `A` to the `colperms[j]`th column of `C.A`
            cholesky!(C_copy_into_A, colperms[j], A[colperms[1:j], colperms[j]], Val(true))
            cholesky!(C_wrap_A, colperms[j])

            for C in [C_copy_into_A, C_wrap_A]
                # Check the fields of `C`
                J = colperms[1:C.ncols[]]
                @test Hermitian(C.A[J, J], :U) ≈ A[J, J]
                @test UpperTriangular(C.U[J, J]) ≈ cholesky(A[J, J]).U
                @test C.colperms[1:C.ncols[]] == J
                @test C.ncols[] == j

                # Check `ldiv!` gives same result as `LinearAlgebra.cholesky`
                b = rand(j)
                @test ldiv!(similar(b), C, b) ≈ cholesky(A[J, J]) \ b
            end
        end
    end

    # Test incremental factorization of wrapped array
    C = ElasticCholesky(copy(A))
    cholesky!(C)

    @test Hermitian(C.A, :U) ≈ A
    @test UpperTriangular(C.U) ≈ cholesky(A).U
    @test C.colperms == 1:max_size
    @test C.ncols[] == max_size

    b = rand(max_size)
    @test ldiv!(similar(b), C, b) ≈ cholesky(A) \ b
end

function random_nodes(n, T, max_size)
    min_bound   = -rand(SVector{n,T})
    max_bound   = rand(SVector{n,T})
    rand_node() = min_bound .+ rand(SVector{n,T}) .* (max_bound .- min_bound)
    rand_dir()  = (x = rand_node(); x/norm(x))

    nodes     = [min_bound, max_bound, (rand_node() for i in 3:max_size)...]
    values    = rand(T, max_size)
    d_nodes   = [rand_node() for _ in 1:n*max_size]
    d_dirs    = [rand_dir() for _ in 1:n*max_size]
    d_values  = rand(T, n*max_size)
    return (; min_bound, max_bound, nodes, values, d_nodes, d_dirs, d_values)
end

@testset "Elastic Gram Matrix" begin
    nhs = NormalHermiteSplines
    max_size = 4
    T = Float64

    @testset "RK_H0" begin
        kernel = RK_H0(0.5 + rand())
        for n in 1:3
            # Test incremental building of Gram matrix, inserting nodes in random order
            (; nodes) = random_nodes(n, T, max_size)
            n₁ = length(nodes)
            A  = nhs._gram(nodes, kernel)
            A′ = zeros(T, n₁, n₁)
            J  = Int[]
            for j in randperm(n₁)
                push!(J, j)
                nhs._gram!(view(A′, J, J), nodes[J[end]], nodes[J[1:end-1]], kernel)
                @test Hermitian(A′[J,J], :U) ≈ A[J,J] # `A′[J,J]` is upper triangular, `A` is Hermitian
            end
        end
    end

    @testset "RK_H1" begin
        kernel = RK_H1(0.5 + rand())
        for n in 1:3
            # Test incremental building of Gram matrix, inserting (derivative-)nodes in random order
            (; nodes, d_nodes, d_dirs) = random_nodes(n, T, max_size)
            n₁, n₂ = length(nodes), length(d_nodes)
            A  = nhs._gram(nodes, d_nodes, d_dirs, kernel)
            A′ = zeros(T, n₁+n₂, n₁+n₂)
            J, J₁, J₂ = Int[], Int[], Int[]
            for j in randperm(n₁+n₂)
                if j <= n₁
                    push!(J₁, j)
                    insert!(J, length(J₁), j)
                    nhs._gram!(view(A′, J, J), nodes[J₁[end]], nodes[J₁[1:end-1]], d_nodes[J₂], d_dirs[J₂], kernel)
                else
                    push!(J₂, j-n₁)
                    push!(J, j)
                    nhs._gram!(view(A′, J, J), d_nodes[J₂[end]], d_dirs[J₂[end]], nodes[J₁], d_nodes[J₂[1:end-1]], d_dirs[J₂[1:end-1]], kernel)
                end
                @test Hermitian(A′[J,J], :U) ≈ A[J,J] # `A′[J,J]` is upper triangular, `A` is Hermitian
            end
        end
    end
end

@testset "Elastic Normal Spline" begin
    nhs = NormalHermiteSplines
    max_size = 3
    T = Float64

    for n in 1:3
        (; min_bound, max_bound, nodes, values, d_nodes, d_dirs, d_values) = random_nodes(n, T, max_size)
        rk_H0 = RK_H0(0.5 + rand())
        rk_H1 = RK_H1(0.5 + rand())
        espl_H0 = ElasticNormalSpline(min_bound, max_bound, max_size, rk_H0)
        espl_H1_0 = ElasticNormalSpline(min_bound, max_bound, max_size, rk_H1)
        espl_H1_1 = ElasticNormalSpline(min_bound, max_bound, max_size, rk_H1)

        for i in 1:max_size
            # Update `ElasticNormalSpline`
            nodes′, values′ = nodes[1:i], values[1:i]
            d_nodes′, d_dirs′, d_values′ = d_nodes[1:n*i], d_dirs[1:n*i], d_values[1:n*i]

            # Insert regular node
            insert!(espl_H0, nodes′[i], values′[i])
            insert!(espl_H1_0, nodes′[i], values′[i])

            # Insert `n` derivative nodes
            insert!(espl_H1_1, nodes′[i], values′[i])
            for k in n*(i-1).+(1:n)
                insert!(espl_H1_1, d_nodes′[k], d_dirs′[k], d_values′[k])
            end

            # Compute `NormalSpline`
            i == 1 && continue # `NormalSpline` requires at least two nodes′
            spl_H0  = interpolate(nodes′, values′, rk_H0)
            spl_H1_0 = interpolate(nodes′, values′, rk_H1)
            spl_H1_1 = interpolate(nodes′, values′, d_nodes′, d_dirs′, d_values′, rk_H1)

            for (espl, spl) in [
                    (espl_H0, spl_H0)
                    (espl_H1_0, spl_H1_0)
                    (espl_H1_1, spl_H1_1)
                ]
                C = espl._chol
                n₁ = espl._num_nodes[]
                J = C.colperms[1:C.ncols[]]

                # Cholesky factorization of `ElasticNormalSpline` is built incrementally in arbitrary column order of the underlying Gram matrix;
                # compare with the Cholesky factorization of the gram matrix from `NormalSpline`, permuted appropriately
                J′ = (j -> ifelse(j > max_size,  j - max_size + n₁, j)).(J)
                C′ = cholesky(nhs._get_gram(spl)[J′, J′])
                b = randn(C.ncols[])

                @test nhs._get_kernel(espl)      == nhs._get_kernel(spl)
                @test nhs._get_nodes(espl)       ≈ nhs._get_nodes(spl)
                @test nhs._get_values(espl)      ≈ nhs._get_values(spl)
                @test nhs._get_d_nodes(espl)     ≈ nhs._get_d_nodes(spl)
                @test nhs._get_d_dirs(espl)      ≈ nhs._get_d_dirs(spl)
                @test nhs._get_d_values(espl)    ≈ nhs._get_d_values(spl)
                @test nhs._get_mu(espl)          ≈ nhs._get_mu(spl)
                @test nhs._get_gram(espl)        ≈ nhs._get_gram(spl)
                @test UpperTriangular(C.U[J, J]) ≈ C′.U
                @test ldiv!(similar(b), C, b)    ≈ C′ \ b
                # @test nhs._get_cond(espl)      ≈ nhs._get_cond(spl)
                @test nhs._get_min_bound(espl)   ≈ nhs._get_min_bound(spl)
                @test nhs._get_max_bound(espl)   ≈ nhs._get_max_bound(spl)
                @test nhs._get_scale(espl)       ≈ nhs._get_scale(spl)
            end
        end
    end
end
