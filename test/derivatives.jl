unit_norm(x::SVector) = x / norm(x)
unit_norm(x::AbstractVector{S}) where {S <: SVector} = unit_norm.(x)
affine(x::SVector, a, b) = a .+ (b - a) .* x
affine(x::AbstractVector{S}, a, b) where {S <: SVector} = affine.(x, a, b)

function derivative_tests(::Val{n}, kernel::RK) where {n, RK <: ReproducingKernel_0}
    node_min, node_max = 0.5, 2.0
    rand_node = (dims...) -> affine(rand(SVector{n, Float64}, dims...), node_min, node_max)
    rand_dir = (dims...) -> unit_norm(randn(SVector{n, Float64}, dims...))
    rand_val = (dims...) -> randn(Float64, dims...)

    η, ξ = rand_node(2)
    û, v̂ = rand_dir(2)
    ∇ = ForwardDiff.gradient(η -> _rk(kernel, η, ξ), η)

    @test @inferred(_rk(kernel, η, ξ)) isa Float64
    @test @allocated(_rk(kernel, η, ξ)) == 0

    @test _∂rk_∂η(kernel, η, ξ) ≈ ∇
    @test @inferred(_∂rk_∂η(kernel, η, ξ)) isa SVector{n, Float64}
    @test @allocated(_∂rk_∂η(kernel, η, ξ)) == 0

    @test _∂rk_∂ηⁱ_ûᵢ(kernel, η, ξ, û) ≈ ∇ ⋅ û
    @test @inferred(_∂rk_∂ηⁱ_ûᵢ(kernel, η, ξ, û)) isa Float64
    @test @allocated(_∂rk_∂ηⁱ_ûᵢ(kernel, η, ξ, û)) == 0

    if RK <: ReproducingKernel_1
        #TODO: define second derivatives for ReproducingKernel_0?
        ∇² = ForwardDiff.jacobian(ξ -> _∂rk_∂η(kernel, η, ξ), ξ)

        @test _∂²rk_∂η∂ξ(kernel, η, ξ) ≈ ∇²
        @test @inferred(_∂²rk_∂η∂ξ(kernel, η, ξ)) isa SMatrix{n, n, Float64}
        @test @allocated(_∂²rk_∂η∂ξ(kernel, η, ξ)) == 0

        @test _∂²rk_∂ηⁱ∂ξ_ûᵢ(kernel, η, ξ, û) ≈ ∇² * û
        @test @inferred(_∂²rk_∂ηⁱ∂ξ_ûᵢ(kernel, η, ξ, û)) isa SVector{n, Float64}
        @test @allocated(_∂²rk_∂ηⁱ∂ξ_ûᵢ(kernel, η, ξ, û)) == 0

        @test _∂²rk_∂ηⁱ∂ξʲ_ûᵢ_v̂ⱼ(kernel, η, ξ, û, v̂) ≈ v̂' * ∇² * û
        @test @inferred(_∂²rk_∂ηⁱ∂ξʲ_ûᵢ_v̂ⱼ(kernel, η, ξ, û, v̂)) isa Float64
        @test @allocated(_∂²rk_∂ηⁱ∂ξʲ_ûᵢ_v̂ⱼ(kernel, η, ξ, û, v̂)) == 0
    end

    npts = 3
    nodes, values, d_nodes, d_dirs, d_values = rand_node(npts), rand_val(npts), rand_node(npts), rand_dir(npts), rand_val(npts)
    min_bound, max_bound = fill(node_min, SVector{n, Float64}), fill(node_max, SVector{n, Float64})

    spl = interpolate(nodes, values, kernel)
    spline_derivative_tests(spl)

    spl = ElasticNormalSpline(min_bound, max_bound, length(nodes), kernel)
    insert!(spl, nodes, values)
    spline_derivative_tests(spl)

    if RK <: ReproducingKernel_1
        # Adding derivative nodes requires ReproducingKernel_1
        spl = interpolate(nodes, values, d_nodes, d_dirs, d_values, kernel)
        spline_derivative_tests(spl)

        spl = ElasticNormalSpline(min_bound, max_bound, length(nodes), kernel)
        insert!(spl, nodes, values, d_nodes, d_dirs, d_values)
        spline_derivative_tests(spl)
    end
end

function spline_derivative_tests(spl::AbstractNormalSpline{n, Float64, RK}) where {n, RK <: ReproducingKernel_0}
    x = randn(SVector{n, Float64})
    @test @inferred(_evaluate(spl, x)) isa Float64
    @test @allocated(_evaluate(spl, x)) == 0

    @test _evaluate_gradient(spl, x) ≈ ForwardDiff.gradient(x -> _evaluate(spl, x), x)
    @test @inferred(_evaluate_gradient(spl, x)) isa SVector{n, Float64}
    @test @allocated(_evaluate_gradient(spl, x)) == 0

    y, dy = _evaluate_with_gradient(spl, x)
    @test y ≈ _evaluate(spl, x)
    @test dy ≈ _evaluate_gradient(spl, x)
    @test @inferred(_evaluate_with_gradient(spl, x)) isa Tuple{Float64, SVector{n, Float64}}
    @test @allocated(_evaluate_with_gradient(spl, x)) == 0
end

@testset "Derivatives: order = $n, kernel = $(nameof(typeof(kernel)))" for n in (1, 2, 3), kernel in (RK_H0(rand(Float64)), RK_H1(rand(Float64)), RK_H2(rand(Float64)))
    derivative_tests(Val(n), kernel)
end
