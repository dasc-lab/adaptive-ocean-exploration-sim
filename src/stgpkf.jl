module STGPKF # spatiotemporal-Gaussian-process Kalman filter

using LinearAlgebra

using ..JordanLakeDomain

M = length(𝐬)

mutable struct KeyParameters
    λₛ::Float64 # 1 / km
    σ::Float64
    λₜ::Float64 # 1 / s
    σᵣ::Float64
    const Δt::Int64
end

mutable struct Estimator
    𝐯ʰᵃᵗⱼₗᵢ::Array{Float64,2} # 𝐯ʰᵃᵗⱼₗᵢ[:, 1] is 𝐯ʰᵃᵗᵢₗᵢ and 𝐯ʰᵃᵗⱼₗᵢ[:, 2] is 𝐯ʰᵃᵗ₍ᵢ₊₁₎ₗᵢ
    𝚺ⱼₗᵢ::Array{Float64,3} # 𝚺ⱼₗᵢ[:, :, 1] is 𝚺ᵢₗᵢ and 𝚺ⱼₗᵢ[:, 2] is 𝚺ᵢ₊₁ₗᵢ
    𝐱ʰᵃᵗⱼₗᵢ::Array{Float64,2} # 𝐱ʰᵃᵗⱼₗᵢ[:, 1] is 𝐱ʰᵃᵗᵢₗᵢ and 𝐱ʰᵃᵗⱼₗᵢ[:, 2] is 𝐱ʰᵃᵗ₍ᵢ₊₁₎ₗᵢ
    𝚺ⱼₗᵢ_𝐱ʰᵃᵗ::Array{Float64,3} # error covariance matrices for 𝐱ʰᵃᵗ

    Estimator(𝐯ʰᵃᵗⱼₗᵢ=zeros(M, 2), 𝚺ⱼₗᵢ=zeros(M, M, 2),
        𝐱ʰᵃᵗⱼₗᵢ=zeros(M, 2),
        𝚺ⱼₗᵢ_𝐱ʰᵃᵗ=zeros(M, M, 2)) = new(𝐯ʰᵃᵗⱼₗᵢ, 𝚺ⱼₗᵢ, 𝐱ʰᵃᵗⱼₗᵢ, 𝚺ⱼₗᵢ_𝐱ʰᵃᵗ)
end

function kₛ(key_parameters::KeyParameters, 𝐬₁, 𝐬₂)
    exp(-key_parameters.λₛ * norm(𝐬₁ - 𝐬₂))
end

function kₜ(key_parameters::KeyParameters, t₁, t₂)
    key_parameters.σ^2 * exp(-key_parameters.λₜ * (t₁ - t₂))
end

function 𝐊ₘ(key_parameters::KeyParameters)
    reshape([kₛ(key_parameters, 𝐬[j], 𝐬[i]) for i = 1:M for j = 1:M], (M, M))
end

function 𝐊ₘʰᵃˡᶠ(key_parameters::KeyParameters)
    cholesky(𝐊ₘ(key_parameters)).L
end

function 𝐊ₘ⁻¹(key_parameters::KeyParameters)
    inv(𝐊ₘ(key_parameters))
end

function Σ₀(key_parameters::KeyParameters)
    1 / (2 * key_parameters.λₜ)
end

function F̄(key_parameters::KeyParameters)
    exp(-key_parameters.λₜ * key_parameters.Δt)
end

function 𝐀(key_parameters::KeyParameters)
    Diagonal(F̄(key_parameters) * ones(M))
end

function 𝐇ᵇᵃʳ(key_parameters::KeyParameters)
    Diagonal(key_parameters.σ * sqrt(2 * key_parameters.λₜ) * ones(M))
end

function 𝐐(key_parameters::KeyParameters)
    Diagonal(1 / (2 * key_parameters.λₜ) * (1 - exp(-2 * key_parameters.λₜ * key_parameters.Δt)) * ones(M))
end

function initialize!(estimator::Estimator, key_parameters::KeyParameters)
    # Update
    estimator.𝐯ʰᵃᵗⱼₗᵢ[:, 1] = zeros(M)
    estimator.𝚺ⱼₗᵢ[:, :, 1] = Diagonal(Σ₀(key_parameters) * ones(M))
    estimator.𝐱ʰᵃᵗⱼₗᵢ[:, 1] = (𝐊ₘʰᵃˡᶠ(key_parameters)
                               * 𝐇ᵇᵃʳ(key_parameters)
                               * estimator.𝐯ʰᵃᵗⱼₗᵢ[:, 1])
    estimator.𝚺ⱼₗᵢ_𝐱ʰᵃᵗ[:, :, 1] = ((𝐊ₘʰᵃˡᶠ(key_parameters)
                                     *
                                     𝐇ᵇᵃʳ(key_parameters))
                                    * estimator.𝚺ⱼₗᵢ[:, :, 1]
                                    * transpose(𝐊ₘʰᵃˡᶠ(key_parameters)
                                                *
                                                𝐇ᵇᵃʳ(key_parameters)))
    # Predict
    estimator.𝐯ʰᵃᵗⱼₗᵢ[:, 2] = 𝐀(key_parameters) * estimator.𝐯ʰᵃᵗⱼₗᵢ[:, 1]
    estimator.𝚺ⱼₗᵢ[:, :, 2] = (𝐀(key_parameters) * estimator.𝚺ⱼₗᵢ[:, :, 1]
                               * transpose(𝐀(key_parameters))
                               +
                               𝐐(key_parameters))
    estimator.𝐱ʰᵃᵗⱼₗᵢ[:, 2] = (𝐊ₘʰᵃˡᶠ(key_parameters)
                               * 𝐇ᵇᵃʳ(key_parameters)
                               * estimator.𝐯ʰᵃᵗⱼₗᵢ[:, 2])
    estimator.𝚺ⱼₗᵢ_𝐱ʰᵃᵗ[:, :, 2] = ((𝐊ₘʰᵃˡᶠ(key_parameters)
                                     *
                                     𝐇ᵇᵃʳ(key_parameters))
                                    * estimator.𝚺ⱼₗᵢ[:, :, 2]
                                    * transpose(𝐊ₘʰᵃˡᶠ(key_parameters)
                                                *
                                                𝐇ᵇᵃʳ(key_parameters)))
end

function update_and_predict!(estimator::Estimator,
    key_parameters::KeyParameters, yᵢ::Float64, 𝐬ᵢᵗⁱˡᵈᵉ::Vector{Float64})
    # Update
    𝐇ᵢᵗⁱˡᵈᵉ = reshape([kₛ(key_parameters, 𝐬ᵢᵗⁱˡᵈᵉ, 𝐬[j]) for j = 1:M], (1, M)) * 𝐊ₘ⁻¹(key_parameters)
    𝐂ᵢ = 𝐇ᵢᵗⁱˡᵈᵉ * 𝐊ₘʰᵃˡᶠ(key_parameters) * 𝐇ᵇᵃʳ(key_parameters)
    𝐑ᵢ = Diagonal(key_parameters.σᵣ * ones(length(yᵢ)))
    𝐋ᵢ = (estimator.𝚺ⱼₗᵢ[:, :, 2] * transpose(𝐂ᵢ)
          * inv(𝐂ᵢ * estimator.𝚺ⱼₗᵢ[:, :, 2] * transpose(𝐂ᵢ) + 𝐑ᵢ))
    estimator.𝐯ʰᵃᵗⱼₗᵢ[:, 1] = (estimator.𝐯ʰᵃᵗⱼₗᵢ[:, 2]
                               +
                               𝐋ᵢ * (yᵢ - (𝐂ᵢ*estimator.𝐯ʰᵃᵗⱼₗᵢ[:, 2])[1]))
    estimator.𝚺ⱼₗᵢ[:, :, 1] = (estimator.𝚺ⱼₗᵢ[:, :, 2]
                               -
                               𝐋ᵢ * 𝐂ᵢ * estimator.𝚺ⱼₗᵢ[:, :, 2])
    estimator.𝐱ʰᵃᵗⱼₗᵢ[:, 1] = (𝐊ₘʰᵃˡᶠ(key_parameters)
                               * 𝐇ᵇᵃʳ(key_parameters)
                               * estimator.𝐯ʰᵃᵗⱼₗᵢ[:, 1])
    estimator.𝚺ⱼₗᵢ_𝐱ʰᵃᵗ[:, :, 1] = ((𝐊ₘʰᵃˡᶠ(key_parameters)
                                     *
                                     𝐇ᵇᵃʳ(key_parameters))
                                    * estimator.𝚺ⱼₗᵢ[:, :, 1]
                                    * transpose(𝐊ₘʰᵃˡᶠ(key_parameters)
                                                *
                                                𝐇ᵇᵃʳ(key_parameters)))
    # Predict
    estimator.𝐯ʰᵃᵗⱼₗᵢ[:, 2] = 𝐀(key_parameters) * estimator.𝐯ʰᵃᵗⱼₗᵢ[:, 1]
    estimator.𝚺ⱼₗᵢ[:, :, 2] = (𝐀(key_parameters) * estimator.𝚺ⱼₗᵢ[:, :, 1]
                               * transpose(𝐀(key_parameters))
                               +
                               𝐐(key_parameters))
    estimator.𝐱ʰᵃᵗⱼₗᵢ[:, 2] = (𝐊ₘʰᵃˡᶠ(key_parameters)
                               * 𝐇ᵇᵃʳ(key_parameters)
                               * estimator.𝐯ʰᵃᵗⱼₗᵢ[:, 2])
    estimator.𝚺ⱼₗᵢ_𝐱ʰᵃᵗ[:, :, 2] = ((𝐊ₘʰᵃˡᶠ(key_parameters)
                                     *
                                     𝐇ᵇᵃʳ(key_parameters))
                                    * estimator.𝚺ⱼₗᵢ[:, :, 2]
                                    * transpose(𝐊ₘʰᵃˡᶠ(key_parameters)
                                                *
                                                𝐇ᵇᵃʳ(key_parameters)))
end

end
