import LinearAlgebra: kron

# ─────────────────────────────────────────────────────────────────────────────
# Low‑level helper – fast permutation of a 3‑tensor
# mt′[a,b,c] = mt[perm[a], perm[b], perm[c]]
function permute_mult_tab(mt::Array{Int,3}, perm::Vector{Int})
    n = length(perm)
    new = similar(mt)
    @inbounds for a in 1:n, b in 1:n, c in 1:n
        new[a, b, c] = mt[perm[a], perm[b], perm[c]]
    end
    return new
end

# ─────────────────────────────────────────────────────────────────────────────
# Helper to permute stored modular‑data dictionaries (S‑matrix & twists)
function permute_modular_data(md::Vector{Dict}, perm::Vector{Int})
    [ Dict(
        "SMatrix"      => M["SMatrix"][perm, perm],
        "TwistFactors" => M["TwistFactors"][:, perm]
      ) for M in md ]
end

# ─────────────────────────────────────────────────────────────────────────────
# 1. permute — relabel the particles of a FusionRing
function permute(r::FusionRing, perm::Vector{Int})::FusionRing
    n = rank(r)
    n == length(perm)           || throw(ArgumentError("perm length ≠ rank"))
    sort(perm) == collect(1:n)  || throw(ArgumentError("perm must be a permutation"))
    perm[1] == 1                || throw(ArgumentError("vacuum (1) must stay fixed"))

    mt_new = permute_mult_tab(multiplication_table(r), perm)

    elnames = element_names(r)[perm]
    fpdims_ = r.frobenius_perron_dimensions === missing ?
              missing : r.frobenius_perron_dimensions[perm]
    chars_  = r.characters === missing ?
              missing : r.characters[:, perm]
    md_     = r.modular_data === missing ?
              missing : permute_modular_data(r.modular_data, perm)

    fusion_ring(mt_new;
        names         = r.names,
        texnames      = r.texnames,
        element_names = elnames,
        barcode       = r.barcode,
        formal_code   = r.formal_code,
        sub_fusion_rings         = r.sub_fusion_rings,
        frobenius_perron_dimensions = fpdims_,
        modular_data  = md_,
        characters    = chars_
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# 2. permutation vectors for canonical orderings
function perm_vec_qd(r::FusionRing; order::Symbol = :increasing)::Vector{Int}
    qd = fpdims(r)
    idx = collect(2:rank(r))
    sort!(idx; by = i -> qd[i], rev = (order == :decreasing))
    return vcat(1, idx)
end

function perm_vec_sd_conj(r::FusionRing)::Vector{Int}
    cm = conjugation_matrix(r)
    qd = fpdims(r)
    n  = rank(r)

    self_dual = [i for i in 2:n if cm[i,i] == 1]
    non_self  = setdiff(2:n, self_dual)

    # Map each non‑self‑dual i to its conjugate partner j
    pairs = Dict{Int,Int}()
    for i in non_self
        j = findfirst(x->x==1, cm[i,:])
        pairs[i] = j
    end

    seen = Set{Int}()
    pair_order = Int[]
    for i in sort(non_self; by = i -> qd[i])
        if !(i in seen)
            push!(pair_order, i, pairs[i])
            push!(seen, i, pairs[i])
        end
    end

    return vcat(1,
                sort(self_dual; by = i -> qd[i]),
                pair_order)
end

# ─────────────────────────────────────────────────────────────────────────────
# 3. sort — wrapper that calls permute with a canonical permutation vector
function sort(r::FusionRing; sortby::String = "fpdims", kwargs...)::FusionRing
    perm = sortby == "fpdims"      ? perm_vec_qd(r; kwargs...) :
           sortby == "sd-conj"     ? perm_vec_sd_conj(r) :
           sortby == "sd−conj"     ? perm_vec_sd_conj(r) :
           throw(ArgumentError("unknown sortby = $sortby"))
    return permute(r, perm)
end

# ─────────────────────────────────────────────────────────────────────────────
# 4. tensor_product — categorical direct product of two fusion rings
function tensor_product(r1::FusionRing, r2::FusionRing)::FusionRing
    m, n = rank(r1), rank(r2)
    mt1, mt2 = multiplication_table(r1), multiplication_table(r2)

    mt = zeros(Int, m*n, m*n, m*n)
    @inbounds for a in 1:m, α in 1:n, b in 1:m, β in 1:n, c in 1:m, γ in 1:n
        i = (a-1)*n + α
        j = (b-1)*n + β
        k = (c-1)*n + γ
        mt[i,j,k] = mt1[a,b,c] * mt2[α,β,γ]
    end

    names_ = (isempty(r1.names) || isempty(r2.names)) ? missing :
             [string(r1.names[1], " × ", r2.names[1])]

    elnames = [string(e1, "⊗", e2) for e1 in element_names(r1) for e2 in element_names(r2)]

    fpd_ = r1.frobenius_perron_dimensions === missing ||
           r2.frobenius_perron_dimensions === missing ?
           missing :
           vec(kron(r1.frobenius_perron_dimensions, r2.frobenius_perron_dimensions))

    fusion_ring(mt;
        names         = names_,
        element_names = elnames,
        frobenius_perron_dimensions = fpd_
    )
end

const ⊗ = tensor_product  # handy infix alias

# ─────────────────────────────────────────────────────────────────────────────
# 5. Place‑holders for unimplemented high‑level routines
function which_permutation(r1::FusionRing, r2::FusionRing)::Union{Vector{Int},Nothing}
    throw(ErrorException("which_permutation not implemented yet"))
end

function replace_by_known(r::FusionRing)::FusionRing
    return r   # stub
end
