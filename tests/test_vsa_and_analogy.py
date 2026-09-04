"""Tests for VSA hypervector algebra and Gentner Structure Mapping Engine."""


from super_solver.core.vsa import VSAEngine
from super_solver.frameworks.gentner_sme import RelationalStatement, StructureMappingEngine


def test_vsa_binding_and_unbinding():
    vsa = VSAEngine(dim=2048, seed=123)

    role = vsa.random_hypervector()
    filler = vsa.random_hypervector()

    # Bound representation
    bound = vsa.bind(role, filler)
    assert bound.shape == (2048,)
    # Quasi-orthogonal to both components
    assert abs(vsa.similarity(bound, role)) < 0.15
    assert abs(vsa.similarity(bound, filler)) < 0.15

    # Unbinding retrieves a noisy version of filler with high positive cosine similarity
    retrieved = vsa.unbind(bound, role)
    sim_recovered = vsa.similarity(retrieved, filler)
    assert sim_recovered > 0.40  # Well above random baseline (<0.05)


def test_vsa_analogical_solver():
    vsa = VSAEngine(dim=2048, seed=456)

    # Analogy: King : Queen :: Man : Woman
    # In VSA: King = Person + Royal + Male, Queen = Person + Royal + Female
    person = vsa.random_hypervector()
    royal = vsa.random_hypervector()
    male = vsa.random_hypervector()
    female = vsa.random_hypervector()

    king = vsa.bundle([person, royal, male])
    queen = vsa.bundle([person, royal, female])
    man = vsa.bundle([person, male])
    woman = vsa.bundle([person, female])

    # Solve King : Queen :: Man : ?
    predicted_woman = vsa.solve_analogy(king, queen, man)
    sim_to_woman = vsa.similarity(predicted_woman, woman)
    sim_to_random = vsa.similarity(predicted_woman, vsa.random_hypervector())

    assert sim_to_woman > sim_to_random + 0.20


def test_gentner_structure_mapping_engine():
    sme = StructureMappingEngine(vsa_dim=2048)

    # Base domain: Solar System (Sun attracts Planet, Sun is more massive than Planet)
    base = [
        RelationalStatement(relation="attracts", entity_a="sun", entity_b="planet", domain="solar_system"),
        RelationalStatement(relation="revolves_around", entity_a="planet", entity_b="sun", domain="solar_system"),
    ]

    # Target domain: Rutherford Atom (Nucleus attracts Electron, Electron revolves around Nucleus)
    target = [
        RelationalStatement(relation="attracts", entity_a="nucleus", entity_b="electron", domain="atomic_physics"),
        RelationalStatement(relation="revolves_around", entity_a="electron", entity_b="nucleus", domain="atomic_physics"),
    ]

    res = sme.align_systems(base, target)
    assert res["isomorphic_transfer_viable"] is True
    assert res["systematic_match_count"] == 2
    assert res["structural_alignment_score"] > 0.35
