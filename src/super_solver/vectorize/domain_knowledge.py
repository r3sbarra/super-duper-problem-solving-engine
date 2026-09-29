"""Domain-knowledge grounding for super-solver generation.

The discovery pipeline's breakthrough step can only decode the latent state to
one of its candidate claims. When those candidates are only TRIZ principle
descriptions, the engine generates *principles*, not *concrete solutions*.

This module supplies a fact base of concrete solution directions (material
science, engineering, chemistry, biology, physics) that can be seeded into the
vector index and retrieved via analogical transfer, so the breakthrough step
can decode to a concrete, domain-grounded solution rather than a bare principle.

Each entry pairs a problem context with the concrete solution direction that
resolved it. The vector index stores these as solutions; hybrid_analogize
retrieves the most relevant ones for a new problem.
"""

from __future__ import annotations

import json
from typing import Dict, List

# ---------------------------------------------------------------------------
# Domain-knowledge fact base: (problem_context, concrete_solution_direction)
# ---------------------------------------------------------------------------
# Each is a real, documented solution direction. The problem_context is what
# gets matched against a new problem; the solution_direction is the concrete
# design/material/mechanism that resolved it.
DOMAIN_KNOWLEDGE: List[Dict[str, str]] = [
    # --- Meta-engineering / self-improvement (added after the recursive
    #     self-review showed the corpus had NO vocabulary for the engine's own
    #     class of problem, so it could not retrieve a solution for itself) ---
    {
        "problem_context": "a decomposer produces generic subproblems that do not lead to solved problems",
        "solution_direction": "Add a solution-synthesis stage that proposes a concrete mechanism for each subproblem, then close the loop with whole-problem solution retrieval",
    },
    {
        "problem_context": "retrieval cannot find a solution for an abstract or meta problem outside the corpus",
        "solution_direction": "Expand the vector corpus with solution ideas for the missing domain so retrieval generalizes beyond hand-authored cases",
    },
    {
        "problem_context": "several competing design proposals must be chosen between",
        "solution_direction": "Rank the options on impact, effort and regression risk with a multi-objective Pareto resolver instead of a pass/fail filter",
    },
    {
        "problem_context": "a subsystem passes every input and cannot discriminate good from bad",
        "solution_direction": "Add a discriminative check that the output varies across a known-good/known-bad pair, not just that it runs",
    },
    {
        "problem_context": "a component silently returns an empty or constant result",
        "solution_direction": "Fail closed with an explicit empty-result sentinel and a degeneracy guard so the caller never trusts a coin-flip",
    },
    {
        "problem_context": "an engine should improve its own reasoning substrate",
        "solution_direction": "Run a recursive self-review: introspect measured weaknesses, frame them as a problem, and feed them back through the engine's own machinery",
    },
    # --- Materials / structural ---
    {
        "problem_context": "make a dense material float or reduce its weight while keeping strength",
        "solution_direction": "Use lightweight aggregates and air-entraining agents to reduce the material's density below that of water",
    },
    {
        "problem_context": "make a material that is both elastic and moldable, deformable yet shape-holding",
        "solution_direction": "Use a viscoelastic polymer (silicone) that exhibits both viscous and elastic behavior",
    },
    {
        "problem_context": "make a surface that nothing sticks to, low friction cooking or coating",
        "solution_direction": "Apply a PTFE (Teflon) coating with extremely low surface energy and friction",
    },
    {
        "problem_context": "make a material that repairs its own cracks or damage over time",
        "solution_direction": "Embed microcapsules of healing agent or bacteria that precipitate calcium carbonate to seal cracks",
    },
    {
        "problem_context": "make a material that changes color or reacts when exposed to light or pollutants",
        "solution_direction": "Use a photocatalytic or photochromic coating that reacts to light and breaks down surface contaminants",
    },
    {
        "problem_context": "make a lightweight but strong structural material for vehicles or aircraft",
        "solution_direction": "Use composite materials (carbon fiber, fiberglass) or aluminum alloys with high strength-to-weight ratio",
    },
    {
        "problem_context": "make a building material that cleans the air or reduces pollution from its surface",
        "solution_direction": "Use photocatalytic cement (titanium dioxide) that breaks down air pollutants when exposed to sunlight",
    },
    {
        "problem_context": "make a material that absorbs impact or shock without breaking",
        "solution_direction": "Use a viscoelastic or cellular material (foam, honeycomb) that dissipates energy through deformation",
    },
    # --- Energy / power ---
    {
        "problem_context": "transport a large volume of gas efficiently in a small container",
        "solution_direction": "Liquefy the gas under pressure and cold to reduce its volume dramatically for transport",
    },
    {
        "problem_context": "store energy for long duration or high density in a battery",
        "solution_direction": "Use a solid-state electrolyte battery with high energy density and no liquid flammability",
    },
    {
        "problem_context": "convert sunlight to electricity more efficiently or cheaply",
        "solution_direction": "Use perovskite solar cells with a tunable bandgap and high absorption coefficient",
    },
    {
        "problem_context": "generate electricity from a renewable source at scale",
        "solution_direction": "Use wind turbines, solar photovoltaics, or hydroelectric generation depending on the resource",
    },
    {
        "problem_context": "heat food quickly and evenly without an external flame or element",
        "solution_direction": "Use a magnetron to generate microwaves that excite water molecules inside the food",
    },
    {
        "problem_context": "achieve net energy gain from nuclear fusion",
        "solution_direction": "Use inertial confinement or magnetic confinement to sustain a plasma that produces more energy than it consumes",
    },
    # --- Biology / chemistry ---
    {
        "problem_context": "treat a bacterial infection that was previously fatal",
        "solution_direction": "Use a mold-derived antibiotic (penicillin) that disrupts bacterial cell wall synthesis",
    },
    {
        "problem_context": "edit or modify the DNA of living organisms precisely",
        "solution_direction": "Use CRISPR-Cas9 with a guide RNA to make targeted cuts in the genome",
    },
    {
        "problem_context": "grow meat without raising and slaughtering animals",
        "solution_direction": "Culture animal cells in a bioreactor with a scaffold to produce lab-grown meat",
    },
    {
        "problem_context": "grow crops in a controlled environment without soil or large land area",
        "solution_direction": "Use vertical farming with hydroponics and LED lighting in stacked layers",
    },
    {
        "problem_context": "remove plastic waste from the ocean or clean up marine pollution",
        "solution_direction": "Use floating booms and collection systems that passively concentrate and harvest plastic debris",
    },
    {
        "problem_context": "capture carbon dioxide directly from the atmosphere to reduce greenhouse gases",
        "solution_direction": "Use direct air capture with chemical sorbents that bind CO2 and release it for storage",
    },
    # --- Mechanisms / design ---
    {
        "problem_context": "make a fastener that can be opened and closed repeatedly without wearing out",
        "solution_direction": "Mimic the hook-and-loop structure of burrs that stick to fabric (Velcro)",
    },
    {
        "problem_context": "make a paper note that sticks temporarily and can be removed without residue",
        "solution_direction": "Use a low-tack adhesive that was originally a failed super-strong glue (Post-it)",
    },
    {
        "problem_context": "fasten clothing quickly and securely without buttons or laces",
        "solution_direction": "Interlock a series of teeth with a sliding mechanism that meshes them together (zipper)",
    },
    {
        "problem_context": "correct both near and far vision in a single pair of glasses",
        "solution_direction": "Combine two lens powers in one lens with the near-vision segment at the bottom (bifocal)",
    },
    {
        "problem_context": "span a wide river without building many support pillars in the water",
        "solution_direction": "Hang the deck from cables suspended between tall towers, transferring load to the towers (suspension bridge)",
    },
    {
        "problem_context": "see inside the human body without cutting it open",
        "solution_direction": "Use high-energy electromagnetic radiation (X-rays) that passes through soft tissue but is absorbed by bone",
    },
    {
        "problem_context": "regulate an irregular heartbeat without open surgery",
        "solution_direction": "Implant a small electronic device (pacemaker) that delivers electrical pulses to the heart",
    },
    {
        "problem_context": "shave without cutting the skin",
        "solution_direction": "Guard the blade with a protective comb so only a thin edge contacts the skin (safety razor)",
    },
    {
        "problem_context": "make furniture that assembles quickly without special tools",
        "solution_direction": "Pre-drill holes in flat-pack furniture so assembly is immediate with simple fasteners",
    },
    {
        "problem_context": "make a vacuum cleaner that works both as a full unit and a portable handheld",
        "solution_direction": "Divide the vacuum into independent parts so the handheld unit detaches (stick vacuum)",
    },
    {
        "problem_context": "hide a large structure from aerial reconnaissance or observation",
        "solution_direction": "Camouflage the structure to resemble its surroundings, e.g. a fake neighborhood on a factory roof",
    },
    {
        "problem_context": "reduce latency in a distributed system while keeping strong consistency",
        "solution_direction": "Use a hybrid logical clock with quorum reads to bound staleness while keeping ordering",
    },
    {
        "problem_context": "prevent catastrophic failure of large liquid storage tanks",
        "solution_direction": "Enforce stricter inspection standards and expert safety analysis of tank integrity",
    },
    {
        "problem_context": "correct errors in quantum computation caused by decoherence",
        "solution_direction": "Use quantum error correction with redundant physical qubits encoding logical qubits",
    },
    {
        "problem_context": "make a clock or time source that stays accurate across distributed nodes",
        "solution_direction": "Use a hybrid logical clock combining physical time with logical counters",
    },
    # --- Mathematics ---
    {
        "problem_context": "sum a long arithmetic series of consecutive numbers quickly without adding each term",
        "solution_direction": "Pair the first and last terms and sum them, then multiply by the count divided by two (Gauss's formula n(n+1)/2)",
    },
    {
        "problem_context": "find the greatest common divisor of two large numbers without factoring them",
        "solution_direction": "Repeatedly replace the larger number by the remainder of dividing it by the smaller to find the greatest common divisor (Euclid's algorithm)",
    },
    {
        "problem_context": "count the prime numbers up to a large limit efficiently",
        "solution_direction": "Iteratively mark multiples of each prime as composite, skipping already-marked numbers (Sieve of Eratosthenes)",
    },
    {
        "problem_context": "compute the area under a curve that has no simple antiderivative",
        "solution_direction": "Approximate the area by summing thin rectangles or trapezoids under the curve (Riemann sum / numerical integration)",
    },
    {
        "problem_context": "solve a system of linear equations with many variables",
        "solution_direction": "Eliminate variables one at a time by row operations to reach triangular form (Gaussian elimination)",
    },
    {
        "problem_context": "find the shortest path between two points in a weighted graph",
        "solution_direction": "Repeatedly relax edges from the nearest unvisited node, updating tentative distances (Dijkstra's algorithm)",
    },
    {
        "problem_context": "prove a statement holds for all natural numbers without checking each one",
        "solution_direction": "Prove the base case, then show that if it holds for n it holds for n+1 (mathematical induction)",
    },
    {
        "problem_context": "find the maximum value of a function on a closed interval",
        "solution_direction": "Check critical points where the derivative is zero plus the interval endpoints (extreme value theorem)",
    },
    {
        "problem_context": "determine whether a large number is prime without trial division by every smaller number",
        "solution_direction": "Use a probabilistic test that checks modular exponentiation witnesses (Miller-Rabin primality test)",
    },
    {
        "problem_context": "compute the nth Fibonacci number without exponential recursion",
        "solution_direction": "Build the Fibonacci sequence iteratively from the bottom up, storing each result (dynamic programming)",
    },
    # --- Coding / algorithms ---
    {
        "problem_context": "search for a value in a large sorted list faster than checking every element",
        "solution_direction": "Repeatedly halve the search range by comparing against the middle element (binary search)",
    },
    {
        "problem_context": "sort a large list of items efficiently without nested comparison of every pair",
        "solution_direction": "Recursively split the list in half, sort each half, and merge the sorted halves (merge sort)",
    },
    {
        "problem_context": "look up a value by key in constant time without scanning the whole collection",
        "solution_direction": "Map each key to a bucket index via a hash function and store the value there (hash table)",
    },
    {
        "problem_context": "find the longest common subsequence of two strings without trying all subsequences",
        "solution_direction": "Fill a table of prefix lengths, building the answer from previously computed cells (dynamic programming)",
    },
    {
        "problem_context": "traverse or search a tree or graph visiting each node exactly once",
        "solution_direction": "Use a stack to explore depth-first or a queue to explore breadth-first, marking visited nodes",
    },
    {
        "problem_context": "find the minimum spanning tree connecting all nodes of a graph with least total weight",
        "solution_direction": "Repeatedly add the cheapest edge that connects a new node without forming a cycle (Kruskal's algorithm)",
    },
    {
        "problem_context": "detect whether a linked list contains a cycle without extra memory",
        "solution_direction": "Advance two pointers at different speeds; if they meet, a cycle exists (Floyd's tortoise and hare)",
    },
    {
        "problem_context": "compress a text file by replacing frequent patterns with shorter codes",
        "solution_direction": "Build a binary tree of character frequencies and assign shorter codes to more frequent characters (Huffman coding)",
    },
    {
        "problem_context": "find the maximum subarray sum in a sequence without checking every subarray",
        "solution_direction": "Scan once, keeping the best sum ending at each position and the global best (Kadane's algorithm)",
    },
    {
        "problem_context": "evaluate a mathematical expression with operator precedence without ambiguity",
        "solution_direction": "Convert the expression to postfix notation and evaluate with a stack (shunting-yard algorithm)",
    },
    # --- Advanced math / number theory ---
    {
        "problem_context": "find the modular inverse of a number under a prime modulus for cryptography",
        "solution_direction": "Use the extended Euclidean algorithm to find the coefficient that makes the product congruent to 1",
    },
    {
        "problem_context": "compute large powers modulo a number efficiently for encryption",
        "solution_direction": "Repeatedly square the base and multiply only when the exponent bit is set (modular exponentiation)",
    },
    {
        "problem_context": "find all prime factors of a very large number faster than trial division",
        "solution_direction": "Use Pollard's rho algorithm with a pseudo-random sequence to find a nontrivial factor",
    },
    {
        "problem_context": "count the number of ways to choose k items from n without enumerating all combinations",
        "solution_direction": "Use the binomial coefficient formula n!/(k!(n-k)!) or Pascal's triangle recurrence",
    },
    {
        "problem_context": "find the nth term of a linear recurrence without iterating all previous terms",
        "solution_direction": "Exponentiate the companion matrix or use the characteristic polynomial (matrix exponentiation)",
    },
    {
        "problem_context": "determine whether two large numbers are coprime without factoring them",
        "solution_direction": "Compute their greatest common divisor with Euclid's algorithm; coprime if the GCD is 1",
    },
    # --- Advanced algorithms / competitive programming ---
    {
        "problem_context": "find the maximum flow from a source to a sink in a network with capacity limits",
        "solution_direction": "Repeatedly find augmenting paths and push flow along them until no more exist (Ford-Fulkerson)",
    },
    {
        "problem_context": "find the minimum cost to connect all nodes in a graph with weighted edges",
        "solution_direction": "Repeatedly add the cheapest edge that connects a new node without forming a cycle (Kruskal's algorithm)",
    },
    {
        "problem_context": "find the longest increasing subsequence of an array without trying all subsequences",
        "solution_direction": "Maintain a sorted list of smallest possible tails and binary-search the position of each element",
    },
    {
        "problem_context": "find the kth smallest element in an unsorted array without fully sorting it",
        "solution_direction": "Partition around a pivot and recurse only into the side containing the kth element (quickselect)",
    },
    {
        "problem_context": "find all pairs of numbers in an array that sum to a target value",
        "solution_direction": "Use a hash set to check for the complement of each element in one pass (two-sum)",
    },
    {
        "problem_context": "merge two sorted arrays into one sorted array efficiently",
        "solution_direction": "Advance two pointers from the start of each array, always taking the smaller element (merge)",
    },
    {
        "problem_context": "find the median of two sorted arrays in logarithmic time",
        "solution_direction": "Binary-search the partition point in the smaller array so the left halves balance (median of two sorted arrays)",
    },
    {
        "problem_context": "detect if a string is a palindrome ignoring spaces and case",
        "solution_direction": "Compare characters from both ends moving inward, skipping non-alphanumeric characters",
    },
    {
        "problem_context": "find the number of islands in a grid of land and water cells",
        "solution_direction": "Flood-fill each unvisited land cell with depth-first search, counting each fill as one island",
    },
    # --- Networking / distributed systems ---
    {
        "problem_context": "keep a distributed system available when one server or node fails",
        "solution_direction": "Replicate state across multiple nodes and fail over to a healthy replica when the leader dies",
    },
    {
        "problem_context": "make all nodes in a distributed system agree on the same value or order",
        "solution_direction": "Use a consensus protocol (Raft or Paxos) where a majority of nodes must agree before committing",
    },
    {
        "problem_context": "deliver a message exactly once across an unreliable network that may drop or duplicate",
        "solution_direction": "Assign each message a unique id and deduplicate on the receiver while retrying on timeout",
    },
    {
        "problem_context": "balance incoming traffic across many servers so no single one is overloaded",
        "solution_direction": "Put a load balancer in front that distributes requests round-robin or by least connections",
    },
    {
        "problem_context": "make a service continue working when the database or backend is slow or down",
        "solution_direction": "Add a circuit breaker that fails fast and serves cached or degraded responses during outages",
    },
    {
        "problem_context": "retry a network call that intermittently fails without hammering the server",
        "solution_direction": "Retry with exponential backoff and jitter, capping the number of attempts",
    },
    {
        "problem_context": "scale a service to handle a sudden spike in traffic without downtime",
        "solution_direction": "Auto-scale horizontally by adding more instances behind a load balancer based on load metrics",
    },
    {
        "problem_context": "keep a queue of work that survives a worker crash and is not lost",
        "solution_direction": "Persist the queue to durable storage and use leases so a crashed worker's work is retried",
    },
    # --- Security / reliability ---
    {
        "problem_context": "prevent unauthorized users from accessing a system or resource",
        "solution_direction": "Require authentication (verify identity) and authorization (check permissions) on every request",
    },
    {
        "problem_context": "protect data stored or transmitted so an eavesdropper cannot read it",
        "solution_direction": "Encrypt the data with a strong cipher and manage keys separately from the data",
    },
    {
        "problem_context": "detect that a system has been compromised or is behaving abnormally",
        "solution_direction": "Monitor for anomalies and known-bad signatures, alerting on deviations from baseline behavior",
    },
    {
        "problem_context": "recover a system quickly after a crash or disaster without losing data",
        "solution_direction": "Take regular backups and define a tested restore procedure with a recovery time objective",
    },
    {
        "problem_context": "make a system fail gracefully instead of crashing completely when a component errors",
        "solution_direction": "Isolate failures with timeouts, fallbacks, and graceful degradation so partial failure is contained",
    },
    {
        "problem_context": "prevent a malicious input from crashing or exploiting a program",
        "solution_direction": "Validate and sanitize all input, and run untrusted code in a sandbox with limited privileges",
    },
    # --- Data engineering / ops ---
    {
        "problem_context": "process a very large dataset that does not fit in memory on one machine",
        "solution_direction": "Partition the data and process it in parallel across a cluster, then aggregate the results",
    },
    {
        "problem_context": "keep a data pipeline from failing silently when a source schema changes",
        "solution_direction": "Add schema validation and contract tests that fail loudly on unexpected data shapes",
    },
    {
        "problem_context": "make a slow database query fast without changing the application code",
        "solution_direction": "Add an index on the columns used in the WHERE and JOIN clauses to avoid full scans",
    },
    {
        "problem_context": "keep a long-running job from losing progress if it is interrupted",
        "solution_direction": "Checkpoint progress periodically so the job can resume from the last checkpoint",
    },
    {
        "problem_context": "detect when a metric or system behavior drifts outside its normal range",
        "solution_direction": "Track a rolling baseline and alert when the current value deviates by more than a threshold",
    },
    {
        "problem_context": "deploy new code without downtime or risk to the running system",
        "solution_direction": "Use blue-green or canary deployment, routing a small fraction of traffic to the new version first",
    },
    {
        "problem_context": "find the root cause of an intermittent failure that is hard to reproduce",
        "solution_direction": "Add structured logging and tracing so the failure path can be reconstructed from evidence",
    },
]


def build_domain_knowledge_index(service, backend=None) -> None:
    """Seed the vector index with the domain-knowledge fact base.

    Each entry is stored as a PROBLEM (the problem_context, which matches a new
    problem's phrasing) LINKED to a SOLUTION (the concrete solution_direction).
    This lets hybrid_analogize match a new problem to the right concrete
    solution via both problem-vector and solution-vector scoring.
    """
    for entry in DOMAIN_KNOWLEDGE:
        pid = service.store_problem(
            specification=entry["problem_context"],
            title=entry["problem_context"][:60],
            metadata={"kind": "domain_knowledge"},
        )
        sid = service.store_solution(
            solution_text=entry["solution_direction"],
            method="domain_knowledge",
            domain="engineering",
            title=entry["problem_context"][:60],
            metadata={"problem_context": entry["problem_context"]},
        )
        # Link the problem to its solution so analogize/hybrid_analogize resolve it.
        service.index._conn.execute(
            "UPDATE vectors SET metadata_json=? WHERE id=?",
            (json.dumps({"solution_id": sid, "kind": "domain_knowledge"}), pid),
        )
    service.index._conn.commit()


def retrieve_concrete_solutions(service, problem_text: str, top_k: int = 3) -> List[Dict]:
    """Retrieve the most relevant concrete solution directions for a problem.

    Uses hybrid_analogize (problem + solution vector scoring) over the
    domain-knowledge fact base, so a problem whose *solution* is semantically
    close ranks higher even if its problem wording differs.
    """
    qv = service.encode_problem(problem_text, title="query")
    results = service.hybrid_analogize(qv, top_k=top_k, min_similarity=0.0)
    out = []
    for r in results:
        sol = r.get("solution")
        if sol and sol.get("content"):
            out.append(
                {
                    "content": sol["content"],
                    "similarity": r["problem"].get("similarity", 0.0),
                    "problem_context": r["problem"].get("metadata", {}).get("problem_context", ""),
                }
            )
    return out
