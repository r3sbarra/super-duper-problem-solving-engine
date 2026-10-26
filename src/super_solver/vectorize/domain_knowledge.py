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
    # --- Systems: mechanisms for measured 0-overlap solve-loop gaps ---
    {
        "problem_context": "reduce cache invalidation spikes in p99 latency when many keys expire at once",
        "solution_direction": "Coordinate the misses with single-flight so only one worker recomputes each expired entry and the rest wait for its result",
    },
    {
        "problem_context": "detect duplicate invoices or payments across two billing systems",
        "solution_direction": "Hash the normalised invoice fields and join on the hash (or a unique idempotency key) to find collisions across systems",
    },
    {
        "problem_context": "estimate how long a queue takes to drain after a traffic spike",
        "solution_direction": "Model arrival and service rates, compute the backlog, and integrate service capacity over time to drain it",
    },
    {
        "problem_context": "explain why a recommendation engine becomes less diverse over time",
        "solution_direction": "Feedback loops amplify popular items and starve the long tail of exposure; add exploration or diversity re-ranking",
    },
    {
        "problem_context": "detect a data race that only fails sometimes under concurrency",
        "solution_direction": "Instrument shared-memory accesses and look for unsynchronised concurrent writes, or use a happens-before detector",
    },
    {
        "problem_context": "make short-route delivery routing respect time windows",
        "solution_direction": "Model the road network as a weighted graph and relax edges with time-window feasibility, using label-setting with dominance",
    },
    {
        "problem_context": "make a battery charge quickly without overheating",
        "solution_direction": "Increase electrode surface area and manage thermal dissipation during charging, tapering current as temperature rises",
    },
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
    # --- Mathematics: number theory ---
    {
        "problem_context": "compute the greatest common divisor of two large integers efficiently",
        "solution_direction": "Use the Euclidean algorithm for the greatest common divisor: repeatedly replace the larger number by the remainder of the division until it reaches zero; the last nonzero remainder is the gcd",
    },
    {
        "problem_context": "find a common measure shared by two big whole numbers",
        "solution_direction": "Compute the greatest common divisor with the Euclidean algorithm by taking the remainder of the division; the last nonzero remainder is the gcd common divisor",
    },
    {
        "problem_context": "decide whether a very large number is prime or composite",
        "solution_direction": "Run a primality test such as Miller-Rabin probabilistic witness testing; a composite has a witness that fails Fermat's condition and proves it composite",
    },
    {
        "problem_context": "count how many prime numbers are up to a large limit",
        "solution_direction": "Use a prime sieve such as the Sieve of Eratosthenes, marking composites from each prime's square upward",
    },
    {
        "problem_context": "find a nontrivial factor of a huge composite number",
        "solution_direction": "Use Pollard's rho method with the birthday paradox to find a nontrivial factor without trial division",
    },
    {
        "problem_context": "compute a large power modulo an integer for cryptography",
        "solution_direction": "Use modular exponentiation by squaring the base and multiplying, reducing modulo at each step",
    },
    {
        "problem_context": "solve for a number whose product with another is congruent to one modulo n",
        "solution_direction": "Compute the modular inverse with the extended Euclidean algorithm so the product is congruent to one",
    },
    {
        "problem_context": "count the number of ways to choose k items from n",
        "solution_direction": "Use the binomial coefficient n choose k, built from Pascal's triangle or factorials",
    },
    # --- Mathematics: algebra, calculus ---
    {
        "problem_context": "add up all the integers from 1 to a large number in your head",
        "solution_direction": "Pair the terms to form an arithmetic series and use Gauss's formula sum = n*(n+1)/2",
    },
    {
        "problem_context": "find the exact area under a curve or the antiderivative of a function",
        "solution_direction": "Compute the integral symbolically: use exact rational integration or a substitution, or numerically integrate if no closed form exists",
    },
    {
        "problem_context": "find the maximum or minimum of a function of several variables",
        "solution_direction": "Compute the gradient, set the derivative to zero at critical points, and check the second derivative (Hessian) to classify the extremum",
    },
    {
        "problem_context": "solve a system of linear equations exactly",
        "solution_direction": "Use Gaussian elimination with row operations to reduce the system to triangular form and back-substitute the solution",
    },
    {
        "problem_context": "prove a statement holds for all natural numbers",
        "solution_direction": "Use mathematical induction: establish the base case and the inductive step that holds for n implies n+1",
    },
    {
        "problem_context": "derive the exact recurrence or closed form of a numerical sequence",
        "solution_direction": "Use a dynamic programming / recurrence relation such as the Fibonacci recurrence, or fit the Lagrange polynomial closed form",
    },
    {
        "problem_context": "find all real roots of a polynomial on an interval",
        "solution_direction": "Isolate the real roots exactly using Sturm's theorem, counting sign changes along the Sturm chain to find where the polynomial crosses zero",
    },
    {
        "problem_context": "find all the places a polynomial crosses zero on an interval",
        "solution_direction": "Isolate the real roots with Sturm's theorem, counting sign changes along the Sturm chain to locate where the polynomial crosses zero",
    },
    {
        "problem_context": "determine whether a given constant is algebraic and find its minimal polynomial",
        "solution_direction": "Use an integer relation algorithm (PSLQ) to find the minimal polynomial with integer coefficients satisfied by the constant",
    },
    {
        "problem_context": "test whether a mathematical identity is actually true",
        "solution_direction": "Stress-test the identity against adversarial counterexample boundary points (Lakatos falsification); if no counterexample exists, prove it algebraically",
    },
    # --- Mathematics: probability & statistics ---
    {
        "problem_context": "summarise a set of measurements by their spread and central tendency",
        "solution_direction": "Compute descriptive statistics: the mean and the variance and standard deviation, with the median and quartiles for robustness",
    },
    {
        "problem_context": "decide whether an observed difference between two groups is statistically significant",
        "solution_direction": "Run a hypothesis test (t-test or chi-square) on the two groups and compute the p-value against the null hypothesis to see if they really differ rather than by chance",
    },
    {
        "problem_context": "tell me if two groups really differ rather than by chance",
        "solution_direction": "Run a hypothesis test such as a t-test and compute the p-value against the null hypothesis of no difference between the two groups",
    },
    {
        "problem_context": "summarise a set of noisy measurements by their spread",
        "solution_direction": "Compute descriptive statistics: the mean and the variance and standard deviation to summarise the spread of the measurements",
    },
    {
        "problem_context": "model the number of successes in a fixed number of independent trials",
        "solution_direction": "Use the binomial distribution pmf with n trials and success probability p",
    },
    {
        "problem_context": "fit a straight line or curve through noisy data points",
        "solution_direction": "Use least-squares regression to solve the normal equations for the slope and intercept (or higher-degree coefficients) so the best-fit line minimises squared error",
    },
    {
        "problem_context": "I have noisy measurements and want a line that fits them best",
        "solution_direction": "Use least-squares regression: solve the normal equations for the slope and intercept of the best-fit line through the noisy measurements",
    },
    # --- Mathematics: geometry & topology ---
    {
        "problem_context": "prove a Euclidean geometry theorem mechanically from its hypotheses",
        "solution_direction": "Use Wu's characteristic method for a geometry theorem: translate the hypotheses into polynomials and pseudodivide the conclusion, checking the remainder is zero",
    },
    {
        "problem_context": "show a fact about triangles using algebra",
        "solution_direction": "Use Wu's method to prove the geometry theorem about triangles: turn the hypotheses into polynomials and pseudodivide the conclusion, checking the remainder is zero",
    },
    {
        "problem_context": "decide whether two knots are topologically different (chiral)",
        "solution_direction": "Compute a knot invariant such as the Jones polynomial; a difference from its mirror proves chirality",
    },
    {
        "problem_context": "count points on an elliptic curve over a finite field",
        "solution_direction": "Use the Hasse bound and point counting to bound the number of points on the Weierstrass elliptic curve over the finite field",
    },
    {
        "problem_context": "how many points lie on a cubic curve over a finite field",
        "solution_direction": "Use point counting on the elliptic curve: count the points on the Weierstrass curve over the finite field and bound the total by the Hasse bound",
    },
    # --- Natural science: physics, chemistry, biology ---
    {
        "problem_context": "predict how a chemical reaction rate changes with temperature",
        "solution_direction": "Use the Arrhenius equation: the reaction rate constant depends exponentially on inverse temperature through the activation energy, so the reaction speeds up as it gets hotter",
    },
    {
        "problem_context": "how does a chemical reaction speed up when things get hotter",
        "solution_direction": "Use the Arrhenius equation: the reaction rate constant rises exponentially with temperature through the activation energy, so the chemical reaction speeds up when it gets hotter",
    },
    {
        "problem_context": "explain how energy is conserved as a pendulum or wave oscillates",
        "solution_direction": "Write the total energy as the sum of kinetic and potential terms and show it is conserved in the absence of damping",
    },
    {
        "problem_context": "estimate how a population of organisms grows over time",
        "solution_direction": "Use the logistic growth model: exponential growth limited by a carrying capacity that saturates the population",
    },
    {
        "problem_context": "model how a disease spreads through a population",
        "solution_direction": "Use a compartmental SIR model tracking susceptible, infected and recovered counts for the infection spreading through the population, and compute the basic reproduction number R0",
    },
    {
        "problem_context": "model how an infection moves through a population",
        "solution_direction": "Use a compartmental SIR model of how the infection spreads through the population across susceptible, infected and recovered compartments, and compute the reproduction number R0",
    },
    {
        "problem_context": "determine the shape of a molecule or the orbitals of an atom",
        "solution_direction": "Solve the Schrodinger equation for the system's quantum states and use symmetry / group representation to classify the orbitals",
    },
    {
        "problem_context": "find the conserved quantities and symmetries of a physical system",
        "solution_direction": "Apply Noether's theorem: identify the continuous symmetry of the Lagrangian and derive its conserved current",
    },
    # --- Applied mathematics: linear algebra algorithms ---
    {
        "problem_context": "compute the determinant of a large matrix exactly without floating-point error",
        "solution_direction": "Use the Bareiss fraction-free elimination algorithm: a determinant computed in O(n^3) with exact integer arithmetic and no division round-off",
    },
    {
        "problem_context": "find the eigenvalues and characteristic polynomial of a matrix",
        "solution_direction": "Compute the characteristic polynomial det(lambda*I - A) and find its roots; the eigenvalues are the roots of the characteristic polynomial, obtained via the Faddeev-LeVerrier algorithm",
    },
    {
        "problem_context": "solve a system of linear equations exactly",
        "solution_direction": "Use Gaussian elimination with row operations to reduce the augmented matrix to triangular form, then back-substitute the exact solution",
    },
    {
        "problem_context": "find the null space or rank of a matrix",
        "solution_direction": "Row-reduce the matrix to reduced row echelon form; the rank is the number of pivots and the null space basis is read off the free columns",
    },
    # --- Applied mathematics: optimization ---
    {
        "problem_context": "solve a linear program with a linear objective and linear constraints",
        "solution_direction": "Use the simplex method: move along the vertices of the feasible polytope, pivoting to improve the objective until the optimum is reached",
    },
    {
        "problem_context": "find the best trade-off among several competing objectives",
        "solution_direction": "Compute the Pareto frontier of non-dominated solutions and choose the knee point that balances the objectives",
    },
    # --- Discrete mathematics / graph theory ---
    {
        "problem_context": "decide whether a graph can be split into two groups with no edge inside a group",
        "solution_direction": "Test bipartiteness by a two-coloring breadth-first traversal: color each node the opposite of its neighbour; a conflict edge proves the graph is not bipartite",
    },
    {
        "problem_context": "find the shortest route that visits every city exactly once",
        "solution_direction": "Use dynamic programming over subsets (Held-Karp) to find the optimal Hamiltonian path through the travelling salesman problem",
    },
    {
        "problem_context": "find the maximum flow through a network with capacities",
        "solution_direction": "Use the Ford-Fulkerson method: find augmenting paths from source to sink and increase flow until no augmenting path remains",
    },
    # --- Information theory / signal processing ---
    {
        "problem_context": "measure the uncertainty or information content of a probability distribution",
        "solution_direction": "Compute the Shannon entropy H = -sum p_i log p_i in bits; it is maximized by the uniform distribution",
    },
    {
        "problem_context": "decompose a signal into its frequency components",
        "solution_direction": "Compute the Fourier transform, decomposing the signal into sine and cosine frequency components, via the FFT algorithm",
    },
    # --- Statistics / estimation ---
    {
        "problem_context": "estimate an unknown parameter from noisy observations",
        "solution_direction": "Use maximum likelihood estimation: write the likelihood of the observations as a function of the parameter and maximize it (or use Bayes' rule for a posterior)",
    },
    {
        "problem_context": "fit a model when the data are noisy and the model is nonlinear",
        "solution_direction": "Use nonlinear least squares: minimize the sum of squared residuals by iterating Gauss-Newton or Levenberg-Marquardt updates",
    },
    # --- Logic / computability ---
    {
        "problem_context": "prove that a program or loop always terminates",
        "solution_direction": "Exhibit a well-founded ranking measure (variant) that strictly decreases on every iteration and is bounded below, proving termination",
    },
    {
        "problem_context": "decide whether a Boolean formula is satisfiable",
        "solution_direction": "Use a SAT solver / DPLL search with unit propagation and conflict-driven clause learning to find a satisfying assignment or prove UNSAT",
    },
    # --- Symbolic / algebraic computation ---
    {
        "problem_context": "find a Groebner basis for a system of polynomial equations",
        "solution_direction": "Run Buchberger's algorithm: repeatedly form S-polynomials of pairs and reduce them until every S-polynomial reduces to zero, giving the Groebner basis",
    },
    {
        "problem_context": "decide whether a polynomial is irreducible over the rationals",
        "solution_direction": "Certify irreducibility by checking the polynomial is square-free and has no rational root, then confirming it is irreducible modulo a prime",
    },
    {
        "problem_context": "prove a polynomial identity symbolically",
        "solution_direction": "Simplify both sides with an E-graph equality-saturation rewrite system until they reach a canonical form, certifying the identity",
    },
    # --- Science / engineering mathematics ---
    {
        "problem_context": "predict how a quantity decays over time",
        "solution_direction": "Model the decay with an exponential law and fit its rate constant from the data by least squares on the log transform",
    },
    {
        "problem_context": "balance a chemical reaction so atoms are conserved",
        "solution_direction": "Set up conservation equations for each element and solve the linear system for the stoichiometric coefficients",
    },
    # --- Bridged from suuri-chan's verified lemma vault (generic law bridge). ---
    # Each solution text is machine-verified to strip to its target primitive.
    {
        "problem_context": "prove a complex-analysis identity symbolically",
        "solution_direction": "Simplify the identity to canonical form and prove it symbolically: e^(i*pi) + 1 = 0 (Euler identity)",
    },
    {
        "problem_context": "measure how many bits of information are needed to store or recall items",
        "solution_direction": "Compute the Shannon entropy in bits of the probability distribution: exact recall of N facts requires at least log2(N) bits",
    },
    {
        "problem_context": "find the nth term of a sequence defined by a recurrence from previous terms",
        "solution_direction": "Use a dynamic programming recurrence where each term follows previous terms: a[n] = c1*a[n-1] + c2*a[n-2]",
    },
    {
        "problem_context": "evaluate a continued fraction expansion of a constant",
        "solution_direction": "Evaluate the expression by iterating the continued fraction to convergence: sqrt2 = GCF(a_n=2, b_n=1)",
    },
    {
        "problem_context": "find the holonomic generating function closed form of a combinatorial sequence",
        "solution_direction": "Simplify the identity to canonical form and prove it symbolically for the holonomic generating function: (a^2 - b^2) = (a-b)(a+b)",
    },
    {
        "problem_context": "add up all the numbers in a series quickly using a closed-form sum",
        "solution_direction": "Use the arithmetic series sum formula to add up the numbers in the series: sum of first n integers = n(n+1)/2",
    },
    {
        "problem_context": "decide primality or count primes",
        "solution_direction": "Run a primality test and prime sieve to count primes: pi(n) = number of primes <= n",
    },
    {
        "problem_context": "compute the integral antiderivative of a function exactly",
        "solution_direction": "Compute the integral antiderivative of the function exactly by Ostrogradsky-Hermite reduction or Risch integration",
    },
    {
        "problem_context": "fit a line or curve through noisy data points",
        "solution_direction": "Fit the best-fit line by least squares regression solving the normal equations for the slope and intercept",
    },
    {
        "problem_context": "classify a knot or braid by computing a topological invariant",
        "solution_direction": "Compute the Jones polynomial knot invariant to test chirality of the braid closure",
    },
    {
        "problem_context": "find and isolate all real roots of a polynomial",
        "solution_direction": "Isolate the real roots with Sturm theorem, counting sign changes along the Sturm chain to bracket every real root",
    },
    {
        "problem_context": "find the eigenvalues of a matrix from its characteristic polynomial",
        "solution_direction": "Find the eigenvalues from the characteristic polynomial det(lambda*I - A) via the Faddeev-LeVerrier algorithm",
    },
    # --- Coverage-gap fill (measured 2026-10-19). Each entry below corresponds
    #     to a probe query the primitive matcher MISSED because the concept was
    #     absent from the fact base entirely (not a ranking failure): a grep for
    #     "community", "scheduling", "spanning", "bloom" and "suffix" over all
    #     174 stored solutions returned zero rows. ---
    {
        "problem_context": "partition a network into communities or clusters so that internal links are dense and links between clusters are sparse",
        "solution_direction": "Detect communities by maximizing modularity: Louvain greedily merges nodes into communities while modularity increases, then aggregates the graph and repeats (hierarchical clustering)",
    },
    {
        "problem_context": "find groups of similar nodes in a graph without being told how many groups there are",
        "solution_direction": "Cluster the graph by modularity optimization or by the leading eigenvectors of the graph Laplacian (spectral clustering), letting the number of communities emerge from the structure",
    },
    {
        "problem_context": "schedule jobs on a single machine to minimise the total weighted completion time",
        "solution_direction": "Order the jobs by Smith's rule: sequence them in decreasing ratio of weight to processing time (weighted shortest processing time), which minimises total weighted completion time",
    },
    {
        "problem_context": "sequence tasks on one machine to minimise the average time a job waits",
        "solution_direction": "Use shortest processing time first: run the jobs in increasing processing time to minimise average waiting time and mean flow time (SPT scheduling rule)",
    },
    {
        "problem_context": "connect every node of a network with the cheapest possible set of new links without creating a cycle",
        "solution_direction": "Build a minimum spanning tree: Kruskal's algorithm repeatedly adds the cheapest edge that does not form a cycle (union-find), or Prim's algorithm grows the tree from one node",
    },
    {
        "problem_context": "test cheaply whether an item is definitely absent from a huge set without storing every element",
        "solution_direction": "Use a Bloom filter: hash each item to several bit positions in a fixed-size bitmap; a zero bit proves absence while all-ones gives a bounded false-positive probability",
    },
    {
        "problem_context": "search a huge static text or genome for patterns without decompressing or loading it all",
        "solution_direction": "Index the text with a compressed full-text index such as the suffix array, FM-index or Burrows-Wheeler transform so patterns are found by backward search over compressed data",
    },
    {
        "problem_context": "find all occurrences of many different patterns in one long document quickly",
        "solution_direction": "Build an Aho-Corasick automaton over the pattern set and scan the document once, reporting every match from a single pass",
    },
    {
        "problem_context": "estimate the probability of a rare event with as few samples as possible",
        "solution_direction": "Use importance sampling: draw samples from a tilted proposal distribution that makes the rare event common, then reweight each sample by the likelihood ratio to stay unbiased",
    },
    {
        "problem_context": "estimate an integral or expectation in high dimension where grid sampling fails",
        "solution_direction": "Use Monte Carlo integration: average the integrand over random samples, with variance reduced by stratification, control variates or antithetic sampling",
    },
    {
        "problem_context": "detect duplicate records across two very large lists without comparing every pair",
        "solution_direction": "Hash a canonical key for each record into a set and join on the hash: build a hash set of one list and probe it with the other in linear time (merge or hash join)",
    },
    {
        "problem_context": "make a model or service answer in under a millisecond",
        "solution_direction": "Reduce per-request work: batch requests into one vectorized operation, cache results for repeated inputs, and precompute a static fused matrix so inference is a single small matrix multiply",
    },
    {
        "problem_context": "prove that every even integer greater than two is the sum of two primes",
        "solution_direction": "Goldbach's conjecture: verify the strong Goldbach decomposition n = p + q over primes for a range and attempt a proof by strong induction or the circle method",
    },
    {
        "problem_context": "count how many numbers up to n share no factor with n",
        "solution_direction": "Compute Euler's totient phi(n) from the prime factorisation as n * product(1 - 1/p) over the distinct prime factors p of n",
    },
    {
        "problem_context": "find the k-th smallest or largest element of an unsorted list without sorting the whole list",
        "solution_direction": "Use quickselect: partition around a pivot and recurse only into the side that contains the k-th element, giving linear expected time",
    },
    {
        "problem_context": "explain why a set of probabilities must sum to one",
        "solution_direction": "Normalise the distribution: divide every weight by the total so the probabilities sum to one, which follows from the axioms of probability and the law of total probability",
    },
    {
        "problem_context": "decode a message that was scrambled by permuting its characters",
        "solution_direction": "Invert the permutation or search the permutation space: use frequency analysis and anagram matching, or recover the cipher permutation from known plaintext",
    },
    {
        "problem_context": "find a short route between two locations on a road network",
        "solution_direction": "Run Dijkstra's algorithm on the weighted graph or A* with an admissible distance heuristic to find the shortest path",
    },
    # --- Video / computational imaging (added wave 3).
    #     Each solution_direction repeats the trigger words of its own primitive
    #     (see SKILL.md step 3): the problem and the solution are phrased in
    #     different vocabularies, so a solution that does not name its own
    #     mechanism strips to empty and can never be retrieved. ---
    {
        "problem_context": "apply the dance moves of a performer in one video clip onto a different person in another video",
        "solution_direction": "Estimate dense optical flow (pixel motion) from source to target and warp the source video pixels along that flow field onto the target's body, instead of regenerating the frames from scratch",
    },
    {
        "problem_context": "reduce the cost of a video-to-video edit that regenerates every frame with a diffusion model",
        "solution_direction": "Render only sparse keyframes and propagate the result forward and backward to neighbouring frames with bidirectional flow warping, regenerating a small fraction of frames",
    },
    {
        "problem_context": "separate a moving person from a static filmed background so only the subject is edited",
        "solution_direction": "Decompose the clip into a low-rank background layer plus a sparse residual foreground layer (RPCA / robust PCA, temporal-median background subtraction) instead of per-pixel thresholding",
    },
    {
        "problem_context": "take the outfit or clothing from one clip and put it on the person in another clip with a different pose and shape",
        "solution_direction": "Canonicalise the body to a UV texture atlas: unwrap both meshes to the same template UV map, retarget the garment texture in the atlas, then re-project to the target pose",
    },
    {
        "problem_context": "warp one image or video frame onto another when the motion is not a single global transform",
        "solution_direction": "Fit a mesh of triangles over the video frame and apply a piecewise-affine warp per triangle (or a thin-plate spline), so locally different motions are handled separately",
    },
    {
        "problem_context": "stop a per-frame neural render from flickering and shimmering between consecutive frames",
        "solution_direction": "Apply motion-compensated temporal filtering to the video: warp the previous output frame into the current frame with flow, then blend it with overlap so texture is stable across frames and the flicker is suppressed",
    },
    {
        "problem_context": "process a long video in a streaming loop without visible seams or joints between batches",
        "solution_direction": "Process the clip in overlapping chunks and blend the overlap-add region with a feathering weight, so the seam between segment boundaries is invisible",
    },
    {
        "problem_context": "make a video pipeline fast enough on weak hardware by spending compute only where it matters",
        "solution_direction": "Allocate compute by ROI and tile: run the expensive network only on the region of interest of the video frame (the face and body tiles) at full resolution and skip static tiles",
    },
    {
        "problem_context": "a per-frame network re-extracts the same identity and appearance features on every frame and is too slow",
        "solution_direction": "Reuse cached features across frames of the video clip: extract the backbone embedding once per keyframe, cache it, and track it to later frames instead of recomputing every frame",
    },
    {
        "problem_context": "process high-resolution video on a machine without enough memory for a full-resolution pipeline",
        "solution_direction": "Use a downsample pyramid: run a separable box filter at low resolution to get a coarse result (coarse-to-fine, multi-scale), then refine only the region of interest at higher resolution",
    },
    {
        "problem_context": "make a 4 hour video cheaper and faster to process on weak hardware with no GPU",
        "solution_direction": "Quantise and distil the model (int8 / fp16 quantisation, pruning) so a small student model runs on CPU, cutting the cost per frame without retraining from scratch",
    },
    {
        "problem_context": "a long video exhausts VRAM or runs out of memory when decoded all at once",
        "solution_direction": "Run a streaming pipeline with a bounded VRAM ring buffer: decode frame by frame through a pipe of raw frames, keep only a small window in memory, and write each processed chunk out before reading the next",
    },
    {
        "problem_context": "upscale or sharpen video footage to a higher resolution without a heavy model per frame",
        "solution_direction": "Do a separable multi-scale pass: run the cheap edge-aware filter on the downsample pyramid coarse level first, then apply the residual only where the difference is large (coarse-to-fine upsampling)",
    },
    {
        "problem_context": "transfer motion from a reference clip to a generated character while keeping its identity and clothing stable",
        "solution_direction": "Two-stage transfer: canonicalise the character to a UV texture atlas so identity and clothing stay fixed, then warp the body piecewise-affine (mesh) by the optical flow sampled from the reference clip's skeleton motion",
    },
    {
        "problem_context": "edit only the actor in a scene and leave the background and other objects untouched",
        "solution_direction": "Decompose the frame into background and foreground layers with low-rank plus sparse residual, build a foreground mask by background subtraction, and composite the edited layer back with feathered mask edges",
    },
    {
        "problem_context": "a video processing job produces a jittery, popping result when run frame by frame with an independent model",
        "solution_direction": "Add temporal consistency: motion-compensated flow warping of the previous output with overlap blending and bidirectional propagation from keyframes, so the model output is temporally filtered rather than independent per frame",
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
