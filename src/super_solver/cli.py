"""Command-line interface for Super-Duper-Problem-Solving-Engine."""

from typing import List, Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from super_solver.engine import SuperDuperProblemSolvingEngine

console = Console()


@click.group()
def main():
    """Super-Duper-Problem-Solving-Engine: Autonomous Neurosymbolic Discovery Engine."""
    pass


@main.command()
@click.argument("problem_title")
@click.option("--spec", required=True, help="Problem specification / anomaly observation")
def solve(problem_title: str, spec: str):
    """Solve or deduce the path for a given problem."""
    console.print(
        Panel(f"[bold cyan]{problem_title}[/bold cyan]\n{spec}", title="Problem Formulation")
    )
    engine = SuperDuperProblemSolvingEngine()
    engine.formulate_problem(title=problem_title, specification=spec)

    triz_matches = engine.triz.suggest_principles(spec, top_k=3)
    t = Table(title="Suggested TRIZ Inventive Operators")
    t.add_column("Principle #", style="cyan")
    t.add_column("Name", style="green")
    t.add_column("Similarity", style="magenta")

    for p in triz_matches:
        t.add_row(str(p["principle_id"]), p["name"], f"{p['similarity_score']:.3f}")
    console.print(t)


@main.command(name="self-improve")
def self_improve_cmd():
    """Runs the engine self-referentially against its own architecture to optimize itself."""
    console.print(
        Panel(
            "[bold yellow]Initiating Autonomous Engine Self-Improvement Cycle[/bold yellow]",
            title="Meta-Optimization",
        )
    )
    engine = SuperDuperProblemSolvingEngine()
    audit = engine.run_self_audit_and_optimization()

    t = Table(title="Self-Audit & Formal Invariant Results")
    t.add_column("Module", style="cyan")
    t.add_column("Verification / Metric", style="green")
    t.add_column("Status / Verdict", style="magenta")

    t.add_row(
        "DPLL Logic Solver",
        "Sound Trajectory Invariant",
        audit["dpll_invariant_verification"]["verdict"],
    )
    t.add_row(
        "Causal VSA (Pearl SCM)",
        f"Repulsion -> Success ACE: +{audit['causal_intervention_audit']['average_causal_effect']:.2f}",
        "EFFECTIVE",
    )
    t.add_row(
        "BOED Designer",
        f"Entropy EIG: {audit['boed_tuning_utility']['expected_information_gain']:.3f}",
        "OPTIMAL_UTILITY",
    )
    console.print(t)

    console.print("[bold green]Executing Self-Optimization Trajectory...[/bold green]")
    discovery = engine.improve_self()
    console.print(f"[bold white]Outcome:[/bold white] {discovery.final_breakthrough}")
    console.print(f"[bold cyan]Tuned Parameters:[/bold cyan] {audit['tuned_parameters']}")


@main.command(name="consolidate-memory")
@click.option(
    "--threshold",
    default=0.75,
    type=float,
    help="Similarity threshold for hazard cluster consolidation",
)
def consolidate_memory_cmd(threshold: float):
    """Prunes and consolidates redundant negative manifold repulsors into compact centroid schemas."""
    engine = SuperDuperProblemSolvingEngine()
    pruned = engine.consolidate_memory(similarity_threshold=threshold)
    console.print(
        f"[bold green]Synaptic Consolidation Complete:[/bold green] Pruned {pruned} redundant vectors."
    )


@main.command(name="tournament")
@click.option(
    "--hyp", "hypotheses", multiple=True, required=True, help="Candidate hypothesis to debate"
)
@click.option("--context", default=None, help="Background context or paper abstract")
def tournament_cmd(hypotheses: List[str], context: Optional[str]):
    """Runs an adversarial Red Team vs Blue Team dialectical tournament across hypotheses."""
    console.print(
        Panel(
            "[bold red]Red Team[/bold red] vs [bold blue]Blue Team[/bold blue] Dialectical Tournament",
            title="Adversarial Cross-Examination",
        )
    )
    engine = SuperDuperProblemSolvingEngine()
    results = engine.run_tournament(list(hypotheses), context=context, top_k=len(hypotheses))

    t = Table(title="Tournament Debate Leaderboard")
    t.add_column("Rank", style="cyan")
    t.add_column("Hypothesis", style="white")
    t.add_column("Verdict", style="yellow")
    t.add_column("Robustness", style="green")
    t.add_column("Survival", style="magenta")

    for i, r in enumerate(results, 1):
        status_str = (
            "[bold green]SURVIVED[/bold green]" if r.survived else "[bold red]ELIMINATED[/bold red]"
        )
        t.add_row(
            str(i), r.hypothesis[:50], r.verdict.value, f"{r.robustness_score:.2f}", status_str
        )
    console.print(t)


# ---------------------------------------------------------------------------
# Problem / Solution / Path vectorizing CLI
# ---------------------------------------------------------------------------


@main.group()
def vectorize():
    """Encode problems, solutions, and paths into vectors (custom embedder)."""
    pass


@vectorize.command(name="problem")
@click.argument("specification")
@click.option("--title", default="", help="Problem title")
@click.option("--goal", multiple=True, help="Goal criterion (repeatable)")
@click.option("--embedder", default=None, help="polarity|ollama|hybrid (default: env SUPER_SOLVER_EMBEDDER or polarity)")
@click.option("--no-store", is_flag=True, help="Do not persist to the vector index")
def vectorize_problem_cmd(specification: str, title: str, goal: List[str], embedder: Optional[str], no_store: bool):
    """Encode a problem specification into a vector."""
    from super_solver.vectorize import VectorizationService
    from super_solver.core.embedder import get_backend

    svc = VectorizationService(backend=get_backend(embedder))
    vec = svc.encode_problem(specification, title=title, goal_criteria=list(goal))
    console.print(Panel(f"[bold cyan]{title or 'Problem'}[/bold cyan]\n{specification}", title="Problem Vector"))
    console.print(f"[bold]Embedder:[/bold] {svc.backend.name}  [bold]Dim:[/bold] {vec.shape[0]}")
    if not no_store:
        pid = svc.store_problem(specification, title=title, goal_criteria=list(goal))
        console.print(f"[green]Stored id:[/green] {pid}")
    console.print(f"[dim]vector[:8] = {vec[:8].round(3).tolist()}[/dim]")


@vectorize.command(name="solution")
@click.argument("solution_text")
@click.option("--method", default="", help="Solution method")
@click.option("--domain", default="", help="Domain")
@click.option("--operator", multiple=True, help="Operator (repeatable)")
@click.option("--title", default="")
@click.option("--embedder", default=None)
@click.option("--no-store", is_flag=True)
def vectorize_solution_cmd(solution_text: str, method: str, domain: str, operator: List[str], title: str, embedder: Optional[str], no_store: bool):
    """Encode a solution into a vector."""
    from super_solver.vectorize import VectorizationService
    from super_solver.core.embedder import get_backend

    svc = VectorizationService(backend=get_backend(embedder))
    vec = svc.encode_solution(solution_text, method=method, domain=domain, operators=list(operator))
    console.print(Panel(f"[bold cyan]{title or 'Solution'}[/bold cyan]\n{solution_text}", title="Solution Vector"))
    console.print(f"[bold]Embedder:[/bold] {svc.backend.name}  [bold]Dim:[/bold] {vec.shape[0]}")
    if not no_store:
        sid = svc.store_solution(solution_text, method=method, domain=domain, operators=list(operator), title=title)
        console.print(f"[green]Stored id:[/green] {sid}")


@vectorize.command(name="path")
@click.argument("steps", nargs=-1, required=True)
@click.option("--operator", multiple=True, help="Operator type per step (repeatable)")
@click.option("--breakthrough", default="", help="Final breakthrough")
@click.option("--title", default="")
@click.option("--embedder", default=None)
@click.option("--no-store", is_flag=True)
def vectorize_path_cmd(steps: List[str], operator: List[str], breakthrough: str, title: str, embedder: Optional[str], no_store: bool):
    """Encode a discovery path (sequence of reasoning steps) into a vector."""
    from super_solver.vectorize import VectorizationService
    from super_solver.core.embedder import get_backend

    svc = VectorizationService(backend=get_backend(embedder))
    vec = svc.encode_path(list(steps), operator_types=list(operator) or None, final_breakthrough=breakthrough)
    console.print(Panel(f"[bold cyan]{title or 'Path'}[/bold cyan]\n{' -> '.join(steps)}", title="Path Vector"))
    console.print(f"[bold]Embedder:[/bold] {svc.backend.name}  [bold]Dim:[/bold] {vec.shape[0]}")
    if not no_store:
        pid = svc.store_path(list(steps), operator_types=list(operator) or None, final_breakthrough=breakthrough, title=title)
        console.print(f"[green]Stored id:[/green] {pid}")


@vectorize.command(name="search")
@click.argument("query")
@click.option("--kind", default=None, type=click.Choice(["problem", "solution", "path"]), help="Kind filter")
@click.option("--top-k", default=5, type=int)
@click.option("--min-sim", default=0.0, type=float)
@click.option("--embedder", default=None)
def vectorize_search_cmd(query: str, kind: Optional[str], top_k: int, min_sim: float, embedder: Optional[str]):
    """Search the vector index for entries similar to a query text."""
    from super_solver.vectorize import VectorizationService
    from super_solver.core.embedder import get_backend

    svc = VectorizationService(backend=get_backend(embedder))
    vec = svc.backend.encode(query)
    hits = svc.search(vec, kind=kind, top_k=top_k, min_similarity=min_sim)
    t = Table(title=f"Search: {query}")
    t.add_column("Kind", style="cyan")
    t.add_column("Title", style="green")
    t.add_column("Similarity", style="magenta")
    t.add_column("Content", style="white")
    for h in hits:
        t.add_row(h["kind"], h.get("title", ""), f"{h['similarity']:.3f}", h["content"][:60])
    console.print(t)


@vectorize.command(name="analogize")
@click.argument("specification")
@click.option("--title", default="")
@click.option("--top-k", default=3, type=int)
@click.option("--min-sim", default=0.0, type=float)
@click.option("--embedder", default=None)
def vectorize_analogize_cmd(specification: str, title: str, top_k: int, min_sim: float, embedder: Optional[str]):
    """Find similar past problems and return their linked solutions + paths."""
    from super_solver.vectorize import VectorizationService
    from super_solver.core.embedder import get_backend

    svc = VectorizationService(backend=get_backend(embedder))
    vec = svc.encode_problem(specification, title=title)
    analogs = svc.analogize(vec, top_k=top_k, min_similarity=min_sim)
    if not analogs:
        console.print("[yellow]No analogs found.[/yellow]")
        return
    for i, a in enumerate(analogs, 1):
        console.print(Panel(
            f"[bold]Problem:[/bold] {a['problem']['content'][:80]}\n"
            f"[bold]Similarity:[/bold] {a['problem']['similarity']:.3f}\n"
            f"[bold]Solution:[/bold] {a['solution']['content'] if a['solution'] else '(none)'}\n"
            f"[bold]Path:[/bold] {a['path']['content'] if a['path'] else '(none)'}",
            title=f"Analog #{i}",
        ))


if __name__ == "__main__":
    main()
