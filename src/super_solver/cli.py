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
    console.print(Panel(f"[bold cyan]{problem_title}[/bold cyan]\n{spec}", title="Problem Formulation"))
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
    console.print(Panel("[bold yellow]Initiating Autonomous Engine Self-Improvement Cycle[/bold yellow]", title="Meta-Optimization"))
    engine = SuperDuperProblemSolvingEngine()
    audit = engine.run_self_audit_and_optimization()

    t = Table(title="Self-Audit & Formal Invariant Results")
    t.add_column("Module", style="cyan")
    t.add_column("Verification / Metric", style="green")
    t.add_column("Status / Verdict", style="magenta")

    t.add_row("DPLL Logic Solver", "Sound Trajectory Invariant", audit["dpll_invariant_verification"]["verdict"])
    t.add_row("Causal VSA (Pearl SCM)", f"Repulsion -> Success ACE: +{audit['causal_intervention_audit']['average_causal_effect']:.2f}", "EFFECTIVE")
    t.add_row("BOED Designer", f"Entropy EIG: {audit['boed_tuning_utility']['expected_information_gain']:.3f}", "OPTIMAL_UTILITY")
    console.print(t)

    console.print("[bold green]Executing Self-Optimization Trajectory...[/bold green]")
    discovery = engine.improve_self()
    console.print(f"[bold white]Outcome:[/bold white] {discovery.final_breakthrough}")
    console.print(f"[bold cyan]Tuned Parameters:[/bold cyan] {audit['tuned_parameters']}")


@main.command(name="consolidate-memory")
@click.option("--threshold", default=0.75, type=float, help="Similarity threshold for hazard cluster consolidation")
def consolidate_memory_cmd(threshold: float):
    """Prunes and consolidates redundant negative manifold repulsors into compact centroid schemas."""
    engine = SuperDuperProblemSolvingEngine()
    pruned = engine.consolidate_memory(similarity_threshold=threshold)
    console.print(f"[bold green]Synaptic Consolidation Complete:[/bold green] Pruned {pruned} redundant vectors.")


@main.command(name="tournament")
@click.option("--hyp", "hypotheses", multiple=True, required=True, help="Candidate hypothesis to debate")
@click.option("--context", default=None, help="Background context or paper abstract")
def tournament_cmd(hypotheses: List[str], context: Optional[str]):
    """Runs an adversarial Red Team vs Blue Team dialectical tournament across hypotheses."""
    console.print(Panel("[bold red]Red Team[/bold red] vs [bold blue]Blue Team[/bold blue] Dialectical Tournament", title="Adversarial Cross-Examination"))
    engine = SuperDuperProblemSolvingEngine()
    results = engine.run_tournament(list(hypotheses), context=context, top_k=len(hypotheses))

    t = Table(title="Tournament Debate Leaderboard")
    t.add_column("Rank", style="cyan")
    t.add_column("Hypothesis", style="white")
    t.add_column("Verdict", style="yellow")
    t.add_column("Robustness", style="green")
    t.add_column("Survival", style="magenta")

    for i, r in enumerate(results, 1):
        status_str = "[bold green]SURVIVED[/bold green]" if r.survived else "[bold red]ELIMINATED[/bold red]"
        t.add_row(str(i), r.hypothesis[:50], r.verdict.value, f"{r.robustness_score:.2f}", status_str)
    console.print(t)


if __name__ == "__main__":
    main()


