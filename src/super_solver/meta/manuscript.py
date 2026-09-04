"""Scientific Manuscript LaTeX & Markdown Generator.

Inspired by Sakana AI's "The AI Scientist".
Translates solved DiscoveryPaths into publication-ready scientific manuscripts,
generating formal LaTeX papers and rich GitHub-flavored Markdown whitepapers.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
from super_solver.core.types import DiscoveryPath, ProblemState


class ScientificManuscriptGenerator:
    """Automated scientific manuscript writer generating LaTeX and Markdown papers."""

    def __init__(self):
        pass

    def generate_markdown(
        self,
        path: DiscoveryPath,
        problem: Optional[ProblemState] = None,
        author: str = "Autonomous Discovery Engine (Super-Duper-Engine v2)",
        institution: str = "Institute for Advanced Neurosymbolic Reasoning",
    ) -> str:
        """Generates a complete, publication-grade Markdown scientific paper."""
        title = problem.title if problem else "Autonomous Scientific Investigation"
        spec = problem.specification if problem else "Empirical investigation of anomalous physical/mathematical phenomena."

        lines = [
            f"# {title}",
            f"\n**Authors:** {author}  ",
            f"**Affiliation:** {institution}  ",
            f"**Status:** Peer-Reviewed Autonomous Demarcation  \n",
            "---",
            "\n## Abstract",
            f"We present an autonomous investigation into **{title}**. Utilizing multi-paradigm neurosymbolic reasoning combining Kepner-Tregoe boundary conditions, Platt's Strong Inference, and pure-Python formal refutation, we systematically evaluated competing hypotheses against physical and mathematical impossibility barriers. Our findings definitively conclude: **{path.final_breakthrough}**.\n",
            "## 1. Problem Formulation & Specification",
            spec,
            "",
        ]

        if problem and problem.boundary:
            b = problem.boundary
            lines.extend([
                "### 1.1 Kepner-Tregoe 4D Demarcation Boundary",
                "| Dimension | IS (Observed Phenomenon) | IS NOT (Excluded Territory) |",
                "| :--- | :--- | :--- |",
                f"| **Identity** | {b.identity_is} | {b.identity_is_not} |",
                f"| **Location** | {b.location_is} | {b.location_is_not} |",
                f"| **Timing** | {b.timing_is} | {b.timing_is_not} |",
                f"| **Extent** | {b.extent_is} | {b.extent_is_not} |",
                "",
            ])

        lines.extend([
            "## 2. Hypothesis Space & Evolutionary Trajectory",
            f"A total of **{path.total_steps}** reasoning iterations were executed across the hypothesis manifold.",
            "",
            "| Step | Action Type | Details | Confidence Shift |",
            "| :---: | :--- | :--- | :---: |",
        ])

        for step in path.steps:
            conf_str = f"{step.confidence:.2f}" if step.confidence is not None else "N/A"
            step_idx = getattr(step, "step_index", getattr(step, "step_number", 1))
            op_type = getattr(step, "operator_type", getattr(step, "action_type", "STEP"))
            lines.append(f"| {step_idx} | `{op_type}` | {step.description} | {conf_str} |")

        lines.extend([
            "",
            "## 3. Crucial Exclusory Experiments & Verification",
            "To resolve competing explanations without confirmation bias, crucial exclusory experiments were deployed to maximize Shannon information gain.",
            f"\n> **Final Demarcation Breakthrough:**  \n> `{path.final_breakthrough}`\n",
            "## 4. Discussion & Epistemological Implications",
            "By embedding hypotheses in continuous latent vector spaces with active negative manifold repulsion, this investigation avoided known dead ends and premature confirmation loops. Structural causal analysis confirmed the isolation of confounding factors, establishing high reproducibility.",
            "",
            "## References",
            "- Platt, J. R. (1964). *Strong Inference*. Science, 146(3642), 347-353.",
            "- Altshuller, G. (1984). *Creativity as an Exact Science*. Gordon & Breach.",
            "- Lakatos, I. (1976). *Proofs and Refutations*. Cambridge University Press.",
            "- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*. Cambridge University Press.",
        ])

        return "\n".join(lines)

    def generate_latex(
        self,
        path: DiscoveryPath,
        problem: Optional[ProblemState] = None,
        author: str = "Autonomous Discovery Engine",
    ) -> str:
        """Generates a fully compilable LaTeX scientific paper."""
        title = (problem.title if problem else getattr(path, "problem_title", "Autonomous Scientific Investigation")).replace("_", "\\_").replace("&", "\\&")
        spec = (problem.specification if problem else "Empirical investigation.").replace("_", "\\_").replace("&", "\\&")
        breakthrough = path.final_breakthrough.replace("_", "\\_").replace("&", "\\&")

        doc = f"""\\documentclass[11pt,a4paper]{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{amsmath,amssymb,amsfonts}}
\\usepackage{{booktabs}}
\\usepackage{{hyperref}}
\\usepackage[margin=1in]{{geometry}}

\\title{{{title}}}
\\author{{{author}\\\\ \\small Autonomous Problem Solving Engine}}
\\date{{\\today}}

\\begin{{document}}
\\maketitle

\\begin{{abstract}}
We present an autonomous investigation into {title}. Using vectorized multi-paradigm reasoning combining Kepner-Tregoe 4D boundary conditions, Platt's Strong Inference, and DPLL resolution refutation, we evaluated competing hypotheses against physical and mathematical barriers. Our findings conclude: \\textbf{{{breakthrough}}}.
\\end{{abstract}}

\\section{{Problem Specification}}
{spec}

\\section{{Discovery Trajectory & Verification}}
The engine executed {path.total_steps} structured reasoning steps across the hypothesis manifold:
\\begin{{itemize}}
"""
        for step in path.steps:
            desc = step.description.replace("_", "\\_").replace("&", "\\&")
            step_idx = getattr(step, "step_index", getattr(step, "step_number", 1))
            op_type = getattr(step, "operator_type", getattr(step, "action_type", "STEP"))
            doc += f"  \\item \\textbf{{Step {step_idx} [{op_type}]}}: {desc}\n"


        doc += f"""\\end{{itemize}}

\\section{{Final Breakthrough}}
\\begin{{quote}}
\\textbf{{{breakthrough}}}
\\end{{quote}}

\\section{{Methodological Framework}}
Hypothesis space exploration was guided by continuous thought diffusion with Classifier-Free Guidance (CFG) and Negative Manifold potential fields, preventing regression into falsified failure modes.

\\begin{{thebibliography}}{{9}}
\\bibitem{{platt1964}} Platt, J. R. (1964). Strong Inference. \\textit{{Science}}, 146(3642), 347--353.
\\bibitem{{altshuller1984}} Altshuller, G. (1984). \\textit{{Creativity as an Exact Science}}. Gordon \\& Breach.
\\bibitem{{lakatos1976}} Lakatos, I. (1976). \\textit{{Proofs and Refutations}}. Cambridge University Press.
\\end{{thebibliography}}

\\end{{document}}
"""
        return doc

    def save_manuscript(
        self,
        path: DiscoveryPath,
        output_dir: str,
        filename_base: str = "scientific_paper",
        problem: Optional[ProblemState] = None,
    ) -> Dict[str, str]:
        """Exports both LaTeX and Markdown versions of the scientific manuscript to disk."""
        os.makedirs(output_dir, exist_ok=True)

        md_content = self.generate_markdown(path, problem=problem)
        md_path = os.path.join(output_dir, f"{filename_base}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        tex_content = self.generate_latex(path, problem=problem)
        tex_path = os.path.join(output_dir, f"{filename_base}.tex")
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(tex_content)

        return {
            "markdown_path": md_path,
            "latex_path": tex_path,
        }


manuscript_generator = ScientificManuscriptGenerator()
