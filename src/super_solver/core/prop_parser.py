"""Precedence-aware propositional formula parser producing CNF clauses.

Why this module exists
----------------------
The zero-daemon DPLL prover used to extract clauses from premise strings with
flat regex splits -- ``re.split(r"\\s+(?:AND|and|\\^|&|\\u2227)\\s+")`` for
conjunction, ``"->" in s`` for implication, and so on.  Those splits know
nothing about parentheses or operator precedence, so **every grouped formula
was silently mis-parsed into garbage atoms**.  Measured before this module
existed:

    premises: ["(A AND B) -> C", "A", "B"]   target: "C"
      -> countermodel {"(a": true, "a": true, "b": true, "b)": false, "c": false}
      -> verdict COUNTERMODEL_EXISTS          # wrong: the argument is valid

    premises: ["P iff (Q AND R)", "Q", "R"]  target: "P"
      -> countermodel {"r)": true, "q": true, "r": true, "p": false, "(q": false}
      -> verdict COUNTERMODEL_EXISTS          # wrong: the argument is valid

The atoms ``"(a"``, ``"b)"``, ``"(q"``, ``"r)"`` are the tell: the conjunction
splitter cut straight through the parentheses.  The failure mode was the bad
one -- not a graceful "inconclusive", but a *confident* ``COUNTERMODEL_EXISTS``
claiming a valid argument is not entailed.

This module replaces that layer with a small recursive-descent parser:

    iff     := impl (IFF impl)*
    impl    := or (IMPL impl)?        # right associative
    or      := and (OR and)*
    and     := unary (AND unary)*
    unary   := NOT unary | primary
    primary := '(' iff ')' | ATOM

Formulas are then rewritten (eliminating ``<->`` and ``->``), pushed to
negation normal form, and distributed into CNF.  Tautological clauses are
dropped, which is what keeps the CNF for ``A <-> B`` identical to the
hand-rolled one the solver produced before.

Atom naming is deliberately *not* done here: clauses carry ``(negated, text)``
literals so the caller keeps its own ``get_var`` normalisation and symbol
table.  Anything this parser cannot handle raises :class:`ParseError`, which
lets the caller fall back to the legacy regex layer rather than guess.

Pure Python, standard library only, deterministic.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

__all__ = ["ParseError", "FormulaTooLarge", "tokenize", "parse_cnf", "MAX_NESTING"]

# Literal = (is_negated, atom_text); Clause = list of literals; CNF = list of clauses.
Literal = Tuple[bool, str]
Clause = List[Literal]
Node = tuple  # ("atom"|"not"|"and"|"or"|"impl"|"iff", ...)

MAX_NESTING = 200
DEFAULT_CLAUSE_CAP = 20000


class ParseError(ValueError):
    """Raised when the input is not a well-formed propositional formula."""


class FormulaTooLarge(ParseError):
    """Raised when CNF distribution would exceed the clause budget."""


# --------------------------------------------------------------------------
# Tokenizer
# --------------------------------------------------------------------------

# Multi-character symbols are matched before single-character ones.
_SYMBOL_OPS: Sequence[Tuple[str, str]] = (
    ("<=>", "IFF"),
    ("<->", "IFF"),
    ("⇔", "IFF"),  # ⇔
    ("↔", "IFF"),  # ↔
    ("≡", "IFF"),  # ≡
    ("->", "IMPL"),
    ("=>", "IMPL"),
    ("⇒", "IMPL"),  # ⇒
    ("→", "IMPL"),  # →
    ("∧", "AND"),  # ∧
    ("∨", "OR"),  # ∨
    ("¬", "NOT"),  # ¬
    ("∼", "NOT"),  # ∼
)

# Word operators.  These must be delimited by non-word characters so that
# e.g. "greater_or_equal" is not read as a disjunction.
_WORD_OPS: Sequence[Tuple[str, str]] = (
    ("if and only if", "IFF"),
    ("iff", "IFF"),
    ("and", "AND"),
    ("or", "OR"),
    ("not", "NOT"),
)

# Symbolic operators that the legacy splitter required whitespace around; kept
# whitespace-delimited for bug-compatibility so "a^b"/"AT&T" stay atoms.
_SPACED_SYMBOL_OPS: Sequence[Tuple[str, str]] = (
    ("^", "AND"),
    ("&", "AND"),
)


def _is_word_char(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


def tokenize(text: str) -> List[Tuple[str, str]]:
    """Split ``text`` into ``(kind, value)`` tokens.

    ``kind`` is one of ``"("``, ``")"``, ``"AND"``, ``"OR"``, ``"NOT"``,
    ``"IMPL"``, ``"IFF"``, ``"ATOM"``.

    Raises :class:`ParseError` on unbalanced parentheses or an empty formula.
    """
    tokens: List[Tuple[str, str]] = []
    i = 0
    n = len(text)
    atom_start: Optional[int] = None
    # True while an operand is expected -- only then may a bare "-" mean NOT.
    expect_operand = True

    def flush(end: int) -> None:
        nonlocal atom_start
        if atom_start is not None:
            raw = text[atom_start:end].strip()
            if raw:
                tokens.append(("ATOM", raw))
            atom_start = None

    while i < n:
        ch = text[i]

        matched = False

        # Parentheses always break an atom run.
        if ch in "()":
            flush(i)
            tokens.append((ch, ch))
            expect_operand = ch == "("
            i += 1
            continue

        # Symbolic operators.
        for sym, kind in _SYMBOL_OPS:
            if text.startswith(sym, i):
                flush(i)
                tokens.append((kind, sym))
                expect_operand = True
                i += len(sym)
                matched = True
                break
        if matched:
            continue

        # Whitespace-delimited symbolic operators ("^", "&").
        for sym, kind in _SPACED_SYMBOL_OPS:
            if text.startswith(sym, i):
                prev_ok = i == 0 or text[i - 1].isspace() or text[i - 1] == "("
                nxt = i + len(sym)
                next_ok = nxt >= n or text[nxt].isspace() or text[nxt] == ")"
                if prev_ok and next_ok:
                    flush(i)
                    tokens.append((kind, sym))
                    expect_operand = True
                    i += len(sym)
                    matched = True
                    break
        if matched:
            continue

        # Bare "-" as NOT: only where an operand is expected, and never as part
        # of "->" (already consumed above) or an in-progress atom such as "-1".
        if ch == "-" and expect_operand and atom_start is None:
            if i + 1 < n and text[i + 1] not in "> ":
                tokens.append(("NOT", "-"))
                expect_operand = True
                i += 1
                continue

        # Word operators (case-insensitive, non-word delimited).
        lower = text[i:].lower()
        for word, kind in _WORD_OPS:
            if lower.startswith(word):
                end = i + len(word)
                prev_ok = i == 0 or not _is_word_char(text[i - 1])
                next_ok = end >= n or not _is_word_char(text[end])
                if prev_ok and next_ok:
                    flush(i)
                    tokens.append((kind, text[i:end]))
                    expect_operand = True
                    i = end
                    matched = True
                    break
        if matched:
            continue

        # Ordinary character: part of an atom.
        if atom_start is None:
            atom_start = i
        expect_operand = False
        i += 1

    flush(n)

    if not tokens:
        raise ParseError("empty formula")

    depth = 0
    for kind, _ in tokens:
        if kind == "(":
            depth += 1
        elif kind == ")":
            depth -= 1
            if depth < 0:
                raise ParseError("unbalanced parentheses")
    if depth != 0:
        raise ParseError("unbalanced parentheses")

    return tokens


# --------------------------------------------------------------------------
# Recursive-descent parser
# --------------------------------------------------------------------------


class _Parser:
    def __init__(self, tokens: List[Tuple[str, str]]) -> None:
        self.toks = tokens
        self.i = 0
        self.depth = 0

    def _peek(self) -> Optional[str]:
        return self.toks[self.i][0] if self.i < len(self.toks) else None

    def _advance(self) -> Tuple[str, str]:
        tok = self.toks[self.i]
        self.i += 1
        return tok

    def parse(self) -> Node:
        node = self._parse_iff()
        if self.i != len(self.toks):
            raise ParseError(f"unexpected trailing token {self.toks[self.i]!r}")
        return node

    def _enter(self) -> None:
        self.depth += 1
        if self.depth > MAX_NESTING:
            raise ParseError("formula nesting too deep")

    def _parse_iff(self) -> Node:
        self._enter()
        node = self._parse_impl()
        while self._peek() == "IFF":
            self._advance()
            node = ("iff", node, self._parse_impl())
        self.depth -= 1
        return node

    def _parse_impl(self) -> Node:
        self._enter()
        node = self._parse_or()
        if self._peek() == "IMPL":
            self._advance()
            node = ("impl", node, self._parse_impl())  # right associative
        self.depth -= 1
        return node

    def _parse_or(self) -> Node:
        self._enter()
        node = self._parse_and()
        while self._peek() == "OR":
            self._advance()
            node = ("or", node, self._parse_and())
        self.depth -= 1
        return node

    def _parse_and(self) -> Node:
        self._enter()
        node = self._parse_unary()
        while self._peek() == "AND":
            self._advance()
            node = ("and", node, self._parse_unary())
        self.depth -= 1
        return node

    def _parse_unary(self) -> Node:
        self._enter()
        if self._peek() == "NOT":
            self._advance()
            node = ("not", self._parse_unary())
        else:
            node = self._parse_primary()
        self.depth -= 1
        return node

    def _parse_primary(self) -> Node:
        kind = self._peek()
        if kind == "(":
            self._advance()
            node = self._parse_iff()
            if self._peek() != ")":
                raise ParseError("missing closing parenthesis")
            self._advance()
            return node
        if kind == "ATOM":
            return ("atom", self._advance()[1])
        raise ParseError(f"expected an atom or '(' but found {kind!r}")


# --------------------------------------------------------------------------
# Rewriting to CNF
# --------------------------------------------------------------------------


def _eliminate(node: Node) -> Node:
    """Rewrite ``<->`` and ``->`` away, leaving atom/not/and/or."""
    kind = node[0]
    if kind == "atom":
        return node
    if kind == "not":
        return ("not", _eliminate(node[1]))
    if kind in ("and", "or"):
        return (kind, _eliminate(node[1]), _eliminate(node[2]))
    if kind == "impl":
        a, b = _eliminate(node[1]), _eliminate(node[2])
        return ("or", ("not", a), b)
    if kind == "iff":
        a, b = _eliminate(node[1]), _eliminate(node[2])
        # (A -> B) AND (B -> A)
        return (
            "and",
            ("or", ("not", a), b),
            ("or", ("not", b), a),
        )
    raise ParseError(f"unknown node kind {kind!r}")


def _nnf(node: Node, neg: bool = False) -> Node:
    """Push negations down to the literals (negation normal form)."""
    kind = node[0]
    if kind == "atom":
        return ("not", node) if neg else node
    if kind == "not":
        return _nnf(node[1], not neg)
    if kind == "and":
        op = "or" if neg else "and"
        return (op, _nnf(node[1], neg), _nnf(node[2], neg))
    if kind == "or":
        op = "and" if neg else "or"
        return (op, _nnf(node[1], neg), _nnf(node[2], neg))
    raise ParseError(f"unexpected node {kind!r} in NNF conversion")


def _simplify_clause(lits: Clause) -> Optional[Clause]:
    """Drop duplicate literals; return None if the clause is a tautology."""
    seen = set()
    out: Clause = []
    for neg, atom in lits:
        if (not neg, atom) in seen:
            return None  # (x OR ~x) -- always true, discard
        if (neg, atom) not in seen:
            seen.add((neg, atom))
            out.append((neg, atom))
    return out


def _cnf(node: Node, cap: int, budget: List[int]) -> List[Clause]:
    kind = node[0]
    if kind == "atom":
        return [[(False, node[1])]]
    if kind == "not":
        # After NNF a negation only ever wraps an atom.
        if node[1][0] == "atom":
            return [[(True, node[1][1])]]
        raise ParseError("negation not in normal form")
    if kind == "and":
        return _cnf(node[1], cap, budget) + _cnf(node[2], cap, budget)
    if kind == "or":
        left = _cnf(node[1], cap, budget)
        right = _cnf(node[2], cap, budget)
        budget[0] += len(left) * len(right)
        if budget[0] > cap:
            raise FormulaTooLarge(
                f"CNF distribution exceeded the {cap}-clause budget"
            )
        combined: List[Clause] = []
        for cl in left:
            for cr in right:
                simplified = _simplify_clause(cl + cr)
                if simplified is not None:
                    combined.append(simplified)
        return combined
    raise ParseError(f"unexpected node {kind!r} in CNF conversion")


def parse_cnf(text: str, clause_cap: int = DEFAULT_CLAUSE_CAP) -> List[Clause]:
    """Parse ``text`` as a propositional formula and return its CNF clauses.

    Each clause is a list of ``(is_negated, atom_text)`` literals.  Tautological
    clauses are dropped.

    Raises :class:`ParseError` (including :class:`FormulaTooLarge`) when the
    input is not a well-formed formula, so the caller can fall back rather than
    guess.
    """
    tokens = tokenize(text)
    node = _Parser(tokens).parse()
    node = _eliminate(node)
    node = _nnf(node)
    clauses = _cnf(node, clause_cap, [0])
    return [c for c in clauses if c]
