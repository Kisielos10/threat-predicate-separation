"""The three agent roles (reduced set from the architecture document).

  Planista  (Triage & Planning) — decomposes the incident into what must be checked.
  Wykonawca (Executor)          — gathers evidence in a ReAct loop over the deterministic tools.
  Audytor   (Evidence Auditor)  — cross-checks the evidence chain and issues the final decision;
                                  may send the case back once for missing evidence.

The Auditor's ability to reject and demand more evidence is the cross-check mechanism (W2) that
distinguishes the multi-agent configuration from the single-agent baseline.

Prompts and outputs are in Polish so the traces are directly presentable.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from .llm import LLMClient
from .policy import requirement_text
from .tools import Case, ToolContext, run_tool, tool_catalogue
from .trace import Decision, Step

OnStep = Callable[[Step], None] | None


def _emit(steps: list[Step], step: Step, on_step: OnStep) -> None:
    """Record a step and, when streaming, hand it to the caller immediately."""
    steps.append(step)
    if on_step:
        on_step(step)

_JSON_RULE = "Odpowiadasz WYŁĄCZNIE jednym obiektem JSON, bez komentarzy i bez markdown."

PLANNER_SYS = (
    "Jesteś Agentem Planistą w systemie wykrywania zagrożeń. Na podstawie opisu żądania HTTP "
    "planujesz, co należy sprawdzić. Nie masz dostępu do logów ani narzędzi i nie wydajesz "
    "werdyktu. " + _JSON_RULE
)

EXECUTOR_SYS = (
    "Jesteś Agentem Wykonawczym (ekspertem od warstwy webowej). Zbierasz dowody, wywołując "
    "narzędzia, a następnie streszczasz zebrane dowody i proponujesz wstępny werdykt. "
    "Opierasz się wyłącznie na tym, co zwróciły narzędzia. " + _JSON_RULE
)

AUDITOR_SYS = (
    "Jesteś Agentem Audytorem Dowodów. Weryfikujesz, czy łańcuch dowodowy jest spójny i "
    "wystarczający do podjęcia decyzji. Jeśli brakuje istotnego dowodu, możesz odesłać sprawę "
    "po uzupełnienie. Na końcu wydajesz ostateczną decyzję, ocenę wiarygodności i rekomendację "
    "reakcji. " + _JSON_RULE
)

SINGLE_SYS = (
    "Jesteś analitykiem bezpieczeństwa. Samodzielnie badasz żądanie HTTP, wywołując narzędzia, "
    "a następnie wydajesz ostateczną decyzję, ocenę wiarygodności i rekomendację reakcji. "
    + _JSON_RULE
)

_DECISION_RULES = (
    "Dozwolone decyzje: 'atak' (żądanie jest złośliwe), 'normalne' (ruch prawidłowy), "
    "'nieznane' (podejrzane, ale dowody niewystarczające). 'confidence' to liczba 0-1."
)


def plan(llm: LLMClient, case: Case, on_step: OnStep = None) -> tuple[str, Step]:
    """Planner: produce a short investigation plan."""
    t0 = time.perf_counter()
    out = llm.chat_json(PLANNER_SYS, (
        f"Zdarzenie do zbadania:\n{case.summary()}\n\n"
        f"Dostępne narzędzia dla wykonawcy:\n{tool_catalogue()}\n\n"
        'Zwróć JSON: {"plan": ["krok 1", "krok 2", ...], "uzasadnienie": "jedno zdanie"}'
    ))
    steps = out.get("plan") or []
    text = "; ".join(str(s) for s in steps) if isinstance(steps, list) else str(steps)
    why = str(out.get("uzasadnienie", ""))
    content = f"Plan: {text}" + (f" | {why}" if why else "")
    step = Step("Planista", "plan", content, seconds=time.perf_counter() - t0)
    if on_step:
        on_step(step)
    return text, step


def execute(llm: LLMClient, ctx: ToolContext, case: Case, plan_text: str,
            max_tool_calls: int = 4, extra_request: str | None = None,
            on_step: OnStep = None) -> tuple[dict, list[Step]]:
    """Executor: ReAct loop over the tools, then an evidence summary + preliminary verdict."""
    steps: list[Step] = []
    observations: list[str] = []
    used: set[str] = set()

    for _ in range(max_tool_calls):
        t0 = time.perf_counter()
        prompt = (
            f"Zdarzenie:\n{case.summary()}\n\nPlan śledztwa: {plan_text}\n"
            + (f"\nAudytor prosi dodatkowo o: {extra_request}\n" if extra_request else "")
            + f"\nNarzędzia:\n{tool_catalogue()}\n"
            + ("\nDotychczasowe obserwacje:\n" + "\n".join(observations) if observations else "")
            + '\n\nWybierz następny krok. Zwróć JSON: {"thought": "...", "action": '
              '"<nazwa narzędzia> albo finish", "argument": "<opcjonalny argument>"}'
        )
        out = llm.chat_json(EXECUTOR_SYS, prompt)
        thought = str(out.get("thought", ""))[:400]
        action = str(out.get("action", "finish")).strip()
        arg = str(out.get("argument", "") or "") or None
        _emit(steps, Step("Wykonawca", "thought", thought, seconds=time.perf_counter() - t0),
              on_step)
        if action == "finish" or action not in _tool_names():
            break
        if action in used:      # don't let it loop on the same tool
            observations.append(f"[{action}] (już wywołane, pomijam)")
            continue
        used.add(action)
        t1 = time.perf_counter()
        result = run_tool(action, ctx, case, arg)
        obs = f"[{action}] {result}"
        observations.append(obs)
        _emit(steps, Step("Wykonawca", "observation", obs[:600], tool=action,
                          seconds=time.perf_counter() - t1), on_step)

    t0 = time.perf_counter()
    out = llm.chat_json(EXECUTOR_SYS, (
        f"Zdarzenie:\n{case.summary()}\n\nZebrane dowody:\n" + "\n".join(observations)
        + f"\n\n{_DECISION_RULES}\n"
        'Zwróć JSON: {"dowody": "streszczenie zebranych dowodów", '
        '"wstepna_decyzja": "atak|normalne|nieznane", "confidence": 0-1}'
    ))
    _emit(steps, Step("Wykonawca", "verdict",
                      f"Dowody: {out.get('dowody','')} | wstępnie: {out.get('wstepna_decyzja','')} "
                      f"({out.get('confidence','?')})", seconds=time.perf_counter() - t0), on_step)
    out["_observations"] = observations
    out["_tools_used"] = sorted(used)      # needed for the W4 evidentiary check
    return out, steps


def audit(llm: LLMClient, case: Case, evidence: dict, allow_followup: bool,
          on_step: OnStep = None) -> tuple[dict, Step]:
    """Auditor: validate the evidence chain and issue the final triple (y, c, r).

    The auditor also states explicitly whether each condition of the threat predicate T(S)
    (Definition 5) holds: W1 observable effect, W2 potential CIA violation, W3 non-normative
    origin. This makes the verdict traceable to the formal definition rather than to an opaque
    judgement, and yields an estimate of the impact vector sigma.
    """
    t0 = time.perf_counter()
    followup = ('Jeśli dowody są niewystarczające, możesz zwrócić "werdykt": "uzupelnij" '
                'wraz z polem "brakuje". ' if allow_followup else
                'Musisz teraz podjąć ostateczną decyzję (nie możesz już prosić o uzupełnienie). ')
    out = llm.chat_json(AUDITOR_SYS, (
        f"Zdarzenie:\n{case.summary()}\n\n"
        f"Dowody od wykonawcy:\n{evidence.get('dowody','')}\n"
        "Obserwacje narzędzi:\n" + "\n".join(evidence.get("_observations", []))
        + f"\n\nWstępna decyzja wykonawcy: {evidence.get('wstepna_decyzja','?')} "
          f"(confidence {evidence.get('confidence','?')})\n\n"
        f"{requirement_text()}\n\n{_DECISION_RULES}\n{followup}"
        "Oceń też warunki predykatu zagrożenia (Definicja 5): W1 obserwowalny efekt odbiegający "
        "od normy, W2 potencjał naruszenia poufności/integralności/dostępności, W3 pochodzenie "
        "pozanormatywne (celowe działanie, a nie awaria).\n"
        'Zwróć JSON: {"werdykt": "zatwierdzam|uzupelnij", "brakuje": "...", '
        '"decision": "atak|normalne|nieznane", "confidence": 0-1, '
        '"W1": true/false, "W2": true/false, "W3": true/false, '
        '"sigma": [0-1, 0-1, 0-1], '
        '"recommendation": "rekomendowane działanie, jedno zdanie"}'
    ))
    verdict = str(out.get("werdykt", "zatwierdzam")).strip().lower()
    conds = "".join(f" {w}={'T' if out.get(w) else 'F'}" for w in ("W1", "W2", "W3"))
    content = (f"Werdykt: {verdict}"
               + (f" | brakuje: {out.get('brakuje','')}" if verdict == "uzupelnij" else
                  f" | decyzja: {out.get('decision','?')} ({out.get('confidence','?')}) |{conds}"))
    step = Step("Audytor", "audit", content, seconds=time.perf_counter() - t0)
    if on_step:
        on_step(step)
    return out, step


def single_agent(llm: LLMClient, ctx: ToolContext, case: Case,
                 max_tool_calls: int = 4) -> tuple[Decision, list[Step]]:
    """Baseline: one agent, same tools, no role separation and no cross-check."""
    steps: list[Step] = []
    observations: list[str] = []
    used: set[str] = set()

    for _ in range(max_tool_calls):
        t0 = time.perf_counter()
        prompt = (
            f"Zdarzenie:\n{case.summary()}\n\nNarzędzia:\n{tool_catalogue()}\n"
            + ("\nDotychczasowe obserwacje:\n" + "\n".join(observations) if observations else "")
            + '\n\nWybierz następny krok. Zwróć JSON: {"thought": "...", "action": '
              '"<nazwa narzędzia> albo finish", "argument": "<opcjonalny argument>"}'
        )
        out = llm.chat_json(SINGLE_SYS, prompt)
        steps.append(Step("Agent", "thought", str(out.get("thought", ""))[:400],
                          seconds=time.perf_counter() - t0))
        action = str(out.get("action", "finish")).strip()
        if action == "finish" or action not in _tool_names() or action in used:
            if action in used:
                continue
            break
        used.add(action)
        t1 = time.perf_counter()
        result = run_tool(action, ctx, case, str(out.get("argument", "") or "") or None)
        observations.append(f"[{action}] {result}")
        steps.append(Step("Agent", "observation", f"[{action}] {result}"[:600], tool=action,
                          seconds=time.perf_counter() - t1))

    t0 = time.perf_counter()
    out = llm.chat_json(SINGLE_SYS, (
        f"Zdarzenie:\n{case.summary()}\n\nZebrane dowody:\n" + "\n".join(observations)
        + f"\n\n{_DECISION_RULES}\n"
        'Zwróć JSON: {"decision": "atak|normalne|nieznane", "confidence": 0-1, '
        '"recommendation": "rekomendowane działanie, jedno zdanie"}'
    ))
    steps.append(Step("Agent", "verdict",
                      f"{out.get('decision','?')} ({out.get('confidence','?')})",
                      seconds=time.perf_counter() - t0))
    dec = Decision(out.get("decision", "nieznane"), out.get("confidence", 0.0),
                   str(out.get("recommendation", ""))).normalised()
    return dec, steps


def _tool_names() -> set[str]:
    from .tools import TOOLS
    return set(TOOLS)
