# Threat as a Checkable Predicate

Code and results for the paper *Threat as a Checkable Predicate: What Anomaly-Based Web Detection
Omits*.

## What this measures

An anomaly-based intrusion detector reports that a request is **unusual**. What matters
operationally is whether it is **harmful**. The two properties are routinely conflated because the
corpora used to evaluate these systems contain only two categories, ordinary normal traffic and
attacks, with nothing in between. Traffic that is unusual *and* legitimate is absent, so a
detector's inability to separate the two is invisible under standard evaluation.

This repository constructs the missing category and measures what happens.

The central quantity is **separation**: the rate at which a criterion fires on attacks minus the
rate at which it fires on legitimate-but-unusual traffic. A detector that distinguishes harm from
mere strangeness has high separation. A detector that only measures strangeness has separation near
zero regardless of how well it scores on a conventional benchmark.

Headline result: across six unsupervised detectors, including the Kruegel and Vigna
character-distribution model and PAYL implemented as published, separation never exceeds **+0.12**
and is negative for three of them. The same six methods separate attacks from *ordinary* traffic by
0.32 to 0.72.

## Constructing the probe set

Legitimate-but-unusual traffic does not exist as a labelled class in any public corpus, so it is
generated (`src/zeroday_verify/probes.py`). The procedure starts from requests the corpus labels
normal and substitutes **the value of a single free-text field** with content that is legitimate for
the application but superficially alarming: surnames containing apostrophes, ordinary Spanish words
that are also SQL keywords, accented characters, punctuation inside passwords.

Path, method, parameter names and every other value are left untouched.

```
POST /tienda1/publico/anadir.jsp
id=1&nombre=Jam%C3%B3n+Ib%C3%A9rico&precio=39&cantidad=91&B1=A%C3%B1adir+al+carrito   <- corpus-normal

POST /tienda1/publico/anadir.jsp
id=1&nombre=Union+Cooperativa&precio=39&cantidad=91&B1=A%C3%B1adir+al+carrito         <- probe
```

Three controls make the construction defensible, and all three are enforced in code rather than
asserted in prose:

- `_assert_pool_is_benign()` runs at import and rejects any pooled value that could close a quoted
  context, comment out a statement, introduce markup or traverse a path.
- `CATEGORY_TARGETS` restricts substitution to free-text fields. An earlier generator wrote into
  control parameters such as `B1`, which is parameter tampering, and therefore produced attacks
  labelled benign.
- `run_probe_validation.py` passes unmodified requests through the same parse-and-rebuild path and
  confirms detector flag rates change by 0.00, so the reconstruction itself is not what detectors
  react to.

An earlier probe set used paths absent from the corpus and was separable by path alone. Both failure
modes are documented in the paper because neither was visible in aggregate statistics.

## Layout

| Path | Contents |
|---|---|
| `src/zeroday_verify/probes.py` | probe generation and the benign-pool guard |
| `src/zeroday_verify/conditions.py` | independent per-condition assessment (W1, W2, W3) |
| `src/zeroday_verify/baselines.py` | Kruegel and Vigna per-(path, parameter) ICD; PAYL |
| `src/zeroday_verify/agents/structure.py` | structural profile of normal traffic |
| `src/zeroday_verify/agents/` | the three-role multi-agent system and its tool layer |
| `src/zeroday_verify/graph/` | threat relation graph used for the W2 retrieval estimate |
| `src/zeroday_verify/{novelty,similarity,embedding}.py` | the reference novelty detector |
| `results/` | result artefacts backing every number in the paper |

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
pytest
```

### Data

Corpora are not redistributed here. Download them into `data/`:

- **CSIC-2010** HTTP requests to `data/raw_text/`, from the Hugging Face dataset
  `bridge4/CSIC2010_dataset_classification`.
- **CIC-IDS2017** flows to `data/raw_full/`, from `Mireu-Lab/CIC-IDS`
  (`ZIP/GeneratedLabelledFlows.zip`), needed only for the network-flow comparison and the graph.

### Experiments

The detector comparison and the flow-domain control need no language model and finish in minutes:

```bash
python run_baseline_check.py
```

```bash
python run_flow_check.py
```

The condition assessments call a language model once per condition per case. Local models are
served through [Ollama](https://ollama.com); the hosted model reads `ZDV_LLM_API_KEY` from the
environment and is never stored in this repository.

```bash
ollama pull qwen2.5:7b && python run_conditions_check.py 40 qwen2.5:7b --generated
```

The ablation that withholds the detector's score from the evidence bundle:

```bash
python run_conditions_check.py 40 qwen2.5:7b --generated --no-novelty
```

The multi-agent comparison is the expensive one, roughly eight hours for three seeds, because
Ollama serves requests one at a time:

```bash
python run_agents_eval.py 75 75 3
```

Regenerating the reported numbers and figures from artefacts already in `results/`, without
re-running any experiment:

```bash
python consolidate_results.py && python make_paper_figures.py
```

## Licence

MIT. See `LICENSE`.
