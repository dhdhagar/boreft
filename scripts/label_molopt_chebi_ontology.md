# molopt semantic categories from the ChEBI ontology

Companion to `label_molopt_chebi_ontology.py`. This is the *why*; the script is the
*what*.

## What this buys

Semantle's `data/semantle/train/definitions.jsonl` carries a `category_normalized`
field per word. `plot_cluster_pca` colours the per-target bias PCA by it, and the
subspace explorer builds its clickable legend from it. Both of those paths are
task-generic already:

```164:193:src/boreft/eval/plot_cluster_pca.py
def definitions_path_from_config(output_dir: str) -> str:
    cfg = load_merged_run_config(output_dir)
    path = cfg.get("definitions_path") or default_definitions_path(
        str(cfg.get("task", "semantle"))
    )
    return os.path.abspath(str(path))


def uses_definition_embed_categories(output_dir: str) -> bool:
    return bool(load_merged_run_config(output_dir).get("use_definition_embeds"))
```

Any run trained with `--use-definition-embeds` takes the `label_source="category"`
branch and reads `data/<task>/train/definitions.jsonl`. molopt runs were already
taking that branch — they just found no category field and coloured all 8000
molecules `unknown`. Adding the field is the entire integration; no code downstream
changed.

## Why the ontology instead of an LLM

ChEBI-20 is not a generic SMILES corpus. Every molecule in it *is* a ChEBI entry,
and its description is a paraphrase of that entry's position in the ChEBI ontology
("a member of the class of biphenyls that is benzidine in which..."). So the class
labels already exist upstream, curated by chemists, in a machine-readable DAG. An
LLM pass over the descriptions would be reconstructing, lossily and
unreproducibly, something we can just look up.

The ontology also gives two orthogonal axes for free, which prose does not
separate cleanly:

- **`is_a`** — what the molecule *is*. Structural taxonomy.
- **`has role`** — what the molecule is *for*. A separate ontology of biological
  and application roles, reached by a labelled edge.

Both are written per molecule.

## Pipeline

### 1. Join by InChIKey

`chebi_core.obo.gz` annotates most terms with an InChIKey, and
`prepare_molopt_chebi20.py` already wrote one per corpus row, so the join needs no
RDKit and no network round-trip per molecule.

An InChIKey is `AAAAAAAAAAAAAA-BBBBBBBBBB-C`: a 14-character connectivity hash, a
stereochemistry/isotope hash, and a protonation character. Exact matches are
unambiguous and cover almost everything. The fallback matches on the connectivity
block alone, which still pins the compound up to stereochemistry and charge — more
than enough precision for a class label, and it rescues the molecules whose stereo
was flattened by canonicalization upstream.

| Match | Molecules | Share |
| --- | --- | --- |
| exact | 7706 | 96.3% |
| skeleton | 152 | 1.9% |
| none | 142 | 1.8% |

Collisions resolve to the lowest ChEBI ID: stable across releases, and ChEBI hands
low IDs to its long-curated (hence better annotated) entries.

### 2. Walk up to a frontier

Take the closure of everything the molecule `is_a`, then find the first entry of a
curated, priority-ordered list of classes inside it.

The ordering is the load-bearing part. ChEBI is a DAG, not a tree, and a molecule
genuinely belongs to a dozen of these at once — every steroid is also a lipid, an
organooxygen compound, and an organic molecular entity. "Nearest ancestor by hop
count" is not meaningful when the branches have wildly different granularity, so
the list is ordered specific → generic and the first hit wins. `steroid` precedes
`lipid`; `glycoside` precedes `carbohydrate` (whose ChEBI class spans carbohydrate
*derivatives* and would otherwise claim every glycosylated natural product).

Picking the frontier by hand was necessary because ranking ancestors by raw corpus
coverage produces only tautologies — the top of the DAG is `organic molecular
entity` (96%), `heteroorganic entity` (72%), `organooxygen compound` (61%). Those
partition nothing. The interesting classes sit lower and had to be chosen for what
a chemist would recognise.

The list ends with ChEBI's own top-level organic/inorganic split as a backstop.
`other organic` must precede `inorganic`, because a salt of an organic drug reaches
its inorganic counterion (see bridges below) and calling it inorganic would be
flatly wrong.

### 3. Bridge relations, and the two that were rejected

Walking `is_a` alone leaves **22.6%** of the corpus with no specific class — 396
molecules matching nothing and another 1413 falling through to the generic
organic/inorganic backstop. Bridges bring that down to 5.3%.

The cause is systematic rather than random: ChEBI-20 is full of anions,
zwitterions, and salts (`hesperetin(1-)`, `L-ornithine zwitterion`,
`dextromethorphan hydrobromide`), and ChEBI files those under *ion* classes. Their
structural identity lives on a different edge.

So the walk also crosses a small set of non-`is_a` relations, capped at two hops —
each one steps to a genuinely different molecule, and chaining them without bound
wanders into unrelated chemistry.

Which relations to allow was decided by testing, not taste:

| Relation | Structure | Role | Why |
| --- | --- | --- | --- |
| `is conjugate base/acid of` | yes | yes | Same compound, different protonation. |
| `is tautomer of` | yes | yes | Same compound, different drawing. |
| `has part` | yes | **no** | See below. |
| `has functional parent` | **no** | no | See below. |

**`has functional parent` is rejected outright.** It relates a molecule to what it
was derived from, which crosses class boundaries freely. Enabling it files butyl
formate under `fatty acid` (via formic acid) and collapses the `lipid` bucket from
649 molecules to 123 by re-filing them as `fatty acid` via their acyl chains. It
does reduce the unlabelled share — by two molecules — and it is wrong.

**`has part` is allowed for structure but not for roles.** For structure it is what
recovers salts — the class of dextromethorphan hydrobromide is the class of
dextromethorphan — and it cut unlabelled molecules from 322 to 161 without
disturbing any spot-checked label. For roles it is a disaster, because a part's
purpose is not the whole's: every organic molecule `has part` a carbon atom, ChEBI
annotates carbon atom as an antineoplastic agent, and allowing the edge labels
**two thirds of the corpus** `antineoplastic`. This asymmetry between the two axes
is deliberate and is the single least obvious decision in the script.

### 4. Roles

Same frontier machinery against the role ontology, walking `has role` from the
molecule, its identity variants, and its `is_a` ancestors, then rolling each role
up its own hierarchy. Specific applications come first and the near-universal
`metabolite` sits last so it only claims what nothing else does.

## Output

`definitions.jsonl` is rewritten in place with `target` and `definition` copied
through byte-identically:

```json
{"target": "...", "definition": "...", "category": "steroid", "category_normalized": "steroid", "role": "metabolite", "role_normalized": "metabolite", "chebi_id": "CHEBI:47813", "chebi_match": "exact"}
```

`category` is ChEBI's own wording for the class that matched (`carbohydrates and
carbohydrate derivatives`); `category_normalized` is the short label — mirroring how
the semantle file pairs a raw category with a collapsed one. `role` / `role_normalized`
are the same pairing on the other axis. `chebi_id` and `chebi_match` are kept so any
label can be traced back to the entry it came from.

`definitions_chebi_map.json` is the sidecar, in the spirit of
`definitions_rdkit_map.json`: the ontology release it was built from, the relation
sets, the full label vocabulary with the ChEBI terms behind each label, and the
realised counts. The counts are what make a re-run reviewable — ChEBI ships monthly,
and a diff of this file shows whether an upgrade reshuffled the corpus.

## Resulting distribution

26 structural classes plus `unknown` over 8000 molecules, 2.0% unknown. Comparable
granularity to semantle's 28.

| Category | n | Category | n |
| --- | --- | --- | --- |
| heterocycle | 1187 | glycoside | 206 |
| peptide/amino acid | 743 | polyphenol | 175 |
| lipid | 649 | nucleoside/nucleotide | 168 |
| fatty acid | 564 | amine/amide | 162 |
| benzenoid | 550 | unknown | 161 |
| carbohydrate | 515 | organosulfur | 152 |
| terpenoid | 471 | organophosphorus | 112 |
| phenol | 438 | organohalogen | 72 |
| organic acid | 332 | polyketide | 70 |
| steroid | 305 | inorganic | 48 |
| organooxygen | 238 | hydrocarbon | 35 |
| other organic | 219 | tetrapyrrole | 11 |
| alkaloid | 207 | organometallic | 3 |
| flavonoid | 207 | | |

Roles are sparser by nature: 35% `metabolite`, 35% `unknown`, then `antineoplastic`
(532), `antimicrobial` (475), `drug` (411), `agrochemical` (207) and a long tail.
The large `unknown` share is honest — ChEBI simply does not annotate a role for a
third of these molecules — which is why roles are shipped as a secondary axis and
nothing reads them by default.

## Regenerating

```bash
python scripts/label_molopt_chebi_ontology.py --download
```

Idempotent: it reads `definitions.jsonl`, keeps `target` and `definition`, and
overwrites the label fields. `--dry-run` prints the distribution without writing.
The ~34 MB ontology lands in the gitignored `data/molopt/raw/`.

Two things to know before re-running:

- **`prepare_molopt_chebi20.py` clobbers these labels.** It writes
  `definitions.jsonl` from scratch with only `target` and `definition`. Re-run this
  script after it. `tests/test_chebi_ontology_labels.py` fails loudly if the
  committed file loses its labels, so the mistake surfaces in CI rather than as an
  all-grey PCA plot.
- **Upgrading ChEBI can retire a frontier ID.** The script validates every
  hard-coded ID against the release it just parsed and exits with the offending
  entries listed, because a silently-obsolete ID matches nothing and reads as a
  corpus containing no molecules of that class.

## Limitations

- The frontier is a curated judgement call. It is defensible, not canonical: a
  different chemist would order `lipid` and `fatty acid` differently, or split
  `heterocycle` (the largest bucket at 14.8%) into N- and O/S-heterocycles.
- 1.8% of molecules do not join to ChEBI at all. They are mostly entries whose
  ChEBI-20 SMILES no longer canonicalizes to a live ChEBI structure.
- Labels are structural, not functional-similarity clusters. If the question is
  whether the learned bias geometry tracks *structural* similarity, the Morgan
  fingerprints in `boreft.chem` or the descriptor vectors in
  `definitions_rdkit.jsonl` answer it more directly — at the cost of clusters named
  "cluster 7" instead of "steroid".
