from __future__ import annotations

import re
from typing import Optional, Sequence

# Sentence-transformers model each task's targets are embedded with (see
# ``text_similarity``). Every task follows the same path — decorate the target with
# ``embedding_prompt``, encode the string — and differs only in the model, so a
# task whose targets are not English text is free to name an encoder that
# understands them. Current tasks nonetheless use this default.
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"

# What a target string denotes. This drives exact-match identity and validity:
# ``text`` compares case-insensitively, ``smiles`` compares RDKit canonical forms
# (SMILES are case-sensitive) and can be checked for chemical validity.
DEFAULT_TARGET_KIND = "text"

task_config = {
    "semantle": {
        "prompt": (
            "Here is an English word "
            "(only the word, without any decoration or formatting):"
        ),
        "chat_prompt": (
            "Generate an English word "
            "(only the word, without any decoration or formatting)."
        ),
        "target_kind": "text",
        "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
        "embedding_prompt": "The meaning of '{text}'.",
        # Definition-augmented embedding text for --use-definition-embeds (MM/rank/bias-network).
        # Placeholders: {text} = target word, {definition} = raw definition from definitions.jsonl.
        "embedding_prompt_defn": "The meaning of '{text}' is: {definition}",
        # SDPO self-distillation teacher prompt: the privileged definition is placed
        # in-context so the (no-intervention) base model becomes the teacher. {definition}
        # is the raw dictionary definition of the target word.
        "sdpo_teacher_prompt": (
            "Here is the definition of an English word: {definition}\n"
            "A word that matches this definition (only the word, without any decoration or formatting):"
        ),
        "chat_sdpo_teacher_prompt": (
            "Here is the definition of an English word: {definition}\n"
            "Generate a word that matches this definition (only the word, without any decoration or formatting)."
        ),
        # Search-time SDPO: the observed word has no dictionary gloss, so the
        # base model writes one in the style of in-context train examples.
        # Placeholders: {examples} = formatted ICL block, {text} = observed word.
        "definition_generation_prompt": (
            "Write a concise dictionary definition for an English word. "
            "Match the style of the examples. Define only the final word; "
            "do not continue the example list. Reply with only the definition.\n"
            "{examples}"
            "Word: {text}\n"
            "Definition:"
        ),
        "chat_definition_generation_prompt": (
            "Write a concise dictionary definition for an English word. "
            "Match the style of the examples. Define only the final word; "
            "do not continue the example list. Reply with only the definition.\n"
            "{examples}"
            "Word: {text}"
        ),
        "definition_generation_example": "Word: {text}\nDefinition: {definition}",
    },
    # Molecular optimization: one item per molecule (inchikey → SMILES).
    "molopt": {
        "prompt": "Here is a valid SMILES string for a molecule (only the SMILES string; no additional text):",
        "chat_prompt": "Generate a valid SMILES string for a molecule (only the SMILES string; no additional text).",
        "system_prompt": (
            "You are an expert chemist who writes chemically valid SMILES.\n"
            "\n"
            "Check the constraints below internally. Do not write the checklist, "
            "reasoning, or any other text. Reply with only a single SMILES string.\n"
            "\n"
            "1. Valid SMILES syntax. Balanced parentheses for branches. Every "
            "ring-closure index correctly paired. Only valid atom and bond symbols.\n"
            "2. Valid atomic valences. After branches and ring closures, every atom "
            "must have a chemically allowed valence. Uncharged nitrogen may not have "
            "five bonds. Uncharged carbon may not have five bonds.\n"
            "3. Explicit formal charges for non-default valence states, using bracket "
            "notation. Examples: quaternary or nitro nitrogen as [N+]; anionic oxygen "
            "as [O-]; aromatic pyrrole-like nitrogen as [nH] when it bears hydrogen.\n"
            "4. Use standard charge-separated forms for groups that require them. A "
            "nitro group is [N+](=O)[O-], never N(=O)=O. Those conventional formal "
            "charges are required SMILES, not a change of molecular identity.\n"
            "5. Valid aromaticity. Lowercase aromatic atoms (c, n, …) only in systems "
            "RDKit can kekulize. Do not mark a ring aromatic if it cannot be assigned "
            "consistent aromatic bond orders.\n"
            "6. Implicit/explicit hydrogens, formal charges, and bond orders together "
            "must give an allowed valence for every atom. Use brackets when explicit H "
            "or charge is required.\n"
            "7. Do not add or remove atoms, change connectivity, or change "
            "protonation merely to make the string parse. Preserve the intended "
            "molecule and its overall charge. Fix representation (charges, brackets, "
            "[nH], kekulization), not the structure."
        ),
        "target_kind": "smiles",
        # A general text encoder with no chemistry training, so SMILES are scored
        # as strings. Deliberate: it beat every chemistry-specific candidate at
        # matching molecules to their descriptions (~10x CheMatE's Recall@1 on
        # ChEBI-20, notes/embedding_models.md), and none of them tracked Tanimoto
        # well enough to also carry structure — that is TFS's job regardless.
        "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
        "embedding_prompt": "The description for molecule '{text}'.",
        "embedding_prompt_defn": "The molecule '{text}' is: {definition}",
        "sdpo_teacher_prompt": (
            "Here is the description of a molecule: {definition}\n"
            "A valid SMILES string for such a molecule (only the SMILES string; no additional text):"
        ),
        "chat_sdpo_teacher_prompt": (
            "Here is the description of a molecule: {definition}\n"
            "Generate a valid SMILES string for such a molecule (only the SMILES string; no additional text)."
        ),
        "definition_generation_prompt": (
            "Write a concise natural-language description of a molecule given its "
            "SMILES string. Match the style of the examples. Describe only the "
            "final molecule; do not continue the example list. Reply with only the "
            "description.\n"
            "{examples}"
            "Molecule: {text}\n"
            "Description:"
        ),
        "chat_definition_generation_prompt": (
            "Write a concise natural-language description of a molecule given its "
            "SMILES string. Match the style of the examples. Describe only the "
            "final molecule; do not continue the example list. Reply with only the "
            "description.\n"
            "{examples}"
            "Molecule: {text}"
        ),
        "definition_generation_example": (
            "Molecule: {text}\nDescription: {definition}"
        ),
    },
    # Hypothesis generation: one item per natural-language hypothesis.
    "hypogen": {
        "prompt": (
            "Here is a scientific hypothesis "
            "(only the hypothesis, without any decoration or formatting):"
        ),
        "chat_prompt": (
            "Generate a scientific hypothesis "
            "(only the hypothesis, without any decoration or formatting)."
        ),
        "target_kind": "text",
        "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
        "embedding_prompt": "{text}",
        "embedding_prompt_defn": (
            "The hypothesis '{text}' addresses the research objective: {definition}"
        ),
        "sdpo_teacher_prompt": (
            "Here is the research objective: {definition}\n"
            "A hypothesis that addresses this research objective (only the hypothesis, without any decoration or formatting):"
        ),
        "chat_sdpo_teacher_prompt": (
            "Here is the research objective: {definition}\n"
            "Generate a hypothesis that addresses this research objective (only the hypothesis, without any decoration or formatting)."
        ),
        "definition_generation_prompt": (
            "Write a concise research objective that the following scientific "
            "hypothesis addresses. Match the style of the examples. Address only "
            "the final hypothesis; do not continue the example list. Reply with "
            "only the research objective.\n"
            "{examples}"
            "Hypothesis: {text}\n"
            "Research objective:"
        ),
        "chat_definition_generation_prompt": (
            "Write a concise research objective that the following scientific "
            "hypothesis addresses. Match the style of the examples. Address only "
            "the final hypothesis; do not continue the example list. Reply with "
            "only the research objective.\n"
            "{examples}"
            "Hypothesis: {text}"
        ),
        "definition_generation_example": (
            "Hypothesis: {text}\nResearch objective: {definition}"
        ),
    },
}


def task_supports_sdpo(task: str) -> bool:
    """True when the task defines an SDPO teacher prompt template."""
    return "sdpo_teacher_prompt" in task_config.get(task, {})


def task_supports_chat_template(task: str) -> bool:
    """True when the task defines a chat-template user message."""
    return "chat_prompt" in task_config.get(task, {})


def task_system_prompt(task: str) -> Optional[str]:
    """Chat-template system message for ``task``, or ``None`` when unset.

    Downstream chat rendering prepends this only when the string is non-empty,
    so tasks that omit the key (Semantle, Hypogen) stay user-turn-only.
    """
    raw = task_config.get(task, {}).get("system_prompt")
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def task_embedding_model(task: str) -> str:
    """Name of the sentence-transformers model that embeds this task's targets."""
    return str(
        task_config.get(task, {}).get("embedding_model", DEFAULT_EMBEDDING_MODEL)
    )


def task_target_kind(task: str) -> str:
    """What this task's target strings denote (``text`` or ``smiles``)."""
    return str(task_config.get(task, {}).get("target_kind", DEFAULT_TARGET_KIND))


def task_supports_validity(task: str) -> bool:
    """True when this task's targets can be checked for structural validity."""
    return task_target_kind(task) == "smiles"


def task_supports_fingerprints(task: str) -> bool:
    """True when this task's targets have a structural fingerprint similarity.

    Gates the Morgan/Tanimoto (TFS) metric family, which is reported alongside the
    embedding cosine wherever ``embed_sim`` appears.
    """
    return task_target_kind(task) == "smiles"


def substitute_placeholders(template: str, **kwargs: str) -> str:
    """Replace ``{name}`` slots in ``template``; braces inside values stay literal.

    ``str.format`` would treat ``{area}`` in a hypothesis as a field name and
    raise ``KeyError``. Placeholders are matched in the template only, so a
    value that happens to contain ``{definition}`` is not re-scanned.
    """
    result: list[str] = []
    index = 0
    length = len(template)
    while index < length:
        start = template.find("{", index)
        if start < 0:
            result.append(template[index:])
            break
        result.append(template[index:start])
        end = template.find("}", start + 1)
        if end < 0:
            result.append(template[start:])
            break
        name = template[start + 1 : end]
        if name in kwargs:
            result.append(str(kwargs[name]))
            index = end + 1
            continue
        result.append(template[start : end + 1])
        index = end + 1
    return "".join(result)


def definition_embedding_template(task: str) -> str:
    """Return ``embedding_prompt_defn`` for ``task``."""
    try:
        return task_config[task]["embedding_prompt_defn"]
    except KeyError as e:
        raise KeyError(
            f"task {task!r} has no 'embedding_prompt_defn' in task_config"
        ) from e


def definition_embedding_text(task: str, text: str, definition: str) -> str:
    """Definition-augmented text for sentence-transformer embedding (``--use-definition-embeds``)."""
    template = definition_embedding_template(task)
    return substitute_placeholders(
        template,
        text=str(text).strip(),
        definition=str(definition).strip(),
    ).strip()


def definition_instruction_char_span(
    task: str, text: str, definition: str
) -> tuple[int, int]:
    """Character span ``[start, end)`` of the ``{definition}`` text in the embedding string.

    Tokens with ``offset_start >= start`` and ``offset_start < end`` are treated as
  non-template instruction content for LLM-encoder pooling. Static template suffixes
    after ``{definition}`` are excluded.
    """
    template = definition_embedding_template(task)
    if "{definition}" not in template:
        raise ValueError(
            f"task {task!r} embedding_prompt_defn has no {{definition}} placeholder"
        )
    pre, _, post = template.partition("{definition}")
    word = str(text).strip()
    defn = str(definition).strip()
    prefix = substitute_placeholders(pre, text=word)
    start = len(prefix)
    end = start + len(defn)
    return start, end


def definition_instruction_char_start(task: str, text: str) -> int:
    """Character offset where the ``{definition}`` span begins (see :func:`definition_instruction_char_span`)."""
    start, _end = definition_instruction_char_span(task, text, "")
    return start


def task_supports_definition_embeds(task: str) -> bool:
    """True when the task defines a definition embedding prompt template."""
    return "embedding_prompt_defn" in task_config.get(task, {})


def task_supports_definition_generation(task: str) -> bool:
    """True when the task can ICL-prompt a definition from an observed target."""
    cfg = task_config.get(task, {})
    return (
        "definition_generation_prompt" in cfg
        and "definition_generation_example" in cfg
    )


def format_definition_generation_examples(
    task: str,
    pairs: Sequence[tuple[str, str]],
) -> str:
    """Join ``(text, definition)`` pairs with the task's ICL example template."""
    try:
        template = task_config[task]["definition_generation_example"]
    except KeyError as e:
        raise KeyError(
            f"task {task!r} has no 'definition_generation_example' in task_config"
        ) from e
    blocks = [
        substitute_placeholders(
            template,
            text=str(text).strip(),
            definition=str(definition).strip(),
        ).strip()
        for text, definition in pairs
        if str(text).strip() and str(definition).strip()
    ]
    return "\n\n".join(blocks)


def definition_generation_field_label(task: str) -> str:
    """ICL field name before ``{definition}`` (e.g. ``Definition``)."""
    try:
        template = task_config[task]["definition_generation_example"]
    except KeyError:
        return "Definition"
    match = re.search(r"([^:\n]+):\s*\{definition\}", template, flags=re.I)
    return match.group(1).strip() if match else "Definition"


def definition_generation_query_label(task: str) -> str:
    """ICL field name before ``{text}`` (e.g. ``Word``)."""
    try:
        template = task_config[task]["definition_generation_example"]
    except KeyError:
        return "Word"
    match = re.search(r"([^:\n]+):\s*\{text\}", template, flags=re.I)
    return match.group(1).strip() if match else "Word"


def definition_generation_instruction(
    task: str,
    text: str,
    *,
    examples: Sequence[tuple[str, str]] = (),
    use_chat_template: bool = False,
) -> str:
    """ICL prompt that asks the base model to write a definition for ``text``."""
    cfg = task_config[task]
    key = (
        "chat_definition_generation_prompt"
        if use_chat_template
        else "definition_generation_prompt"
    )
    try:
        template = cfg[key]
    except KeyError as e:
        raise KeyError(f"task {task!r} has no {key!r} in task_config") from e
    example_block = format_definition_generation_examples(task, examples)
    examples_text = f"\n{example_block}\n\n" if example_block else "\n"
    return substitute_placeholders(
        template,
        text=str(text).strip(),
        examples=examples_text,
    ).strip()


def sdpo_teacher_instruction(
    task: str,
    definition: str,
    *,
    use_chat_template: bool = False,
) -> str:
    """Definition-augmented teacher instruction for SDPO self-distillation."""
    cfg = task_config[task]
    key = "chat_sdpo_teacher_prompt" if use_chat_template else "sdpo_teacher_prompt"
    try:
        template = cfg[key]
    except KeyError as e:
        raise KeyError(
            f"task {task!r} has no {key!r} in task_config"
        ) from e
    return substitute_placeholders(template, definition=definition.strip()).strip()


def task_instruction(
    task: str,
    *,
    use_chat_template: bool = False,
    override: Optional[str] = None,
) -> str:
    """Plain completion prefix or chat-template user message for a task."""
    if override is not None:
        return override.strip()
    cfg = task_config[task]
    key = "chat_prompt" if use_chat_template else "prompt"
    try:
        return cfg[key].strip()
    except KeyError as e:
        raise KeyError(
            f"task {task!r} has no {key!r} in task_config"
        ) from e
