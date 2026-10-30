"""Recap formula ported from recap-en-to-mm (context memory -> batch translate -> audit).

Used by GeminiRewriter for Recap mode only. Timing/ids always come from the source
segments; Gemini never owns timestamps.
"""
import difflib
import json
import re
from typing import Any, Callable, Dict, List, Optional

LogFn = Callable[[str, float], None]

BATCH_MAX_ITEMS = 40
BATCH_MAX_CHARS = 24000
CONTEXT_ROWS = 6


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _fmt_ts(sec: float) -> str:
    sec = max(0.0, float(sec))
    return f"{int(sec // 60):02d}:{sec % 60:05.2f}"


def _comparison_text(value: Any) -> str:
    return re.sub(r"[^\w\u1000-\u109f]+", "", _clean(value).lower(), flags=re.UNICODE)


def _has_speakable(value: Any) -> bool:
    return bool(re.search(r"[\w\u1000-\u109f]", _clean(value), flags=re.UNICODE))


def build_units(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {"id": s["id"], "timestamp": f"{_fmt_ts(s['start'])}-{_fmt_ts(s['end'])}",
         "source_text": _clean(s.get("text"))}
        for s in segments
    ]


def _batches(units: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    out, cur, chars = [], [], 0
    for u in units:
        cost = len(u["source_text"]) + 96
        if cur and (len(cur) >= BATCH_MAX_ITEMS or chars + cost > BATCH_MAX_CHARS):
            out.append(cur)
            cur, chars = [], 0
        cur.append(u)
        chars += cost
    if cur:
        out.append(cur)
    return out


def _source_lines(items: List[Dict[str, Any]]) -> str:
    return "\n".join(f"[ID {int(x['id']):05d}] [{x['timestamp']}] {x['source_text']}" for x in items)


def _translation_lines(items: List[Dict[str, Any]]) -> str:
    return "\n".join(
        f"[ID {int(x['id']):05d}] [{x['timestamp']}] SOURCE: {x['source_text']}\n"
        f"OUTPUT: {_clean(x.get('translation'))}" for x in items
    )


def _duplicate_pairs(items, previous=None) -> List[tuple]:
    seq = list(previous[-1:] if previous else []) + list(items)
    bad = []
    for a, b in zip(seq, seq[1:]):
        ta, tb = _comparison_text(a.get("translation")), _comparison_text(b.get("translation"))
        if min(len(ta), len(tb)) < 12:
            continue
        if difflib.SequenceMatcher(None, ta, tb).ratio() < 0.94:
            continue
        sa, sb = _comparison_text(a.get("source_text")), _comparison_text(b.get("source_text"))
        if difflib.SequenceMatcher(None, sa, sb).ratio() < 0.86:
            bad.append((int(a["id"]), int(b["id"])))
    return bad


def _check_rows(items, previous=None) -> None:
    pairs = _duplicate_pairs(items, previous)
    if pairs:
        raise ValueError("Suspicious adjacent duplicate Burmese translation IDs: " +
                         ", ".join(f"{a}/{b}" for a, b in pairs))
    for it in items:
        if _has_speakable(it["source_text"]) and not _has_speakable(it.get("translation")):
            raise ValueError(f"ID {it['id']} has no speakable words")


def build_story_context(rw, units, model_order, log: LogFn) -> Dict[str, Any]:
    memory: Dict[str, Any] = {k: [] for k in
        ("characters", "relationships", "places", "terms", "plot_state", "name_consistency", "unresolved")}
    chunks = _batches(units)
    for i, chunk in enumerate(chunks, 1):
        prompt = f"""
You are building CONTEXT MEMORY for a later translation task.
Do not translate the transcript now. Do not write narration. Do not summarize away named facts.
Read this next chronological source chunk and UPDATE the existing compact context memory.
The memory exists only to keep names, relationships, terminology, locations, identities,
and the current story state consistent when later translation requests are sent in separate API calls.

RULES:
- Preserve exact character/place/object names as they appear in the source when known.
- Keep important relationships, aliases, roles, reveals, and terminology.
- Keep plot_state chronological but compact. Do not invent anything.
- If an earlier assumption is corrected by this chunk, update it.
- Return JSON only with these keys:
  characters, relationships, places, terms, plot_state, name_consistency, unresolved
- Each key must contain a compact JSON array. Keep the total memory concise enough to reuse in every later request.

EXISTING MEMORY:
{json.dumps(memory, ensure_ascii=False)}

SOURCE CHUNK {i}/{len(chunks)}:
{_source_lines(chunk)}
"""
        data = rw._generate_json(prompt, model_order)
        if not isinstance(data, dict):
            raise RuntimeError("Story context response must be a JSON object")
        memory = {k: list(data.get(k) or []) for k in memory}
        log(f"Story Context {i}/{len(chunks)} updated", 10.0 + 20.0 * i / len(chunks))
    return memory


def translate_batch(rw, model_order, *, lang_name, story_context, previous, current, nxt, log, depth=0):
    rules = f"""
You are a CONTEXT-AWARE {lang_name.upper()} NARRATION TRANSLATOR.
Translate ONLY the CURRENT source items into smooth, natural spoken {lang_name} that sounds as if it was originally written by a fluent {lang_name} movie/story narrator.

CORE PRINCIPLE:
Preserve the SOURCE MEANING and VISUAL EVENTS 100%, NOT the original wording or sentence structure.

NATIVE NARRATOR STYLE:
- Use natural spoken narrator phrasing, rhythm, and connectors.
- Avoid stiff word-for-word translation; restructure wording and clause order INSIDE THE SAME ID when needed.
- Keep pronouns, kinship terms, honorifics, gendered self-reference, names, and relationships contextually correct.
- Keep dialogue intent and social register natural without turning ordinary speech into formal literary language.

HARD RULES:
1. Do NOT remove, condense away, or omit meaningful source information. Every action, event, object, person, relationship, cause/effect, dialogue, and important visual detail must remain in the same ID.
2. Do NOT translate word-for-word when it sounds stiff or machine-like. You MAY restructure wording and clause order INSIDE THE SAME ID.
3. Small expansion is allowed only for clarity/natural flow; never exaggerate or add unrelated filler.
4. Do not invent plot events, opinions, hooks, intros, outros, or facts.
5. Keep every CURRENT ID exactly once, in order. Do not merge or split IDs.
6. Context is READ-ONLY. Never copy events from previous/next context into a current ID.
7. Maintain consistent names, relationships, places, objects, and terminology.
8. Timestamps are local metadata. Return no timestamps.
9. Return JSON only with top-level key "items". Each item must contain integer "id" and non-empty string "translation".
10. NEVER return placeholders such as "...", dashes, empty text, or schema/example text.
"""
    ids = [int(x["id"]) for x in current]
    prompt = f"""
{rules}

STORY CONTEXT MEMORY:
{json.dumps(story_context, ensure_ascii=False)}

PREVIOUS CONTEXT (READ-ONLY):
{_translation_lines(previous) if previous else '(none)'}

CURRENT ITEMS — CONVERT ALL OF THESE, ONE-FOR-ONE:
{_source_lines(current)}

NEXT CONTEXT (READ-ONLY LOOK-AHEAD):
{_source_lines(nxt) if nxt else '(none)'}

CRITICAL OUTPUT ID PROTOCOL:
- CURRENT IDs are immutable SOURCE IDs, not row positions.
- Return exactly {len(current)} items.
- Copy this exact ID sequence without dropping, renumbering, or inventing any ID:
{json.dumps(ids)}
"""
    last: Optional[Exception] = None
    for attempt in range(1, 4):
        try:
            payload = rw._extract_json(rw._generate(prompt, model_order))
            items = payload.get("items") if isinstance(payload, dict) else None
            if not isinstance(items, list):
                raise ValueError("response must contain an items array")
            got = [int(i.get("id")) for i in items if isinstance(i, dict)]
            if got != ids:
                raise ValueError(f"ID mismatch. expected={ids[:3]}...{ids[-3:]} got={got[:3]}...{got[-3:]}")
            by_id = {int(i["id"]): _clean(i.get("translation")) for i in items}
            out = [{**src, "translation": by_id[int(src["id"])]} for src in current]
            _check_rows(out, previous)
            return out
        except Exception as exc:
            last = exc
            log(f"Translation validation attempt {attempt}/3 failed: {exc}", -1)
            prompt += (f"\n\nCORRECTION REQUIRED: Previous response was invalid because: {exc}. "
                       f"Return the full CURRENT batch again with exactly {len(ids)} items. "
                       f"The exact items[].id sequence MUST be {json.dumps(ids)}.")
    if len(current) > 6:
        mid = len(current) // 2
        left, right = current[:mid], current[mid:]
        log(f"Splitting IDs {ids[0]}-{ids[-1]} into smaller translation windows", -1)
        tl = translate_batch(rw, model_order, lang_name=lang_name, story_context=story_context,
                             previous=previous[-CONTEXT_ROWS:], current=left,
                             nxt=(right + list(nxt))[:CONTEXT_ROWS], log=log, depth=depth + 1)
        tr = translate_batch(rw, model_order, lang_name=lang_name, story_context=story_context,
                             previous=(previous + tl)[-CONTEXT_ROWS:], current=right,
                             nxt=list(nxt), log=log, depth=depth + 1)
        return tl + tr
    raise RuntimeError(f"Translation batch could not be validated: {last}")


def audit_batch(rw, model_order, translated, *, story_context, previous, nxt, log, depth=0):
    ids = [int(x["id"]) for x in translated]
    ids_json = json.dumps(ids)
    sample = [{"id": ids[0], "status": "pass"}]
    if len(ids) > 1:
        sample.append({"id": ids[1], "status": "repair",
                       "translation": f"complete corrected output for ID {ids[1]} only"})
    base = f"""
You are a STRICT SENTENCE-LOCKED TRANSLATION AUDITOR.
Audit the Burmese/Myanmar narrator output. Every action/fact/object/dialogue/detail must remain inside the same ID as its SOURCE.
PREVIOUS and NEXT context are READ-ONLY. This is never a summarization task.

CRITICAL ID PROTOCOL:
- CURRENT IDs are SOURCE IDs, not positions and not local row numbers.
- NEVER renumber, restart, shorten, substitute, or invent IDs.
- Return exactly {len(ids)} checks.
- The exact required CURRENT ID sequence is: {ids_json}

For EVERY CURRENT ID check:
- semantic completeness: no meaningful source information omitted or condensed away;
- no event borrowed from previous/next IDs;
- no shifted chain where a row translates another row;
- no adjacent duplicate/near-duplicate outputs for different sources;
- natural spoken narrator style, stable names/pronouns, and no invented information.

Return JSON ONLY with every CURRENT ID once and in the exact sequence above.
Example shape using ACTUAL IDs from this batch:
{json.dumps({"checks": sample}, ensure_ascii=False)}
Use status=pass to preserve a good translation byte-for-byte. Use repair only when necessary.

STORY CONTEXT:
{json.dumps(story_context, ensure_ascii=False)}

PREVIOUS APPROVED CONTEXT (READ-ONLY):
{_translation_lines(previous) if previous else '(none)'}

CURRENT ITEMS TO AUDIT:
{_translation_lines(translated)}

NEXT SOURCE CONTEXT (READ-ONLY):
{_source_lines(nxt) if nxt else '(none)'}

FINAL ID CHECK BEFORE YOU RESPOND:
Your checks[].id list MUST equal exactly: {ids_json}
"""
    prompt, last = base, None
    for attempt in range(1, 3):
        try:
            payload = rw._extract_json(rw._generate(prompt, model_order))
            checks = payload.get("checks") if isinstance(payload, dict) else None
            if not isinstance(checks, list):
                raise ValueError("audit response must contain a checks array")
            by_id = {int(x["id"]): dict(x) for x in translated}
            got, repaired = [], 0
            for c in checks:
                uid = int(c.get("id"))
                got.append(uid)
                status = _clean(c.get("status")).lower()
                if status == "pass":
                    continue
                if status != "repair":
                    raise ValueError(f"invalid audit status for ID {uid}: {status}")
                repl = _clean(c.get("translation"))
                if uid not in by_id or not repl:
                    raise ValueError(f"missing repair translation for ID {uid}")
                by_id[uid]["translation"] = repl
                repaired += 1
            if got != ids:
                raise ValueError(f"audit ID mismatch expected={ids} got={got}")
            audited = [by_id[i] for i in ids]
            _check_rows(audited, previous)
            log(f"Sentence-lock audit repaired {repaired} item(s)" if repaired
                else "Sentence-lock audit: every ID passed", -1)
            return audited
        except Exception as exc:
            last = exc
            log(f"Audit attempt {attempt}/2 failed: {exc}", -1)
            prompt = base + (f"\n\nCORRECTION REQUIRED: {exc}. Return the full checks array for every "
                             f"CURRENT ID. The exact ID list is {ids_json}; do not renumber it.")
    if len(translated) > 6:
        mid = len(translated) // 2
        left, right = translated[:mid], translated[mid:]
        al = audit_batch(rw, model_order, left, story_context=story_context,
                         previous=previous[-CONTEXT_ROWS:], nxt=(right + list(nxt))[:CONTEXT_ROWS],
                         log=log, depth=depth + 1)
        ar = audit_batch(rw, model_order, right, story_context=story_context,
                         previous=(previous + al)[-CONTEXT_ROWS:], nxt=list(nxt),
                         log=log, depth=depth + 1)
        return al + ar
    raise RuntimeError(f"Translation audit could not be validated: {last}")


def run_recap_formula(rw, segments, model_order, lang_name, progress=None):
    """Return (context, [{id,start,end,text}]) for Recap mode."""
    def log(msg: str, pct: float) -> None:
        if progress and pct >= 0:
            progress(msg, pct)

    units = build_units(segments)
    context = build_story_context(rw, units, model_order, log)
    batches = _batches(units)
    done: List[Dict[str, Any]] = []
    for i, batch in enumerate(batches, 1):
        start = sum(len(b) for b in batches[:i - 1])
        prev = done[-CONTEXT_ROWS:]
        nxt = units[start + len(batch): start + len(batch) + CONTEXT_ROWS]
        tr = translate_batch(rw, model_order, lang_name=lang_name, story_context=context,
                             previous=prev, current=batch, nxt=nxt, log=log)
        try:
            tr = audit_batch(rw, model_order, tr, story_context=context, previous=prev, nxt=nxt, log=log)
        except Exception as exc:
            log(f"Audit skipped for batch {i} ({exc}); keeping validated translation", -1)
        done.extend(tr)
        log(f"Recap translate+audit {i}/{len(batches)}", 35.0 + 50.0 * i / len(batches))
    by_id = {x["id"]: x["translation"] for x in done}
    return context, [{"id": s["id"], "start": float(s["start"]), "end": float(s["end"]),
                      "text": by_id[s["id"]]} for s in segments]
