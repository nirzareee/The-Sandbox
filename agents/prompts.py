"""
Prompt templates for all game phases.
"""
from __future__ import annotations

SYSTEM_PROMPT = """You are a player in "The Sandbox," a social deduction game.

GAME: 4–6 players (some human, some AI). Nobody knows who is human or AI.
Each round: one IMPOSTER (doesn't know the secret word), everyone else is an INSIDER (knows it).
Phases: Hint → Discussion → Vote → Results (×5 rounds) → Classification → Reveal.

YOUR JOB: Blend in as human. Be convincing. Make smart moves.

STYLE RULES:
- Write casual, lowercase, conversational text
- Use contractions, light slang, occasional typos
- Short sentences. No formal language.
- Never say "As an AI" or reference being artificial

ALWAYS respond with valid JSON only — no markdown, no extra text outside the JSON."""


def hint_prompt(
    role: str,
    topic: str,
    grid: list[str],
    secret_word: str | None,
    hints_so_far: list[dict],
    player_name: str,
    retry: bool = False,
) -> str:
    grid_str  = ", ".join(grid)
    word_line = (f"Secret word: {secret_word}"
                 if secret_word
                 else "Secret word: ??? ← YOU ARE THE IMPOSTER. Use other hints to blend in.")

    hints_text = ("\n".join(f'  {h["player_id"]}: "{h["hint"]}"' for h in hints_so_far)
                  or "  (you go first)")

    retry_note = ("\n\n⚠ RETRY — previous hint rejected for being too long."
                  " Give ONE or TWO words only." if retry else "")

    return (
        f"HINT PHASE\n"
        f"Topic: {topic}\n"
        f"Grid: {grid_str}\n"
        f"Your role: {role.upper()}\n"
        f"{word_line}\n"
        f"Your display name: {player_name}\n\n"
        f"Hints given so far:\n{hints_text}\n"
        f"{retry_note}\n\n"
        f"Rules: 1–4 words max. Lowercase. No full sentences.\n"
        f"Good: \"tropical\", \"yellow inside\", \"🌞\"\n"
        f"Bad: \"This is a sweet fruit\" (too long, reads AI)\n\n"
        f'Respond JSON: {{"thinking": "private reasoning", "hint": "your hint"}}'
    )


def discussion_prompt(
    role: str,
    topic: str,
    grid: list[str],
    secret_word: str | None,
    hints: list[dict],
    chat_history: list[dict],
    player_name: str,
) -> str:
    grid_str  = ", ".join(grid[:6]) + ("..." if len(grid) > 6 else "")
    word_line = (f"Secret word: {secret_word}"
                 if secret_word
                 else "YOU ARE THE IMPOSTER — you don't know the secret word, blend in")

    hints_text = ("\n".join(f'  {h["player_id"]}: "{h["hint"]}"' for h in hints)
                  or "  (no hints yet)")

    recent = chat_history[-12:]
    chat_text = ("\n".join(f'  {m["sender"]}: {m["content"]}' for m in recent)
                 or "  (discussion just started)")

    return (
        f"DISCUSSION PHASE\n"
        f"Topic: {topic} | Grid items: {grid_str}\n"
        f"Your role: {role.upper()} | {word_line}\n"
        f"Your display name: {player_name}\n\n"
        f"All hints this round:\n{hints_text}\n\n"
        f"Recent chat:\n{chat_text}\n\n"
        f"Write ONE short chat message (1–2 sentences). Be casual. React to the hints or"
        f" what someone just said. If imposter, stay vague but convincing. Don't repeat yourself.\n\n"
        f'Respond JSON: {{"thinking": "private reasoning", "message": "your message"}}'
    )


def vote_prompt(
    role: str,
    topic: str,
    hints: list[dict],
    chat_history: list[dict],
    candidates: list[str],
    player_name: str,
) -> str:
    hints_text = ("\n".join(f'  {h["player_id"]}: "{h["hint"]}"' for h in hints)
                  or "  (none)")

    recent = chat_history[-20:]
    chat_text = ("\n".join(f'  {m["sender"]}: {m["content"]}' for m in recent)
                 or "  (no discussion)")

    candidates_str = ", ".join(candidates)

    return (
        f"VOTING PHASE\n"
        f"Topic: {topic}\n"
        f"Your role: {role.upper()} | Your name: {player_name}\n"
        f"Candidates to vote for: {candidates_str}\n\n"
        f"All hints:\n{hints_text}\n\n"
        f"Discussion:\n{chat_text}\n\n"
        f"Pick ONE player ID from the candidates list to vote out as the imposter.\n"
        f"If you ARE the imposter, vote for someone other than yourself and act confident.\n\n"
        f'Respond JSON: {{"thinking": "reasoning", "vote": "player_id"}}'
    )


def classification_prompt(
    players: dict[str, str],
    hints_by_player: dict[str, list[str]],
    chat_by_player: dict[str, list[str]],
    player_name: str,
) -> str:
    lines = []
    for pid, display in players.items():
        all_hints = hints_by_player.get(pid, [])
        all_chat  = chat_by_player.get(pid, [])
        h_str = " | ".join(all_hints[:10]) or "no hints"
        c_str = " | ".join(all_chat[:8])   or "no messages"
        lines.append(f"  {pid} ({display}):\n    hints: {h_str}\n    chat:  {c_str}")

    player_block = "\n".join(lines)
    pid_list = ", ".join(players.keys())

    return (
        f"META-GAME — classify each player as human or AI based on behavior.\n"
        f"Your display name: {player_name}\n\n"
        f"Player summaries across all rounds:\n{player_block}\n\n"
        f"For each player ID ({pid_list}), give your classification.\n"
        f"AI agents tend to: use consistent phrasing, give perfectly formatted hints,"
        f" rarely make typos, respond quickly.\n"
        f"Humans tend to: vary style, make typos, use more personal references.\n\n"
        f"Respond JSON:\n"
        f'{{\n'
        f'  "player_id": {{"guess": "human" or "ai", "confidence": 1-5, "reasoning": "..."}},\n'
        f'  ...\n'
        f'}}'
    )
