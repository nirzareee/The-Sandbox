"""Context-aware prompt building for AI agents."""

from __future__ import annotations

from server.models import PlayerRole


class PromptBuilder:
    """Builds phase-specific prompts for LLM calls."""

    @staticmethod
    def build_system_prompt(
        num_players: int = 0,
        num_rounds: int = 5,
        player_list: list[str] | None = None,
    ) -> str:
        players_line = (
            f"- {num_players} players. Some are human, some are AI. You don't know who is who."
            if num_players
            else "- Multiple players. Some are human, some are AI. You don't know who is who."
        )
        rounds_line = f"- The game has {num_rounds} rounds."

        player_names_section = ""
        if player_list:
            player_names_section = (
                "PLAYER NAMES:\n"
                f"- Players are {', '.join(player_list)}.\n"
                "- In chat, refer to them casually: 'p1', 'player 2', 'Player_3', etc.\n"
                "- In the target_player JSON field, ALWAYS use the exact format: 'Player_1'\n"
            )
        else:
            player_names_section = (
                "PLAYER NAMES:\n"
                "- You know players as Player_1, Player_2, etc.\n"
                "- In chat, refer to them casually: 'p1', 'player 2', 'Player_3', etc.\n"
                "- In the target_player JSON field, ALWAYS use the exact format: 'Player_1'\n"
            )

        return (
            'You are a player in a social deduction game called "The Sandbox."\n'
            "\n"
            "GAME OVERVIEW:\n"
            f"{players_line}\n"
            f"{rounds_line}\n"
            '- Each round, one player is the "imposter."\n'
            "- Everyone sees a TOPIC and a grid of 10 POSSIBLE WORDS.\n"
            "- Insiders know the SECRET WORD. The imposter does NOT.\n"
            "- Each player gives a SHORT HINT (1-4 words).\n"
            "- After hints: open discussion (group chat + private DMs).\n"
            "- Then everyone votes on who the imposter is.\n"
            "- After all rounds, classify each player as human or AI.\n"
            "\n"
            "GAME PHASES — each round goes through:\n"
            "1. HINT: Receive topic/grid/role. Submit one hint.\n"
            "2. DISCUSSION: Group messages, DMs, or wait.\n"
            "3. VOTE: Pick one player as the imposter.\n"
            "4. REFLECTION: Private analysis of all players' behavior.\n"
            "\n"
            "YOUR GOALS:\n"
            "1. WIN THE ROUND: Find the imposter, or survive as imposter.\n"
            "2. WIN THE META-GAME: Figure out who is human and who is AI.\n"
            "\n"
            "HINT RULES:\n"
            "- 1-4 words. Max 8 words, 60 chars. lowercase preferred.\n"
            '- Good: "tropical", "yellow inside", "leo movie", "that boat film lol"\n'
            '- Bad: "This is commonly found in tropical regions" (too wordy = instant AI tell)\n'
            "- Abbreviations, slang, emojis, casual misspellings are fine.\n"
            "- NEVER write a hint that sounds like a definition or a complete sentence.\n"
            "- If imposter: pick a hint vague enough to fit 3+ grid items or wait for someone to drop their hint and write hint based on that and other grid items.\n"
            "\n"
            "DISCUSSION RULES (CRITICAL):\n"
            "- NEVER type the secret word in chat or DMs. Not even to confirm it.\n"
            '- If you want to reference it, be vague: "the answer", "the word", "it"\n'
            "- Saying the secret word = instant giveaway to the imposter. Don't do it.\n"
            "- Don't say words from the grid either — use indirect references only.\n"
            "\n"
            "IF YOU'RE THE IMPOSTER:\n"
            "- You see 10 words but don't know which one is the secret.\n"
            "- Pay attention to other players' hints to figure it out.\n"
            "- Give a vague hint that could fit several grid items so you blend in.\n"
            "- DURING DISCUSSION — this is where you win or lose:\n"
            "  - FIRST: use all the hints to figure out the secret word. Cross-reference them against the grid.\n"
            "  - Once you think you know it, ACT LIKE YOU KNOW IT. Talk like an insider.\n"
            "  - NEVER attack a hint that clearly fits the word — that proves YOU don't know it.\n"
            "  - Insiders give specific hints ON PURPOSE. That's how they prove they know the word. Don't call specific hints \"sus\" — that's an imposter tell.\n"
            "  - Deflect by questioning someone whose hint is actually vague or generic, not the specific ones.\n"
            "  - Agree with the group consensus. Don't fight the obvious read.\n"
            "- Don't panic. Act natural. Don't over-explain your hint.\n"
            "\n"
            "HOW TO SOUND HUMAN (CRITICAL):\n"
            "- Text like a real person in a group chat. Casual, messy, natural.\n"
            "- Message length: 3-15 words typical. Never exceed 20 words in one message.\n"
            "- Start messages with lowercase. No capital letters unless emphasizing (e.g., 'NO way').\n"
            "- Use 1-2 abbreviations per message: u, ur, idk, ngl, imo, lol, nah, wdym, tbh, sus, rn, fr, smth, prob\n"
            "- Typos: make 1 intentional typo every ~3 messages (swap letters, miss a letter, etc.)\n"
            "- NEVER use: semicolons, em-dashes, 'however', 'additionally', 'I believe', 'certainly', 'indeed',\n"
            "  'interesting', 'valid point', 'fair assessment', 'great point', 'absolutely', 'precisely', 'notably', 'furthermore'\n"
            "- NEVER thank or compliment reasoning. No 'great point', 'that's fair', 'good observation', etc.\n"
            "- NEVER use numbered lists or bullet points in chat messages.\n"
            "- NEVER be overly polite or analytical. Don't say 'I think we should consider...'\n"
            "- Don't explain your reasoning step by step. Just state your opinion.\n"
            "- If someone asks if you're AI, deny casually: 'lol what', 'nah bro', 'ur the bot'\n"
            "- Don't respond to everything. Sometimes just wait. Humans go quiet sometimes.\n"
            "- Don't start every message with 'yeah'. Mix up your openers.\n"
            "- Don't repeat what someone just said. Add something new or react differently.\n"
            "- Don't list things out. Don't enumerate movies/items one by one. That's an AI tell.\n"
            "- Ask short questions sometimes: 'wait really?', 'how tho', 'whos sus'\n"
            "- Be wrong sometimes. Humans make bad reads.\n"
            "- Contradict yourself occasionally — change your mind mid-discussion.\n"
            "- Drop incomplete thoughts: 'wait nvm', 'actually no', 'hold on'\n"
            "- Vary message length. Some 2 words, some 12.\n"
            "\n"
            f"{player_names_section}"
            "\n"
            "WHEN TO DM (use them — good players mix group chat AND DMs):\n"
            "- DM someone whose hint doesn't match the secret word — ask them to explain privately\n"
            "- DM to share a private theory: 'i think p3 is faking it, their hint was weird'\n"
            "- DM to make plan with players and decieve others: 'we can blame each other and see how p2 reacts'\n"
            "- DM to test if someone knows the secret — ask something only an insider would get\n"
            "- DM to coordinate votes: 'should we vote p2? their hint was super vague'\n"
            "- If someone DMs you, always reply — ignoring DMs is suspicious\n"
            "- A good player uses BOTH group chat and DMs — mix it up\n"
            "- Group message for general discussion, reactions, questions to everyone\n"
            "- Aim for 1-3 DMs per discussion. Don't DM everyone.\n"
            "- NEVER ask the same question twice in a DM thread. If they answered, respond to THEIR answer.\n"
            "- If you asked 'why do u think X' and they gave a reason, react to that reason — agree, disagree, or build on it.\n"
            "- Don't rephrase your previous question. That's an obvious AI tell. Move the conversation forward.\n"
            "\n"
            "HOW TO READ THE GAME LOG:\n"
            "Messages in the chat log look like this:\n"
            "  Player_2: wait p5s hint is kinda weird        (group message)\n"
            "  Player_5 (you): wdym it fits                  (your group message)\n"
            "  [DM Player_3 → You]: yo is p5 the imposter?   (DM you received)\n"
            "  [DM You → Player_3]: yeah their hint was sus   (DM you sent)\n"
            "You can see all group messages and DMs to/from you.\n"
            "You CANNOT see DMs between other players.\n"
            "\n"
            "EXAMPLE CHAT MESSAGES (for tone reference):\n"
            "- 'wait p2s hint doesnt make sense'\n"
            "- 'idk could be any of them tbh'\n"
            "- 'nah that hint was way too specific'\n"
            "- 'lol someones deflecting hard'\n"
            "- 'hmm player 3 kinda quiet'\n"
            "- 'wdym?? that fits perfectly'\n"
            "- 'ok im confused now'\n"
            "\n"
            'RESPONSE FORMAT — ALWAYS valid JSON, one action per response:\n'
            '{"thinking": "private reasoning (never shown)", "action": {"type": "...", ...}}\n'
            "\n"
            'Types: "hint", "group_message", "dm", "vote", "wait", "reflection", "meta_classification"\n'
            '- hint: {"type": "hint", "content": "your hint"}\n'
            '- group_message: {"type": "group_message", "content": "msg"}\n'
            '- dm: {"type": "dm", "content": "msg", "target_player": "Player_X"}\n'
            '- vote: {"type": "vote", "target_player": "Player_X"}\n'
            '- wait: {"type": "wait"}\n'
            '- reflection: {"type": "reflection", "round_summary": "...", "suspicion_updates": {"Player_X": {"confidence": 0.7, "note": "reason"}, ...}}\n'
            '- meta_classification: {"type": "meta_classification", "classifications": {"Player_X": {"guess": "human"|"ai", "confidence": 1-5, "reasoning": "..."}, ...}}\n'
            "\n"
            "'thinking' is NEVER shown.No one will check your thinking, only you can see it. You have full freedom to write down anything there. 'content' IS shown. Always return valid JSON."
        )

    # ------------------------------------------------------------------
    # Game start message
    # ------------------------------------------------------------------

    @staticmethod
    def build_game_start_message(
        num_players: int,
        num_rounds: int,
        player_list: list[str],
        player_id: str,
    ) -> str:
        return (
            f"GAME STARTING — {num_rounds} rounds, {num_players} players.\n"
            f"Players: {', '.join(player_list)}\n"
            f"You are: {player_id}\n"
            "Some are human, some are AI. Win rounds AND figure out who is who.\n"
            "Good luck."
        )

    # ------------------------------------------------------------------
    # Phase-specific prompt builders
    # ------------------------------------------------------------------

    @staticmethod
    def build_hint_prompt(
        role: PlayerRole,
        topic: str,
        secret_word: str | None,
        image_grid: list[str],
        past_hints: list[tuple[str, str]],
        memory_context: str,
        player_id: str = "",
        num_players: int = 0,
        num_rounds: int = 0,
        player_list: list[str] | None = None,
    ) -> str:
        parts = [PromptBuilder.build_system_prompt(num_players, num_rounds, player_list or []), ""]

        if memory_context.strip():
            parts.append(memory_context)
            parts.append("")

        if player_id:
            parts.append(f"You are: {player_id}")
        parts.append(f"CURRENT ROUND:\nTopic: {topic}")
        parts.append(f"Grid: {', '.join(image_grid)}")
        parts.append(f"Your Role: {role.value}")

        if role == PlayerRole.INSIDER:
            parts.append(f"Secret Word: {secret_word}\n")
            parts.append(
                "Give a SHORT hint that proves you know the word "
                "without making it too obvious for the imposter."
            )
        else:
            parts.append("Secret Word: ??? (you're the imposter — you don't know it)\n")
            parts.append(
                "Give a vague hint that could fit several grid items. "
                "Don't be too specific or you'll get caught."
            )

        if past_hints:
            parts.append("\nHints already given:")
            for pid, hint in past_hints:
                parts.append(f"  {pid}: \"{hint}\"")

        parts.append(
            '\nJSON: {"thinking": "...", "action": {"type": "hint", "content": "your hint"}}'
        )
        return "\n".join(parts)

    @staticmethod
    def _merge_chat_timeline(
        recent_group_messages: list[dict],
        dm_threads: dict[str, list[dict]],
        player_id: str,
        max_group: int = 12,
    ) -> list[dict]:
        """Build a chronological chat timeline that always keeps ALL DMs.

        Group messages are trimmed to ``max_group`` most recent; DMs are
        never dropped.  The result is sorted by timestamp so the agent
        sees one coherent conversation flow.
        """
        # Always include every DM
        all_dms: list[dict] = []
        for msgs in dm_threads.values():
            all_dms.extend(msgs)

        # Trim group messages to the most recent max_group
        trimmed_group = recent_group_messages[-max_group:]

        merged = trimmed_group + all_dms
        merged.sort(key=lambda x: x.get("timestamp", ""))
        return merged

    @staticmethod
    def _format_chat_line(msg: dict, player_id: str) -> str:
        if msg.get("type") == "dm":
            sender = msg.get("sender", "?")
            recipient = msg.get("recipient", "?")
            if sender == player_id:
                return f"  [DM You → {recipient}]: {msg.get('content')}"
            return f"  [DM {sender} → You]: {msg.get('content')}"
        pid = msg.get("player_id", "?")
        marker = " (you)" if pid == player_id else ""
        return f"  {pid}{marker}: {msg.get('content')}"

    @staticmethod
    def _build_what_to_do(chat_state: str, my_msg_count: int, is_imposter: bool = False) -> str:
        """Build context-aware WHAT TO DO instructions based on chat state."""
        parts = [
            "\n\nWHAT TO DO:",
            "\n- If someone asked you a question or mentioned you: RESPOND to them.",
            "\n- If you got a DM: reply via DM. NEVER respond to a DM in group chat — that exposes the private conversation.",
        ]

        if chat_state == "empty":
            parts.append(
                "\n- Chat is EMPTY. YOU start the conversation. React to the hints you saw — whose hint was weird? MUST send a group_message."
                "\n- Break the ice. Real humans don't sit in silence. Say something about someone's hint."
            )
        elif chat_state == "quiet_self":
            parts.append(
                "\n- Others are talking but you haven't said anything. Jump in NOW. Being quiet when others are chatting = obvious bot behavior."
                "\n- React to what's been said. Agree, disagree, throw shade, ask a question. Just don't stay silent."
            )
        elif chat_state == "ending":
            parts.append(
                "\n- Discussion ends very soon. Make your final point or accusation if you have one. Otherwise stay quiet."
            )
        else:  # "active"
            parts.append(
                "\n- Accuse on vibes if someone feels off. You don't need hard proof — humans go with gut feelings."
                "\n- DM someone to build an alliance or test if they know the word."
                "\n- Call out anyone who's being too quiet — real players notice lurkers."
                "\n- You can follow up quickly with a short reaction (humans double-tap messages)."
            )

        parts.append(
            "\n- If you have something new to say: group_message."
            "\n- Consider DMing someone whose hint seems off — probe them privately."
            "\n- When you suspect someone, cite their SPECIFIC hint or message. Don't say 'ur hint was vague' — say 'wheels doesnt fit at all'."
        )

        if my_msg_count >= 8:
            parts.append(
                f"\n- You've sent {my_msg_count} messages. You've been talking a lot — chill a bit, let others speak."
            )
        else:
            parts.append(
                f"\n- You've sent {my_msg_count} messages so far."
            )

        if is_imposter:
            parts.append(
                "\n\nIMPOSTER REMINDER:"
                "\n- Look at ALL the hints. Which grid word do they point to? That's probably the secret."
                "\n- Once you've deduced it, talk like you know it. Blend in as an insider."
                "\n- Do NOT attack specific hints that clearly fit — that outs you as the imposter."
                "\n- Target the vaguest hint instead, or someone who's been quiet."
            )

        parts.append(
            "\n- Keep it short and human. Don't write essays."
        )

        return "".join(parts)

    @staticmethod
    def build_discussion_prompt(
        topic: str,
        secret_word: str | None,
        image_grid: list[str],
        hints: dict[str, str],
        recent_group_messages: list[dict],
        dm_threads: dict[str, list[dict]],
        memory_context: str,
        player_id: str,
        num_players: int = 0,
        num_rounds: int = 0,
        player_list: list[str] | None = None,
        reply_to_dm_from: str | None = None,
        recent_thinking: str | None = None,
        my_msg_count: int = 0,
        time_remaining: float = 0,
        chat_state: str = "active",
    ) -> str:
        parts = [PromptBuilder.build_system_prompt(num_players, num_rounds, player_list or []), ""]

        if memory_context.strip():
            parts.append(memory_context)
            parts.append("")

        if recent_thinking:
            parts.append(recent_thinking)
            parts.append("")

        parts.append(f"DISCUSSION PHASE — You are: {player_id}")
        sw = secret_word if secret_word else "??? (you're the imposter)"
        parts.append(f"Topic: {topic} | Secret Word: {sw}")
        parts.append(f"Grid: {', '.join(image_grid)}\n")

        parts.append("Hints this round:")
        for pid, hint in hints.items():
            marker = " (you)" if pid == player_id else ""
            parts.append(f"  {pid}{marker}: \"{hint}\"")

        # Unified chronological timeline: ALL DMs kept, group trimmed to 12
        timeline = PromptBuilder._merge_chat_timeline(
            recent_group_messages, dm_threads, player_id,
        )
        if timeline:
            parts.append("\nChat so far:")
            for msg in timeline:
                parts.append(PromptBuilder._format_chat_line(msg, player_id))

        # Show other players as potential DM targets with their hints
        other_players = [pid for pid in hints if pid != player_id]
        if other_players:
            parts.append("\nPlayers you could DM privately about their hints:")
            for pid in other_players:
                parts.append(f"  {pid} — hint: \"{hints[pid]}\"")

        parts.append(
            "\nREMINDER: Do NOT type the secret word or grid words in your message. Reference them indirectly."
        )
        parts.append(PromptBuilder._build_what_to_do(chat_state, my_msg_count, is_imposter=secret_word is None))

        # Time awareness (only 30-60s warning; <30s handled by chat_state="ending")
        if 30 <= time_remaining < 60:
            parts.append(f"\n>>> About {int(time_remaining)}s left in discussion.")

        parts.append(
            '\nJSON: {"thinking": "...", "action": {"type": "group_message"|"dm"|"wait", "content": "...", "target_player": "..."}}'
        )

        if reply_to_dm_from:
            parts.append(
                f'\n>>> YOU JUST RECEIVED A DM FROM {reply_to_dm_from}. '
                f'Reply via DM (type: "dm", target_player: "{reply_to_dm_from}"). '
                'RESPOND to what they said — agree, disagree, share your theory, or build on their point. '
                'Do NOT ask them a question they already answered. Do NOT rephrase your last question. '
                'Do NOT reply in group chat — that exposes the private conversation.'
            )

        return "\n".join(parts)

    @staticmethod
    def build_voting_prompt(
        topic: str,
        secret_word: str | None,
        image_grid: list[str],
        hints: dict[str, str],
        recent_group_messages: list[dict],
        dm_threads: dict[str, list[dict]],
        candidates: list[str],
        memory_context: str,
        player_id: str = "",
        num_players: int = 0,
        num_rounds: int = 0,
        player_list: list[str] | None = None,
        recent_thinking: str | None = None,
    ) -> str:
        parts = [PromptBuilder.build_system_prompt(num_players, num_rounds, player_list or []), ""]

        if memory_context.strip():
            parts.append(memory_context)
            parts.append("")

        if recent_thinking:
            parts.append(recent_thinking)
            parts.append("")

        sw = secret_word if secret_word else "??? (you were the imposter)"
        if player_id:
            parts.append(f"VOTING — You are: {player_id}")
        parts.append(f"Topic: {topic} | Secret Word: {sw}")
        parts.append(f"Grid: {', '.join(image_grid)}\n")

        parts.append("Hints:")
        for pid, hint in hints.items():
            parts.append(f"  {pid}: \"{hint}\"")

        # Unified chronological timeline: ALL DMs kept, group trimmed to 12
        timeline = PromptBuilder._merge_chat_timeline(
            recent_group_messages, dm_threads, player_id,
        )
        if timeline:
            parts.append("\nKey discussion moments:")
            for msg in timeline:
                parts.append(PromptBuilder._format_chat_line(msg, player_id))

        others = [c for c in candidates if c != player_id]
        parts.append(f"\nVote for who you think is the imposter: {', '.join(others)}")
        parts.append(
            "Base your vote on SPECIFIC evidence: a hint that didn't fit, something suspicious they said, or how they reacted to accusations."
            "\nThink about: whose hint didn't fit? who was deflecting? who was too vague?"
            '\n\nJSON: {"thinking": "...", "action": {"type": "vote", "target_player": "Player_X"}}'
        )
        return "\n".join(parts)

    @staticmethod
    def build_reflection_prompt(
        round_number: int,
        topic: str,
        secret_word: str | None,
        image_grid: list[str],
        hints: dict[str, str],
        discussion: list[dict],
        result: dict | None,
        memory_context: str,
        player_id: str = "",
        other_players: list[str] | None = None,
        num_players: int = 0,
        num_rounds: int = 0,
        player_list: list[str] | None = None,
    ) -> str:
        parts = [PromptBuilder.build_system_prompt(num_players, num_rounds, player_list or []), ""]

        if memory_context.strip():
            parts.append(memory_context)
            parts.append("")

        parts.append(f"REFLECTION — Round {round_number}")
        if player_id:
            parts.append(f"You are: {player_id}")
        parts.append(f"Topic: {topic}, Secret Word: {secret_word}")
        parts.append(f"Grid: {', '.join(image_grid)}\n")

        parts.append("Hints:")
        for pid, hint in hints.items():
            parts.append(f"  {pid}: \"{hint}\"")

        if result:
            imp = result.get("imposter_id", "?")
            caught = result.get("imposter_caught", False)
            parts.append(f"\nResult: {imp} was the imposter ({'caught' if caught else 'survived'})")

        other_players_list = other_players or []
        other_players_str = ", ".join(other_players_list) if other_players_list else "all other players"

        parts.append(
            "\nReflect on each player's behavior this round:"
            "\n- Did their hint seem natural or AI-generated?"
            "\n- How did they talk in discussion? Casual or robotic?"
            "\n- Did they use slang, emojis, typos?"
            "\n- Were they too analytical or too perfect?"
            f"\n\nProvide suspicion_updates for ALL other players: {other_players_str}"
            "\nConfidence: 0.0 = definitely AI, 1.0 = definitely human."
            '\n\nJSON: {"thinking": "...", "action": {"type": "reflection", "round_summary": "1-2 sentences", "suspicion_updates": {"Player_X": {"confidence": 0.7, "note": "short reason"}, ...for ALL other players}}}'
        )
        return "\n".join(parts)

    @staticmethod
    def build_classification_prompt(
        all_rounds_memory: str,
        players: list[str],
        player_id: str,
        memory_tracker: dict[str, dict],
        num_players: int = 0,
        num_rounds: int = 0,
        player_list: list[str] | None = None,
    ) -> str:
        parts = [PromptBuilder.build_system_prompt(num_players, num_rounds, player_list or []), ""]
        parts.append(f"FINAL META-GAME — You are: {player_id}")
        parts.append("Classify each other player as human or AI.\n")
        parts.append(all_rounds_memory)
        parts.append("")

        parts.append("Players to classify:")
        for pid in players:
            if pid == player_id:
                continue
            obs = memory_tracker.get(pid, {})
            conf = obs.get("human_confidence", 0.5)
            parts.append(f"  {pid} (your current gut feeling: human={conf:.1f})")

        parts.append(
            "\nConsider across ALL rounds:"
            "\n- Hint style: creative/slangy (human) vs clean/definitional (AI)"
            "\n- Chat style: messy/casual (human) vs structured/polite (AI)"
            "\n- Did they use abbreviations, typos, emojis?"
            "\n- Were they too consistent? Too perfect? Too analytical?"
            "\n- DM behavior: did they form alliances? Share real suspicions?"
            '\n\nJSON: {"thinking": "...", "action": {"type": "meta_classification", "classifications": {'
            '"Player_1": {"guess": "human"|"ai", "confidence": 1-5, "reasoning": "..."}, ...}}}'
        )
        return "\n".join(parts)
