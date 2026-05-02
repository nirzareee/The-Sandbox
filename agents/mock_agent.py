"""Mock agent for testing — no LLM API calls required."""

from __future__ import annotations

import json
import random

from agents.base_agent import BaseAgent


# Canned hint pools
_INSIDER_HINTS = [
    "nice one", "classic", "easy pick", "vibes", "yep",
    "obvious", "simple", "you know it", "no brainer", "for sure",
]
_IMPOSTER_HINTS = [
    "interesting", "hmm", "tricky", "not sure", "yeah",
    "could be", "tough call", "maybe", "wild guess", "idk",
]
_GROUP_MESSAGES = [
    "I think it's {target}",
    "hmm suspicious",
    "agreed",
    "not sure about this",
    "idk anyone could be it",
    "lol who even knows",
    "{target} is being weird",
    "seems fine to me",
]
_DM_MESSAGES = [
    "what do you think?",
    "who are you voting for?",
    "kinda sus ngl",
    "lets team up",
]


class MockAgent(BaseAgent):
    """AI agent that returns canned/random responses — no LLM API calls needed."""

    def __init__(
        self,
        player_id: str,
        display_name: str,
        server_url: str = "ws://localhost:8000",
    ) -> None:
        super().__init__(
            player_id=player_id,
            display_name=display_name,
            model_name="mock-llm",
            server_url=server_url,
        )

    async def call_llm(self, prompt: str) -> str:
        """Return canned JSON responses based on prompt context."""
        prompt_lower = prompt.lower()

        if "give a short hint" in prompt_lower or "give a plausible hint" in prompt_lower:
            return self._mock_hint()
        elif "discussion phase" in prompt_lower:
            return self._mock_discussion()
        elif "voting phase" in prompt_lower:
            return self._mock_vote()
        elif "final meta-game" in prompt_lower:
            return self._mock_classification()
        elif "reflection" in prompt_lower:
            return self._mock_reflection()

        # Fallback — safe no-op
        return json.dumps({
            "thinking": "mock fallback",
            "action": {"type": "wait"},
        })

    # ------------------------------------------------------------------
    # Response generators
    # ------------------------------------------------------------------

    def _mock_hint(self) -> str:
        rs = self.memory.current_round_state
        secret_word = rs.get("secret_word")

        if secret_word:
            # Insider — pick from insider pool
            hint = random.choice(_INSIDER_HINTS)
        else:
            # Imposter — pick from imposter pool
            hint = random.choice(_IMPOSTER_HINTS)

        return json.dumps({
            "thinking": "mock hint logic",
            "action": {"type": "hint", "content": hint},
        })

    def _mock_discussion(self) -> str:
        others = [p for p in self._all_players if p != self.player_id]
        roll = random.random()

        if roll < 0.5 and others:
            target = random.choice(others)
            msg = random.choice(_GROUP_MESSAGES).format(target=target)
            return json.dumps({
                "thinking": "mock discussion",
                "action": {"type": "group_message", "content": msg},
            })
        elif roll < 0.8 and others:
            target = random.choice(others)
            msg = random.choice(_DM_MESSAGES)
            return json.dumps({
                "thinking": "mock discussion",
                "action": {"type": "dm", "content": msg, "target_player": target},
            })
        else:
            return json.dumps({
                "thinking": "mock discussion — waiting",
                "action": {"type": "wait"},
            })

    def _mock_vote(self) -> str:
        others = [p for p in self._all_players if p != self.player_id]
        target = random.choice(others) if others else self.player_id
        return json.dumps({
            "thinking": "mock voting",
            "action": {"type": "vote", "target_player": target},
        })

    def _mock_classification(self) -> str:
        classifications = {}
        for pid in self._all_players:
            if pid != self.player_id:
                classifications[pid] = {
                    "guess": random.choice(["human", "ai"]),
                    "confidence": 3,
                    "reasoning": "mock guess",
                }
        return json.dumps({
            "thinking": "mock classification",
            "classifications": classifications,
        })

    def _mock_reflection(self) -> str:
        return json.dumps({
            "thinking": "Everyone played well this round. Need to watch more carefully next time.",
            "action": {"type": "wait"},
        })
