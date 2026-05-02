"""Loads and serves word bank entries for each round."""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

from server.models import GridItem, WordBankEntry


class WordBank:
    """Loads and serves word bank entries for each round."""

    def __init__(self, data_path: str = "data/word_bank.json") -> None:
        self._data_path = Path(data_path)
        self.entries: list[WordBankEntry] = []
        self._used_indices: set[int] = set()
        self._round_cache: dict[int, WordBankEntry] = {}

    def load(self) -> None:
        """Read the JSON file and parse each entry as a WordBankEntry.

        Raises:
            FileNotFoundError: If the word bank JSON file does not exist.
            ValueError: If the JSON content is invalid or fails validation.
        """
        if not self._data_path.exists():
            raise FileNotFoundError(f"Word bank file not found: {self._data_path}")

        try:
            raw_text = self._data_path.read_text(encoding="utf-8")
            raw_data = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in word bank file: {exc}") from exc

        if not isinstance(raw_data, list):
            raise ValueError("Word bank JSON must be a top-level array")

        entries: list[WordBankEntry] = []
        for idx, item in enumerate(raw_data):
            try:
                entries.append(WordBankEntry(**item))
            except Exception as exc:
                raise ValueError(
                    f"Invalid word bank entry at index {idx}: {exc}"
                ) from exc

        # Shuffle so topic order is random each game session.
        random.shuffle(entries)
        self.entries = entries

    def get_entry_for_round(self, round_number: int) -> WordBankEntry:
        """Return a copy of the entry for the given round with shuffled grid order.

        If round_number exceeds available entries, cycles using modulo.
        Marks the entry index as used.

        Args:
            round_number: The 1-based round number.

        Returns:
            A deep copy of the WordBankEntry with grid_items in random order.
        """
        if round_number in self._round_cache:
            return copy.deepcopy(self._round_cache[round_number])

        if not self.entries:
            raise ValueError("Word bank is empty — call load() first")

        index = (round_number - 1) % len(self.entries)
        self._used_indices.add(index)

        entry = copy.deepcopy(self.entries[index])
        # Pick a random grid item as the secret word each round.
        entry.secret_word = random.choice([item.label for item in entry.grid_items])
        entry.grid_items = self.get_shuffled_grid(entry)
        self._round_cache[round_number] = copy.deepcopy(entry)
        return entry

    def get_shuffled_grid(self, entry: WordBankEntry) -> list[GridItem]:
        """Return a new list with grid_items in random order.

        Args:
            entry: The WordBankEntry whose grid to shuffle.

        Returns:
            A randomly ordered copy of the grid_items list.
        """
        return random.sample(entry.grid_items, len(entry.grid_items))

    def get_grid_labels(self, entry: WordBankEntry) -> list[str]:
        """Return just the labels from an entry's grid items.

        Args:
            entry: The WordBankEntry to extract labels from.

        Returns:
            A list of label strings in the same order as grid_items.
        """
        return [item.label for item in entry.grid_items]

    def reset(self) -> None:
        """Clear used indices for a new game session."""
        self._used_indices.clear()

    def available_count(self) -> int:
        """Return the total number of entries in the word bank."""
        return len(self.entries)
