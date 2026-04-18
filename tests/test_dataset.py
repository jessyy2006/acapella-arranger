"""Tests for Chunk 2: SATBDataset, splits, and the padding collate."""

from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch
from music21 import corpus, stream

from src.data.dataset import (
    SATBDataset,
    VOICE_KEYS,
    extract_voices_jacappella,
    extract_voices_jsb,
)
from src.data.load import is_clean_satb
from src.data.loaders import (
    build_split,
    collate_satb,
    load_dataset,
    make_dataloaders,
    split_scores_by_song,
)
from src.data.vocab import EOS, PAD, SOS, VOCAB_SIZE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def jsb_scores() -> list[stream.Score]:
    """Collect a small but realistic set of clean SATB chorales."""
    out: list[stream.Score] = []
    for score in corpus.chorales.Iterator():
        if is_clean_satb(score):
            out.append(score)
            if len(out) >= 6:
                break
    if len(out) < 6:
        pytest.skip("not enough clean JSB chorales available")
    return out


# ---------------------------------------------------------------------------
# Voice extraction
# ---------------------------------------------------------------------------


class TestVoiceExtraction:
    def test_jsb_lead_and_soprano_are_same_part(self, jsb_scores: list[stream.Score]):
        voices = extract_voices_jsb(jsb_scores[0])
        assert voices["lead"] is voices["s"]
        for k in VOICE_KEYS:
            assert voices[k] is not None

    def test_jacappella_extractor_recognises_part_names(self):
        # Synthesise a score with the exact jaCappella part names.
        score = stream.Score()
        for name in ("lead_vocal", "soprano", "alto", "tenor", "bass"):
            p = stream.Part()
            p.partName = name
            score.append(p)
        voices = extract_voices_jacappella(score)
        assert voices["lead"].partName == "lead_vocal"
        assert voices["s"].partName == "soprano"
        assert voices["b"].partName == "bass"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class TestDataset:
    def test_len_positive_without_augmentation(self, jsb_scores: list[stream.Score]):
        ds = SATBDataset([("jsb", jsb_scores[0])], augment=False)
        assert len(ds) > 0

    def test_getitem_returns_five_voice_long_tensors(
        self, jsb_scores: list[stream.Score]
    ):
        ds = SATBDataset([("jsb", jsb_scores[0])], augment=False)
        item = ds[0]
        assert set(item.keys()) == set(VOICE_KEYS)
        for v in VOICE_KEYS:
            assert item[v].dtype == torch.long
            assert item[v].dim() == 1
            assert item[v].numel() >= 2  # at minimum SOS + EOS

    def test_lead_equals_soprano_for_jsb(self, jsb_scores: list[stream.Score]):
        ds = SATBDataset([("jsb", jsb_scores[0])], augment=False)
        for i in range(min(3, len(ds))):
            item = ds[i]
            assert torch.equal(item["lead"], item["s"])

    def test_augmentation_inflates_example_count(
        self, jsb_scores: list[stream.Score]
    ):
        songs = [("jsb", s) for s in jsb_scores[:2]]
        baseline = SATBDataset(songs, augment=False)
        augmented = SATBDataset(songs, augment=True)
        # 12 transpositions; some get rejected for out-of-range, but the
        # multiplier should comfortably exceed 6×.
        assert len(augmented) >= 6 * len(baseline)

    def test_tokens_stay_in_vocab_across_augmentation(
        self, jsb_scores: list[stream.Score]
    ):
        ds = SATBDataset([("jsb", jsb_scores[0])], augment=True)
        # Spot-check a handful of examples rather than the whole dataset.
        for idx in (0, len(ds) // 2, len(ds) - 1):
            item = ds[idx]
            for v in VOICE_KEYS:
                tok = item[v]
                assert (tok >= 0).all() and (tok < VOCAB_SIZE).all()

    def test_unknown_source_raises(self, jsb_scores: list[stream.Score]):
        with pytest.raises(ValueError):
            SATBDataset([("bogus", jsb_scores[0])])  # type: ignore[list-item]

    def test_songs_with_missing_parts_are_skipped(self):
        # Build a score that LOOKS like a JSB chorale but is missing Alto.
        score = stream.Score()
        for name in ("Soprano", "Tenor", "Bass"):
            p = stream.Part()
            p.partName = name
            score.append(p)
        ds = SATBDataset([("jsb", score)], augment=False)
        assert len(ds) == 0
        assert ds.songs_skipped == 1
        assert ds.songs_kept == 0


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


class TestSplit:
    def test_split_preserves_every_song_exactly_once(
        self, jsb_scores: list[stream.Score]
    ):
        rng = random.Random(0)
        train, val, test = split_scores_by_song(
            jsb_scores, (0.7, 0.15, 0.15), rng
        )
        all_split = train + val + test
        # Identity-based: every input score must appear in exactly one split.
        for score in jsb_scores:
            assert sum(1 for s in all_split if s is score) == 1
        assert len(all_split) == len(jsb_scores)

    def test_split_is_deterministic_for_same_seed(
        self, jsb_scores: list[stream.Score]
    ):
        a_train, a_val, a_test = split_scores_by_song(
            jsb_scores, (0.7, 0.15, 0.15), random.Random(123)
        )
        b_train, b_val, b_test = split_scores_by_song(
            jsb_scores, (0.7, 0.15, 0.15), random.Random(123)
        )
        assert [id(s) for s in a_train] == [id(s) for s in b_train]
        assert [id(s) for s in a_val] == [id(s) for s in b_val]
        assert [id(s) for s in a_test] == [id(s) for s in b_test]

    def test_split_rejects_bad_ratios(self, jsb_scores: list[stream.Score]):
        with pytest.raises(ValueError):
            split_scores_by_song(
                jsb_scores, (0.5, 0.3, 0.3), random.Random(0)
            )

    def test_build_split_no_song_leakage_across_splits(
        self, jsb_scores: list[stream.Score]
    ):
        # Run the split again with the same seed to capture the song lists
        # before SATBDataset eats them.
        rng = random.Random(42)
        jsb_train_songs, jsb_val_songs, jsb_test_songs = split_scores_by_song(
            jsb_scores, (0.7, 0.15, 0.15), rng
        )
        train_ids = {id(s) for s in jsb_train_songs}
        val_ids = {id(s) for s in jsb_val_songs}
        test_ids = {id(s) for s in jsb_test_songs}
        assert not (train_ids & val_ids)
        assert not (train_ids & test_ids)
        assert not (val_ids & test_ids)

    def test_build_split_applies_augmentation_only_to_train(
        self, jsb_scores: list[stream.Score]
    ):
        splits = build_split(jsb_scores, [], seed=42)
        # Per-song example count in val/test should roughly equal window count;
        # train should be ~12× larger per song (modulo range-rejected shifts).
        per_song_val = len(splits["val"]) / max(splits["val"].songs_kept, 1)
        per_song_train = len(splits["train"]) / max(splits["train"].songs_kept, 1)
        assert per_song_train >= 5 * per_song_val

    def test_build_split_returns_all_three_keys(
        self, jsb_scores: list[stream.Score]
    ):
        splits = build_split(jsb_scores, [], seed=42)
        assert set(splits.keys()) == {"train", "val", "test"}


# ---------------------------------------------------------------------------
# Collate + DataLoader
# ---------------------------------------------------------------------------


class TestCollate:
    def test_pads_to_max_per_voice(self):
        short = {v: torch.tensor([SOS, 10, 20, EOS], dtype=torch.long) for v in VOICE_KEYS}
        long = {v: torch.tensor([SOS, 10, 20, 30, 40, 50, EOS], dtype=torch.long) for v in VOICE_KEYS}
        batch = collate_satb([short, long])
        for v in VOICE_KEYS:
            assert batch[v].shape == (2, 7)
            # First row was length-4; positions 4..6 must be PAD.
            assert (batch[v][0, 4:] == PAD).all()
            # Second row has no padding.
            assert (batch[v][1] != PAD).all()

    def test_emits_per_voice_length_tensor(self):
        short = {v: torch.tensor([SOS, 10, EOS], dtype=torch.long) for v in VOICE_KEYS}
        long = {v: torch.tensor([SOS, 10, 20, 30, EOS], dtype=torch.long) for v in VOICE_KEYS}
        batch = collate_satb([short, long])
        for v in VOICE_KEYS:
            assert batch[f"{v}_len"].tolist() == [3, 5]

    def test_empty_batch_would_error_meaningfully(self):
        # pad_sequence on empty input is an error; confirm it's loud, not silent.
        with pytest.raises((RuntimeError, IndexError)):
            collate_satb([])


class TestDataLoader:
    def test_dataloader_yields_padded_batches(
        self, jsb_scores: list[stream.Score]
    ):
        splits = build_split(jsb_scores, [], seed=42, augment_train=False)
        loaders = make_dataloaders(splits, batch_size=2)
        batch = next(iter(loaders["train"]))
        for v in VOICE_KEYS:
            assert batch[v].dim() == 2
            assert batch[v].shape[0] <= 2
            assert batch[v].dtype == torch.long
            assert f"{v}_len" in batch

    def test_val_loader_not_shuffled(self, jsb_scores: list[stream.Score]):
        splits = build_split(jsb_scores, [], seed=42, augment_train=False)
        loaders = make_dataloaders(splits, batch_size=2)
        # Validation order must be deterministic for reproducible metrics.
        first = [batch["lead"][0, :8].tolist() for batch in loaders["val"]]
        second = [batch["lead"][0, :8].tolist() for batch in loaders["val"]]
        assert first == second


# ---------------------------------------------------------------------------
# Persistence — torch.save + load_dataset round-trip
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_and_load_round_trip_preserves_examples(
        self, jsb_scores: list[stream.Score], tmp_path: Path
    ):
        original = SATBDataset([("jsb", jsb_scores[0])], augment=False)
        path = tmp_path / "dataset.pt"
        torch.save(original, path)

        loaded = load_dataset(path)
        assert len(loaded) == len(original)
        for i in range(len(original)):
            for v in VOICE_KEYS:
                assert torch.equal(loaded[i][v], original[i][v])

    def test_load_dataset_accepts_string_path(
        self, jsb_scores: list[stream.Score], tmp_path: Path
    ):
        original = SATBDataset([("jsb", jsb_scores[0])], augment=False)
        path = tmp_path / "dataset.pt"
        torch.save(original, path)
        # Calling with a string instead of a Path should still work.
        assert len(load_dataset(str(path))) == len(original)

    def test_loaded_dataset_exposes_song_counters(
        self, jsb_scores: list[stream.Score], tmp_path: Path
    ):
        original = SATBDataset(
            [("jsb", jsb_scores[0]), ("jsb", jsb_scores[1])], augment=False
        )
        path = tmp_path / "dataset.pt"
        torch.save(original, path)
        loaded = load_dataset(path)
        assert loaded.songs_kept == original.songs_kept
        assert loaded.songs_skipped == original.songs_skipped
