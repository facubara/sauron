"""Unit tests for settings, phrases and stats persistence."""

import json

import config
from config import (
    Settings,
    clean_streak,
    load_phrases,
    load_settings,
    load_stats,
    parse_phrases,
    record_bite,
    save_stats,
    settings_from_dict,
    today_count,
    write_settings,
)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
def test_defaults_are_valid():
    assert Settings().validate() == []


def test_settings_from_dict_overrides_known_keys():
    s, warnings = settings_from_dict({"bite_dwell_seconds": 3, "start_muted": True,
                                      "camera_index": 1.0})
    assert warnings == []
    assert s.bite_dwell_seconds == 3.0 and isinstance(s.bite_dwell_seconds, float)
    assert s.start_muted is True
    assert s.camera_index == 1 and isinstance(s.camera_index, int)


def test_settings_from_dict_warns_and_keeps_defaults_on_bad_values():
    s, warnings = settings_from_dict({
        "bite_dwell_seconds": "soon",   # wrong type
        "start_muted": "yes",           # bool must be real bool
        "not_a_setting": 1,             # unknown key
        "volume": 7,                    # out of range -> clamped
        "clear_seconds": -1,            # must be > 0 -> default
        "log_level": "loud",            # unknown level
    })
    assert s.bite_dwell_seconds == Settings().bite_dwell_seconds
    assert s.start_muted is False
    assert s.volume == 1.0
    assert s.clear_seconds == Settings().clear_seconds
    assert s.log_level == "INFO"
    joined = "\n".join(warnings)
    for needle in ("bite_dwell_seconds", "start_muted", "not_a_setting",
                   "volume", "clear_seconds", "log_level"):
        assert needle in joined


def test_ratio_band_is_swapped_if_inverted():
    s, warnings = settings_from_dict({"hand_face_ratio_min": 2.0,
                                      "hand_face_ratio_max": 0.5})
    assert (s.hand_face_ratio_min, s.hand_face_ratio_max) == (0.5, 2.0)
    assert any("swapping" in w for w in warnings)


def test_load_settings_creates_default_file(tmp_path):
    path = tmp_path / "sub" / "config.json"
    s, warnings = load_settings(str(path))
    assert s == Settings() and warnings == []
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8"))["bite_dwell_seconds"] == 1.5


def test_load_settings_round_trip(tmp_path):
    path = tmp_path / "config.json"
    custom = Settings(snooze_minutes=10, camera_index=2)
    write_settings(custom, str(path))
    loaded, warnings = load_settings(str(path))
    assert loaded == custom and warnings == []


def test_load_settings_corrupt_file_falls_back(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{not json", encoding="utf-8")
    s, warnings = load_settings(str(path))
    assert s == Settings()
    assert warnings and "could not read" in warnings[0]


def test_load_settings_non_object_falls_back(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("[1, 2]", encoding="utf-8")
    s, warnings = load_settings(str(path))
    assert s == Settings()
    assert warnings


# ---------------------------------------------------------------------------
# Phrases
# ---------------------------------------------------------------------------
def test_parse_phrases_skips_comments_and_blanks():
    text = "# header\n\n  First phrase.  \n#comment\nSecond phrase.\n"
    assert parse_phrases(text) == ["First phrase.", "Second phrase."]


def test_load_phrases_prefers_user_file(tmp_path):
    user = tmp_path / "user"
    bundled = tmp_path / "bundled"
    user.mkdir()
    bundled.mkdir()
    (user / "phrases.txt").write_text("user phrase\n", encoding="utf-8")
    (bundled / "phrases.txt").write_text("bundled phrase\n", encoding="utf-8")
    assert load_phrases(str(user), str(bundled)) == ["user phrase"]


def test_load_phrases_falls_through_empty_user_file(tmp_path):
    user = tmp_path / "user"
    bundled = tmp_path / "bundled"
    user.mkdir()
    bundled.mkdir()
    (user / "phrases.txt").write_text("# only comments\n", encoding="utf-8")
    (bundled / "phrases.txt").write_text("bundled phrase\n", encoding="utf-8")
    assert load_phrases(str(user), str(bundled)) == ["bundled phrase"]


def test_load_phrases_builtin_fallback(tmp_path):
    assert load_phrases(str(tmp_path), str(tmp_path)) == config.DEFAULT_PHRASES


def test_bundled_phrases_file_is_usable():
    phrases = load_phrases(user_dir="/nonexistent", bundled_dir=config.SCRIPT_DIR)
    assert len(phrases) > 20
    assert all(not p.startswith("#") for p in phrases)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
def test_stats_missing_file(tmp_path):
    stats = load_stats(str(tmp_path / "stats.json"))
    assert stats == {"version": 2, "days": {}}
    assert today_count(stats, "2026-10-06") == 0


def test_stats_record_and_round_trip(tmp_path):
    path = str(tmp_path / "stats.json")
    stats = load_stats(path)
    assert record_bite(stats, "2026-10-06") == 1
    assert record_bite(stats, "2026-10-06") == 2
    assert record_bite(stats, "2026-10-07") == 1
    save_stats(stats, path)
    again = load_stats(path)
    assert today_count(again, "2026-10-06") == 2
    assert today_count(again, "2026-10-07") == 1
    assert not (tmp_path / "stats.json.tmp").exists()


def test_stats_upgrades_v1_format(tmp_path):
    path = tmp_path / "stats.json"
    path.write_text(json.dumps({"date": "2026-10-05", "count": 4}), encoding="utf-8")
    stats = load_stats(str(path))
    assert stats["days"] == {"2026-10-05": 4}
    assert today_count(stats, "2026-10-06") == 0


def test_stats_tolerates_garbage(tmp_path):
    path = tmp_path / "stats.json"
    path.write_text("not json", encoding="utf-8")
    assert load_stats(str(path))["days"] == {}
    path.write_text(json.dumps({"days": {"2026-10-01": "three", "2026-10-02": 2}}),
                    encoding="utf-8")
    assert load_stats(str(path))["days"] == {"2026-10-02": 2}


def test_clean_streak():
    assert clean_streak({"days": {}}, "2026-10-06") == 0
    days = {"2026-10-01": 2, "2026-10-02": 0}
    assert clean_streak({"days": days}, "2026-10-06") == 4  # 2nd..5th clean
    days["2026-10-05"] = 1
    assert clean_streak({"days": days}, "2026-10-06") == 0
    # Bites today do not break the streak shown for previous days
    days = {"2026-10-03": 1, "2026-10-06": 3}
    assert clean_streak({"days": days}, "2026-10-06") == 2
