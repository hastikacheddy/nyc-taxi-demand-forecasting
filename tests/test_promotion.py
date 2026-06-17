"""Tests for the champion-challenger promotion gate."""
from unittest.mock import MagicMock

from src.common.promotion import promote_if_better


def _champion_with_mae(mae):
    champ = MagicMock()
    champ.tags = {"holdout_mae": str(mae)}
    return champ


def test_promotes_when_no_incumbent():
    client = MagicMock()
    client.get_model_version_by_alias.side_effect = Exception("no champion")
    assert promote_if_better(client, "M", "5", new_mae=10.0) is True
    client.set_registered_model_alias.assert_called_once_with("M", "champion", "5")


def test_promotes_when_strictly_better():
    client = MagicMock()
    client.get_model_version_by_alias.return_value = _champion_with_mae(20.0)
    assert promote_if_better(client, "M", "6", new_mae=15.0) is True
    client.set_registered_model_alias.assert_called_once_with("M", "champion", "6")


def test_rejects_and_parks_as_challenger_when_worse():
    client = MagicMock()
    client.get_model_version_by_alias.return_value = _champion_with_mae(10.0)
    assert promote_if_better(client, "M", "7", new_mae=25.0) is False
    client.set_registered_model_alias.assert_called_once_with("M", "challenger", "7")


def test_records_holdout_mae_tag():
    client = MagicMock()
    client.get_model_version_by_alias.return_value = _champion_with_mae(10.0)
    promote_if_better(client, "M", "8", new_mae=9.0)
    client.set_model_version_tag.assert_any_call("M", "8", "holdout_mae", "9.000000")


def test_min_improvement_blocks_marginal_gain():
    client = MagicMock()
    client.get_model_version_by_alias.return_value = _champion_with_mae(10.0)
    # 9.9 is better but not by the required 0.5 margin -> rejected
    assert promote_if_better(client, "M", "9", new_mae=9.9, min_improvement=0.5) is False


def test_non_finite_mae_never_promotes():
    client = MagicMock()
    assert promote_if_better(client, "M", "10", new_mae=float("inf")) is False
    assert promote_if_better(client, "M", "11", new_mae=float("nan")) is False
    client.set_registered_model_alias.assert_not_called()
