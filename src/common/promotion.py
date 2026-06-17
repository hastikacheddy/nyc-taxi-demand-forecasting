"""
Champion-challenger promotion gate.

A newly trained model version is NOT blindly aliased @champion. It is promoted
only if its out-of-sample holdout MAE beats the incumbent champion's by the
configured margin; otherwise it is parked as @challenger and the champion is
left untouched. Every version records its holdout_mae tag so future runs can
compare. This is what stops a degraded retrain from silently reaching prod.
"""
import logging

logger = logging.getLogger(__name__)


def promote_if_better(client, name, new_version, new_mae, min_improvement=0.0):
    """
    Decide whether to promote `new_version` of model `name` to @champion.

    Args:
        client: mlflow.MlflowClient
        name: registered model name
        new_version: version string just registered
        new_mae: the candidate's out-of-sample holdout MAE (lower is better)
        min_improvement: required absolute MAE improvement over the champion

    Returns:
        True if promoted to @champion, False if parked as @challenger.
    """
    # Never promote a model with an undefined metric.
    if new_mae is None or new_mae != new_mae or new_mae == float("inf"):
        logger.error("[promotion] %s v%s: non-finite holdout MAE — refusing to promote", name, new_version)
        return False

    # Round to the stored precision so an identical retrain compares equal
    # (avoids a spurious reject from float noise beyond 6 decimals).
    new_mae = round(float(new_mae), 6)
    client.set_model_version_tag(name, new_version, "holdout_mae", f"{new_mae:.6f}")

    try:
        champ = client.get_model_version_by_alias(name, "champion")
    except Exception:
        champ = None

    if champ is None:
        client.set_registered_model_alias(name, "champion", new_version)
        logger.info("[promotion] %s: no incumbent — promoting v%s (mae=%.4f)",
                    name, new_version, new_mae)
        return True

    try:
        champ_mae = float(champ.tags.get("holdout_mae", "inf"))
    except (TypeError, ValueError):
        champ_mae = float("inf")

    if new_mae <= champ_mae - min_improvement:
        client.set_registered_model_alias(name, "champion", new_version)
        logger.info("[promotion] %s: PROMOTE v%s (mae %.4f <= champion %.4f)",
                    name, new_version, new_mae, champ_mae)
        return True

    client.set_registered_model_alias(name, "challenger", new_version)
    logger.warning("[promotion] %s: REJECT v%s (mae %.4f > champion %.4f) — champion unchanged",
                   name, new_version, new_mae, champ_mae)
    return False
