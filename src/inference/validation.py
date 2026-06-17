import pandera as pa
from pandera.typing import Series
import pandas as pd

class DemandDataSchema(pa.DataFrameModel):
    TimePeriod: Series[pa.DateTime] = pa.Field(nullable=False)
    # ge/le bounds reject negatives, NaN (nullable=False) and inf (le bound) —
    # the data-poisoning vectors an adversary could feed the circuit breaker.
    Volume: Series[float] = pa.Field(ge=0.0, le=1_000_000.0, nullable=False)

def validate_input_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates incoming trip data before inference.
    Raises pandera.errors.SchemaError if validation fails, triggering the circuit breaker.
    """
    df = df.copy()
    # Strip timezone so pandera's datetime64[ns] check passes regardless of source tz
    if isinstance(df['TimePeriod'].dtype, pd.DatetimeTZDtype):
        df['TimePeriod'] = df['TimePeriod'].dt.tz_convert('UTC').dt.tz_localize(None)
    # Coerce Volume to float64 to satisfy schema regardless of source dtype
    df['Volume'] = df['Volume'].astype(float)
    try:
        return DemandDataSchema.validate(df)
    except pa.errors.SchemaError as e:
        raise e
