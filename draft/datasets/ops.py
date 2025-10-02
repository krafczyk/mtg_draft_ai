import pandas as pd
from draft.utils import sort_card_df


def attach_card_metadata_row(df, card_df, priority_sets=None):
    booster_cards_df = card_df[card_df['is_booster']]
    b_cards_df = booster_cards_df.groupby('name', sort=False)

    def map_func(r):
        df = card_df[card_df['is_booster']]
        try:
            g = b_cards_df.get_group(r.name)
        except KeyError:
            return pd.Series({
                'abundance': 0.,
                'name': pd.NA, 
                'rarity': pd.NA,
                'expansion': pd.NA })
        if len(g) > 1 and priority_sets is not None and isinstance(priority_sets, list):
            for prio_set in priority_sets:
                if g['expansion'].str.contains(prio_set).any():
                    g = g[g['expansion'] == prio_set]
                    break
        g['abundance'] = r.iloc[0]
        return g[['expansion', 'rarity', 'abundance']].iloc[-1,:]

    return df.apply(map_func, axis=1)

def attach_card_metadata_2(df, card_df, priority_sets=None):
    """Attach card metadata to a dataframe to card names in the column index"""

    booster_cards_df = card_df[card_df['is_booster']]
    b_cards_df = booster_cards_df.groupby('name', sort=False)

    def map_func(c):
        try:
            g = b_cards_df.get_group(r.name)
        except KeyError:
            return pd.Series({
                'abundance': 0.,
                'name': pd.NA, 
                'rarity': pd.NA,
                'expansion': pd.NA })
        if len(g) > 1 and priority_sets is not None and isinstance(priority_sets, list):
            for prio_set in priority_sets:
                if g['expansion'].str.contains(prio_set).any():
                    g = g[g['expansion'] == prio_set]
                    break
        g['abundance'] = r.iloc[0]
        return g[['expansion', 'rarity', 'abundance']].iloc[-1,:]

    return df.apply(map_func, axis=1)


def attach_card_metadata(
    df: pd.DataFrame,
    card_df: pd.DataFrame,
    *,
    index: str = "columns",           # "columns" or "rows"
    meta=("expansion", "rarity"),           # which metadata columns to attach (in order)
    priority_sets=('MKM', 'OTP', 'SPG'),
) -> pd.DataFrame:
    """
    Attach metadata from `card_df` to either df.columns or df.index, producing a MultiIndex:
    level-0 = card name, subsequent levels = `meta`.

    - `card_df` must have one row per card; either:
        * indexed by card name, or
        * contain a `name_col` with names (will be set as index).
    """
    axis = 1 if str(index).lower() in {"columns", "cols", "col"} or index == 1 else 0
    labels = df.columns if axis == 1 else df.index

    name_col = 'name'

    if meta is None:
        meta = [c for c in base.columns]  # attach all columns if not specified

    temp_df = sort_card_df(card_df)[[name_col]+list(meta)].groupby(name_col).first()
    temp_df[name_col] = temp_df.index.to_series()
    temp_df = temp_df.set_index([name_col]+list(meta))
    new_labels = temp_df.loc[labels,:].index

    out = df.copy()
    if axis == 1:
        out.columns = new_labels
    else:
        out.index = new_labels
    return out
