import pandas as pd


def attach_card_metadata(df, card_df, priority_sets=None):
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
