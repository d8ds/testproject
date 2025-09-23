def pure_polars_rolling(df):
    """
    Pure Polars solution that works with large datasets
    """
    # First, get all the non-null data
    clean_df = (
        df
        .filter(pl.col('section').is_not_null())
        .sort(['qid', 'date'])
    )
    
    # Create a helper function for each group
    def rolling_agg(group_df):
        dates = group_df.get_column('date').to_list()
        sections = group_df.get_column('section').to_list()
        
        result_sections = []
        for i, current_date in enumerate(dates):
            start_date = current_date - pl.duration(days=180)
            
            # Get all sections within the window
            window_sections = [
                sections[j] for j, date in enumerate(dates)
                if start_date <= date <= current_date
            ]
            result_sections.append(window_sections)
        
        return group_df.with_columns([
            pl.Series('sections_180d', result_sections)
        ])
    
    # Process by qid groups
    return (
        clean_df
        .group_by('qid', maintain_order=True)
        .map_groups(rolling_agg)
    )
