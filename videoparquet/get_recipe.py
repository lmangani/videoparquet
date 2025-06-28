def get_recipe(df, variables=None, shape=None, bits=8, n_components=0, lossy_params=None, lossless_params=None):
    '''
    Returns a default recipe for compressing a Parquet DataFrame using parquet2video.
    Parameters:
        df: The input DataFrame loaded from Parquet.
        variables: List of columns/arrays to include (default: all columns).
        shape: Shape to reshape each array to (required for video encoding).
        bits: Bit depth for encoding.
        n_components: Number of PCA components (0 = no PCA).
        lossy_params: Dict of lossy codec params.
        lossless_params: Dict of lossless codec params.
    '''
    if variables is None:
        variables = list(df.columns)
    if lossy_params is None:
        lossy_params = {'c:v': 'libx264', 'crf': 18}
    if lossless_params is None:
        lossless_params = {'c:v': 'ffv1'}
    recipe = {}
    # For simplicity, treat all as lossy for now
    recipe['main'] = (variables, shape, n_components, lossy_params, bits)
    return recipe 