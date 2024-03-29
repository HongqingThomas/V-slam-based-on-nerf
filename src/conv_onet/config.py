from src.conv_onet import models


def get_model(cfg):
    """
    Return the network model.

    Args:
        cfg (dict): imported yaml config.

    Returns:
        decoder (nn.module): the network model.
    """

    dim = cfg['data']['dim']
    coarse_grid_len = cfg['grid_len']['coarse']
    middle_grid_len = cfg['grid_len']['middle']
    fine_grid_len = cfg['grid_len']['fine']
    color_grid_len = cfg['grid_len']['color']
    c_dim = cfg['model']['c_dim']  # feature dimensions
    pos_embedding_method = cfg['model']['pos_embedding_method']
    decoder = models.decoder_dict['nice'](
        dim=dim, c_dim=c_dim, coarse=cfg['coarse'], coarse_grid_len=coarse_grid_len,
        middle_grid_len=middle_grid_len, fine_grid_len=fine_grid_len,
        color_grid_len=color_grid_len, pos_embedding_method=pos_embedding_method)
    return decoder
