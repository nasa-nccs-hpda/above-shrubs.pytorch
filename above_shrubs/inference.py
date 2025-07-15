import torch
import numpy as np
from tiler import Tiler, Merger

from above_shrubs.processing import normalize
from above_shrubs.processing import global_standardization
from above_shrubs.processing import local_standardization
from above_shrubs.processing import standardize_image


# ---------------------------------------------------------------------------
# Module Methods
# ---------------------------------------------------------------------------
def sliding_window_tiler_multiclass(
            xraster,
            model,
            n_classes: int,
            img_size: int,
            pad_style: str = 'reflect',
            overlap: float = 0.50,
            constant_value: int = 600,
            batch_size: int = 1024,
            threshold: float = 0.50,
            standardization: str = None,
            mean=None,
            std=None,
            normalize: float = 1.0,
            rescale: str = None,
            window: str = 'triang',  # 'overlap-tile'
            probability_map: bool = False
        ):
    """
    Sliding window using tiler.
    """

    tile_channels = xraster.shape[-1]  # model.layers[0].input_shape[0][-1]
    print(f'Standardizing: {standardization}')
    # n_classes = out of the output layer, output_shape

    tiler_image = Tiler(
        data_shape=xraster.shape,
        tile_shape=(img_size, img_size, tile_channels),
        channel_dimension=-1,
        overlap=overlap,
        mode=pad_style,
        constant_value=constant_value
    )

    # Define the tiler and merger based on the output size of the prediction
    tiler_mask = Tiler(
        data_shape=(xraster.shape[0], xraster.shape[1], n_classes),
        tile_shape=(img_size, img_size, n_classes),
        channel_dimension=-1,
        overlap=overlap,
        mode=pad_style,
        constant_value=constant_value
    )

    merger = Merger(tiler=tiler_mask, window=window)

    # Iterate over the data in batches
    for batch_id, batch_i in tiler_image(xraster, batch_size=batch_size):

        # Standardize
        batch = batch_i.copy()

        if standardization is not None:
            for item in range(batch.shape[0]):
                batch[item, :, :, :] = standardize_image(
                    batch[item, :, :, :], standardization, mean, std)

        input_batch = batch.astype('float32')
        input_batch_tensor = torch.from_numpy(input_batch).cuda()
        input_batch_tensor = input_batch_tensor.transpose(-1, 1)

        with torch.no_grad():
            y_batch = model(input_batch_tensor)
            print(y_batch.shape)
        y_batch = y_batch.transpose(1, -1).cpu().numpy()
        merger.add_batch(batch_id, batch_size, y_batch)

    prediction = merger.merge(unpad=True)

    prediction = np.squeeze(prediction)
    return prediction
