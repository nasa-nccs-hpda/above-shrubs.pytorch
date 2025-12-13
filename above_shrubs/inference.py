import os
import torch
import logging
import numpy as np
from tqdm import tqdm
from tiler import Tiler, Merger
import rasterio
from above_shrubs.processing import standardize_image
from osgeo import gdal
import rioxarray as rxr

def sliding_window_tiler_multiclass(
    xraster,
    model,
    n_classes: int,
    img_size: int,
    pad_style: str = "reflect",
    overlap: float = 0.50,
    constant_value: int = 600,
    batch_size: int = 1024,
    threshold: float = 0.50,
    standardization: str = None,
    mean=None,
    std=None,
    normalize: float = 1.0,
    rescale: str = None,
    window: str = "triang",  # or 'overlap-tile'
    probability_map: bool = False,
):
    """
    Sliding window inference for multiclass segmentation, compatible with DeepLabV3 and SegFormer.
    """

    tile_channels = xraster.shape[-1]
    print(f"Standardizing: {standardization}")

    # Tiler for the input images
    tiler_image = Tiler(
        data_shape=xraster.shape,
        tile_shape=(img_size, img_size, tile_channels),
        channel_dimension=-1,
        overlap=overlap,
        mode=pad_style,
        constant_value=constant_value,
    )

    # Tiler for the output predictions
    tiler_mask = Tiler(
        data_shape=(xraster.shape[0], xraster.shape[1], n_classes),
        tile_shape=(img_size, img_size, n_classes),
        channel_dimension=-1,
        overlap=overlap,
        mode=pad_style,
        constant_value=constant_value,
    )

    merger = Merger(tiler=tiler_mask, window=window)

    for batch_id, batch_i in tiler_image(xraster, batch_size=batch_size):
        # Standardize each tile if requested
        batch = batch_i.copy()

        if standardization is not None:
            for item in range(batch.shape[0]):
                batch[item, :, :, :] = standardize_image(
                    batch[item, :, :, :],
                    standardization,
                    mean,
                    std
                )

        # Convert to tensor and move channel dimension
        input_batch = batch.astype("float32")
        input_batch_tensor = torch.from_numpy(input_batch).cuda()
        input_batch_tensor = input_batch_tensor.permute(0, 3, 1, 2)  # NCHW

        with torch.no_grad():
            output = model(input_batch_tensor)

        # Handle different output types
        if hasattr(output, "logits"):
            logits = output.logits
        else:
            logits = output

        # Resize if needed (SegFormer)
        if logits.shape[-2:] != input_batch_tensor.shape[-2:]:
            logits = torch.nn.functional.interpolate(
                logits,
                size=(img_size, img_size),
                mode="bilinear",
                align_corners=False
            )

        # Convert to numpy
        y_batch = logits.permute(0, 2, 3, 1).cpu().numpy()

        merger.add_batch(batch_id, batch_size, y_batch)

    prediction = merger.merge(unpad=True)
    prediction = np.squeeze(prediction)
    return prediction

# Convert tif to decameters and make 0 min
def convert_predictions_to_decameters_int16(predictions):
    """
    Convert predictions from meters to decameters, round to nearest integer, 
    and convert to int16 data type. Preserves -9999 NoData values.
    
    Parameters:
    predictions: numpy array with height predictions in meters (may contain -9999)
    
    Returns:
    converted_predictions: numpy array of int16 type with values in decameters
    """
    # Create NoData mask
    nodata_mask = predictions == -9999
    
    # Work with valid data only
    valid_predictions = predictions.copy()
    valid_predictions[nodata_mask] = 0  # Temporarily set to 0 for processing
    
    # Multiply by 10 to convert meters to decameters
    decameters = valid_predictions * 10
    
    # Round to nearest integer
    rounded = np.round(decameters)
    
    # Convert to int16
    converted_predictions = rounded.astype(np.int16)

    # Make all negative vlaues = 0
    converted_predictions[converted_predictions < 0] = 0 
    
    # Restore NoData values
    converted_predictions[nodata_mask] = -9999
    
    print(f"Converted to decameters. NoData pixels preserved: {np.sum(nodata_mask):,}")
    
    return converted_predictions

#Compress and Save Tif
def save_predictions_as_geotiff(predictions_decameters, reference_tif_path, output_path):
    """
    Save predictions as a GeoTIFF with matching geotransform and projection from reference,
    with LZW compression, NoData value of -9999, and metadata about units.
    
    Parameters:
    predictions_decameters: numpy array of int16 predictions in decameters
    reference_tif_path: path to the original .tif file for geospatial reference
    output_path: path where to save the output GeoTIFF
    """
    # Open the reference dataset to get geospatial information
    ref_ds = gdal.Open(reference_tif_path)
    if ref_ds is None:
        raise ValueError(f"Could not open reference file: {reference_tif_path}")
    
    # Get geospatial information from reference
    geotransform = ref_ds.GetGeoTransform()
    projection = ref_ds.GetProjection()
    
    # Get dimensions
    height, width = predictions_decameters.shape
    
    # Replace NaN values with -9999 (NoData value)
    output_array = predictions_decameters.copy()
    if np.issubdtype(predictions_decameters.dtype, np.floating):
        output_array = np.where(np.isnan(predictions_decameters), -9999, predictions_decameters)
    output_array = output_array.astype(np.int16)
    
    # Create the output dataset
    driver = gdal.GetDriverByName('GTiff')
    
    # Create dataset with LZW compression
    out_ds = driver.Create(
        output_path, 
        width, 
        height, 
        1,  # number of bands
        gdal.GDT_Int16,  # data type
        options=['COMPRESS=LZW', 'TILED=YES']  # LZW compression + tiling for efficiency
    )
    
    if out_ds is None:
        raise ValueError(f"Could not create output file: {output_path}")
    
    # Set geospatial information
    out_ds.SetGeoTransform(geotransform)
    out_ds.SetProjection(projection)
    
    # Get the band and write data
    band = out_ds.GetRasterBand(1)
    band.WriteArray(output_array)
    
    # Set NoData value
    band.SetNoDataValue(-9999)
    
    # Add metadata about units
    band.SetDescription("Canopy Height Model - Values in decameters")
    band.SetMetadataItem("UNITS", "decameters")
    band.SetMetadataItem("DESCRIPTION", "Predicted canopy heights in decameters (1 meter = 10 decameters)")
    
    # Add dataset-level metadata
    out_ds.SetMetadataItem("PROCESSING", "DINOv3 model prediction")
    out_ds.SetMetadataItem("NODATA_VALUE", "-9999")
    out_ds.SetMetadataItem("UNITS", "decameters")
    
    # Flush and close
    band.FlushCache()
    out_ds.FlushCache()
    band = None
    out_ds = None
    ref_ds = None
    
    print(f"Successfully saved predictions to: {output_path}")
    print(f"- Compression: LZW")
    print(f"- NoData value: -9999") 
    print(f"- Units: decameters")
    print(f"- Data type: int16")
    print(f"- Dimensions: {height} x {width}")


def complete_dinov3_inference_pipeline_batch(tif_list, model, data_config, inference_config, conf):
    
    # Extract configuration values
    data_name = data_config['data_name']
    means = data_config['means']
    stds = data_config['stds']
    
    nir_min = inference_config['nir_min']
    nir_max = inference_config['nir_max']
    red_min = inference_config['red_min']
    red_max = inference_config['red_max']
    green_min = inference_config['green_min']
    green_max = inference_config['green_max']
    img_size = inference_config['img_size']
    overlap = inference_config['overlap']
    batch_size = inference_config['batch_size']
    visualize_first = inference_config['visualize_first']
    
    print("🚀 Starting Batch DINOv3 Inference Pipeline")
    print("=" * 60)

    output_dir = conf.inference_save_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created folder: {output_dir}")
    
    print(f"Total TIF files to process: {len(tif_list)}")
    print("=" * 60)
    
    # Set up model
    model.eval()
    device = next(model.parameters()).device
    
    results = []
    output_h, output_w = None, None  # Initialize for model output dimensions
    
    for file_idx, tif_path in enumerate(tif_list, 1):

        print(f"\n🔄 Processing TIF {file_idx}/{len(tif_list)}: {os.path.basename(tif_path)}")
        print("-" * 50)

        # Generate output filename
        base_name = os.path.basename(tif_path)
        if len(base_name) >= 46:
            string = base_name[-46:-10]
        else:
            string = base_name[:-4]
        output_tif = os.path.join(output_dir, f'{string}sr-02m.chm.tif')

        # lock file for multi-node, multi-processing
        lock_filename = f'{output_tif}.lock'

        # predict only if file does not exist and no lock file
        if not os.path.isfile(output_tif) and \
                not os.path.isfile(lock_filename):

            try:
                # Step 1: Load and preprocess imagery
                print("📂 Step 1: Loading and preprocessing imagery...")
                
                with rasterio.open(tif_path) as src:
                    print(f"   Input shape: {src.shape}")
                    print(f"   Bands: {src.count}")
                    
                    # Extract NIR, Red, Green channels (assuming bands 4, 3, 2)
                    nir_band = src.read(4)   # NIR
                    red_band = src.read(3)   # Red
                    green_band = src.read(2) # Green
                    
                    # Stack into NRG array (channels first)
                    nrg_array = np.stack([nir_band, red_band, green_band], axis=0)
                
                print(f"   Extracted NRG shape: {nrg_array.shape}")
                
                # Preprocess data
                nrg_array = nrg_array.astype(np.float32)
                nrg_array[nrg_array == -9999] = np.nan
                
                # Scale to 0-1 range first (like in training)
                nrg_array[0] = (nrg_array[0] - nir_min) / (nir_max - nir_min)
                nrg_array[1] = (nrg_array[1] - red_min) / (red_max - red_min) 
                nrg_array[2] = (nrg_array[2] - green_min) / (green_max - green_min)
                
                print("nir_mean: ", np.nanmean(nrg_array[0]), " red_mean: ", np.nanmean(nrg_array[1]), " green_mean: ", np.nanmean(nrg_array[2]))            
                
                # Clip to ensure 0-1 range
                nrg_array = np.clip(nrg_array, 0, 1)
                
                # Normalize using training statistics
                nrg_array[0] = (nrg_array[0] - means[0]) / stds[0]
                nrg_array[1] = (nrg_array[1] - means[1]) / stds[1]
                nrg_array[2] = (nrg_array[2] - means[2]) / stds[2]
                
                print(f"   ✓ Preprocessing complete")
                print(f"   NaN pixels: {np.isnan(nrg_array).sum():,}")
                
                # Step 2: Set up sliding window inference
                print(f"🧠 Step 2: Running DINOv3 inference...")
                
                # Transpose to (H, W, C) for tiler
                xraster = nrg_array.transpose(1, 2, 0)
                
                # Set up image tiler
                tiler_image = Tiler(
                    data_shape=xraster.shape,
                    tile_shape=(img_size, img_size, 3),
                    channel_dimension=-1,
                    overlap=overlap,
                    mode='reflect'
                )
                
                # Get model output dimensions (only for first file)
                if file_idx == 1:
                    test_input = torch.randn(1, 3, img_size, img_size).to(device)
                    with torch.no_grad():
                        test_output = model(test_input)
                        output_h, output_w = test_output.shape[1], test_output.shape[2]
                    print(f"   Model output size: {output_h}x{output_w}")
                
                # Set up merger for combining tile predictions
                tiler_mask = Tiler(
                    data_shape=(xraster.shape[0], xraster.shape[1], 1),
                    tile_shape=(output_h, output_w, 1),
                    channel_dimension=-1,
                    overlap=overlap,
                    mode='reflect'
                )
                merger = Merger(tiler=tiler_mask, window='triang')
                
                # Process tiles in batches
                total_tiles = len(tiler_image)
                print(f"   Processing {total_tiles} tiles...")
                
                tile_count = 0
                for batch_id, batch_i in tiler_image(xraster, batch_size=batch_size):
                    actual_batch_size = len(batch_i)
                    
                    # Prepare input batch
                    input_batch = batch_i.transpose(0, 3, 1, 2).astype('float32')
                    input_batch_tensor = torch.from_numpy(input_batch).to(device)
                    input_batch_tensor = torch.nan_to_num(input_batch_tensor, nan=0.0)
                    
                    # Run inference
                    with torch.no_grad():
                        y_batch = model(input_batch_tensor)
                        y_batch_numpy = y_batch.cpu().numpy()
                        
                        # Format for merger (B, H, W, C)
                        if len(y_batch_numpy.shape) == 3:
                            formatted_output = np.expand_dims(y_batch_numpy, axis=-1)
                        else:
                            formatted_output = y_batch_numpy
                    
                    # Add predictions to merger
                    for j in range(actual_batch_size):
                        tile_id = batch_id * batch_size + j
                        merger.add(tile_id, formatted_output[j])
                    
                    tile_count += actual_batch_size
                
                # Merge tile predictions
                print("   🔄 Merging tiles...")
                predictions_raw = merger.merge(unpad=True)
                predictions_meters = np.squeeze(predictions_raw)
                
                print(f"   ✓ Inference complete")
                print(f"   Final prediction shape: {predictions_meters.shape}")
                
                # Step 3: Post-process predictions
                print(f"🔧 Step 3: Post-processing predictions...")
                
                # Restore original NoData locations
                original_nodata = np.isnan(nrg_array[0])
                predictions_meters[original_nodata] = -9999
                
                # Convert to decameters (int16 for storage efficiency)
                predictions_decameters = convert_predictions_to_decameters_int16(predictions_meters)
                print(f"   ✓ Converted to decameters (int16)")
                
                # Step 4: Save as GeoTIFF
                print(f"💾 Step 4: Saving GeoTIFF...")
                
                save_predictions_as_geotiff(predictions_decameters, tif_path, output_tif)
                
                # Add to results
                results.append(predictions_meters)
                
                # Print summary for this file
                valid_pixels = (predictions_meters != -9999).sum()
                nodata_pixels = (predictions_meters == -9999).sum()
                
                print(f"✅ Completed TIF {file_idx}/{len(tif_list)}")
                print(f"   📊 Valid predictions: {valid_pixels:,} pixels")
                print(f"   🚫 NoData pixels: {nodata_pixels:,} pixels")
                
                if valid_pixels > 0:
                    valid_heights = predictions_meters[predictions_meters != -9999]
                    print(f"   🌲 Height range: {valid_heights.min():.2f}m to {valid_heights.max():.2f}m")
                    print(f"   📈 Mean height: {valid_heights.mean():.2f}m ± {valid_heights.std():.2f}m")
                
                                # delete lock file
                try:
                    os.remove(lock_filename)
                except FileNotFoundError:
                    logging.info(f'Lock file not found {lock_filename}')
                    continue

            except Exception as e:
                print(f"❌ Error processing {tif_path}: {str(e)}")
                import traceback
                traceback.print_exc()
                results.append(None)
                continue
        # This is the case where the prediction was already saved
        else:
            logging.info(f'{output_tif} already predicted.')

    # Print final summary
    successful_files = sum(1 for r in results if r is not None)
    failed_files = len(results) - successful_files

    print("\n" + "="*60)
    print("⭐ BATCH INFERENCE COMPLETE ⭐")
    print("="*60)
    print(f"📁 Output directory: {output_dir}")
    print(f"✅ Successfully processed: {successful_files}/{len(tif_list)} files")
    if failed_files > 0:
        print(f"❌ Failed: {failed_files} files")
    print("="*60)

    return results
