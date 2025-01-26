import rasterio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def load_band(file_path):
    with rasterio.open(f'./data/{file_path}') as src:
        return src.read(1)


def normalize_band(band):
    return band / np.max(band)


def save_image(image_data, image_name):
    # save the image
    plt.imsave(f'./{image_name}.png', (image_data * 255).astype(np.uint8))
    print(f'Image {image_name} saved.')


def save_plot(image_name):
    # save the entire plot
    plt.savefig(f'./{image_name}.png', dpi=300, bbox_inches='tight')
    print(f"Displayed plot saved as '{image_name}.png'")


def display_save_ndvi(ndvi_season, title, image_name):
    # display NDVI with a color scale and save
    plt.imshow(ndvi_season, cmap='RdYlGn')
    plt.title(title)
    plt.colorbar(label="NDVI Value")  # Add a color scale to the image
    plt.axis('off')
    save_plot(image_name)
    plt.show()


def display_save_histogram(ndvi_season, title, image_name):
    # flatten NDVI array for histogram
    ndvi_values = ndvi_season.flatten()

    plt.figure(figsize=(10, 6))
    plt.hist(ndvi_values, bins=100, color='blue', alpha=0.7, edgecolor='black', zorder=3)
    plt.title(title)
    plt.xlabel('NDVI Value')
    plt.ylabel('Frequency [pixels]')
    plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    save_plot(image_name)
    plt.show()


def display_save_distribution(ndvi_season, title, image_name):
    # flatten NDVI array
    ndvi_values = ndvi_season.flatten()

    # sort NDVI values
    sorted_ndvi = np.sort(ndvi_values)

    # calculate percentiles
    percentiles = np.linspace(0, 100, 100)  # 100 evenly spaced percentiles
    thresholds = np.percentile(sorted_ndvi, percentiles)  # NDVI values corresponding to each percentile

    # plot
    plt.figure(figsize=(10, 6))
    plt.bar(percentiles, thresholds, color='red', alpha=0.7, edgecolor='black', zorder=3)
    plt.title(title)
    plt.xlabel("Percentile [%]")
    plt.ylabel("Value Threshold")
    plt.grid(linestyle='--', alpha=0.7, zorder=0)
    save_plot(image_name)
    plt.show()


def calculate_statistics(ndvi_season, season):
    # flatten the NDVI image array to 1D
    ndvi_values = ndvi_season.flatten()

    mean_ndvi = np.mean(ndvi_values)
    median_ndvi = np.median(ndvi_values)
    std_ndvi = np.std(ndvi_values)
    max_deviation_from_mean = np.max(np.abs(ndvi_values - mean_ndvi))
    max_deviation_from_median = np.max(np.abs(ndvi_values - median_ndvi))
    max_error_using_std = np.max(
        np.abs(ndvi_values - mean_ndvi))

    # a df to store the results
    data = {
        "Statistic": ["Mean", "Median", "Standard Deviation", "Max Deviation from Mean",
                      "Max Deviation from Median", "Max Error"],
        "Value": [mean_ndvi, median_ndvi, std_ndvi, max_deviation_from_mean, max_deviation_from_median,
                  max_error_using_std]
    }

    df = pd.DataFrame(data)

    # display the table
    print(f'Statistics for NDVI – {season}')
    print(df)


# BANDS for 12 April 2020 – spring

blue_band_spring = normalize_band(load_band('April_B2.tif'))
green_band_spring = normalize_band(load_band('April_B3.tif'))
red_band_spring = normalize_band(load_band('April_B4.tif'))
nir_band_spring = normalize_band(load_band('April_B8.tif'))


# compose RGB image
rgb_spring = np.dstack((red_band_spring, green_band_spring, blue_band_spring))

save_image(rgb_spring, "rgb_composite_spring")


# NDVI calculation
ndvi_spring = (nir_band_spring - red_band_spring) / (nir_band_spring + red_band_spring)

# save NDVI image
save_image(ndvi_spring, "ndvi_spring")

# display NDVI with a color scale and save plot
display_save_ndvi(ndvi_spring,'NDVI – Spring', 'ndvi_spring_full_plot')

# plot the histogram and save
display_save_histogram(ndvi_spring, 'NDVI Histogram – Spring', 'ndvi_histogram_spring')


display_save_distribution(ndvi_spring, 'NDVI Distribution – Spring', 'ndvi_distribution_spring')

calculate_statistics(ndvi_spring, 'Spring')

# BANDS for 15 August 2020 – summer

blue_band_summer = normalize_band(load_band('August_B2.tif'))
green_band_summer = normalize_band(load_band('August_B3.tif'))
red_band_summer = normalize_band(load_band('August_B4.tif'))
nir_band_summer = normalize_band(load_band('August_B8.tif'))


# compose RGB image
rgb_summer = np.dstack((red_band_summer, green_band_summer, blue_band_summer))

save_image(rgb_summer, "rgb_composite_summer")


# NDVI calculation
ndvi_summer = (nir_band_summer - red_band_summer) / (nir_band_summer + red_band_summer)

# save NDVI image
save_image(ndvi_summer, "ndvi_summer")

# display NDVI with a color scale and save plot
display_save_ndvi(ndvi_summer,'NDVI – Summer', 'ndvi_summer_full_plot')

# plot the histogram and save
display_save_histogram(ndvi_summer, 'NDVI Histogram – Summer', 'ndvi_histogram_summer')

display_save_distribution(ndvi_summer, 'NDVI Distribution – Summer', 'ndvi_distribution_summer')

calculate_statistics(ndvi_summer, 'Summer')
