from flask import Flask, render_template, request, redirect, url_for, flash, session,send_file,jsonify
import rasterio
import numpy as np
from io import BytesIO
from werkzeug.utils import secure_filename
from PIL import Image
import os
import matplotlib.pyplot as plt
import joblib
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)
app.secret_key = '46256734634734678@'

# MongoDB connection
client = MongoClient('mongodb+srv://dinesh123:dinesh5733@environ.opvcd.mongodb.net/')
db = client['users']  # Change this to your database name
users_collection = db['users']  # Change this to your collection name

# Load the trained Random Forest model (ensure this path is correct)
rf_model_path = 'C:/Users/Dinesh Kumar M/OneDrive/Desktop/aqus/rf_model.pkl'  # Adjust the model path
rf_model = joblib.load(rf_model_path)  # Load the RF model

# Create a directory for storing output file
HISTOGRAM_DR1='static/histogram_ndvi'
HISTOGRAM_DR2='static/histogram_ndwi'
HISTOGRAM_DR3='static/histogram_nsmi'
OUTPUT_DIR1 = 'static/output_files_ndvi'
OUTPUT_DIR2 = 'static/output_files_ndwi'
OUTPUT_DIR3 = 'static/output_files_nsmi'
BASE_UPLOAD_FOLDER = 'uploads/'
NDVI_UPLOAD_FOLDER = os.path.join(BASE_UPLOAD_FOLDER, 'ndvi/')
NDWI_UPLOAD_FOLDER = os.path.join(BASE_UPLOAD_FOLDER, 'ndwi/')
NSMI_UPLOAD_FOLDER = os.path.join(BASE_UPLOAD_FOLDER, 'nsmi/')
DOWNLOAD_FOLDER = 'downloads/'
CLASSIFIED_DIR1 = 'static/classified_ndvi_files'
CLASSIFIED_DIR2 = 'static/classified_ndwi_files'
CLASSIFIED_DIR3 = 'static/classified_nsmi_files'
os.makedirs(NDVI_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(NDWI_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(NSMI_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_DIR1, exist_ok=True)
os.makedirs(OUTPUT_DIR2, exist_ok=True)
os.makedirs(OUTPUT_DIR2, exist_ok=True)
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
os.makedirs(CLASSIFIED_DIR1, exist_ok=True)
os.makedirs(CLASSIFIED_DIR2, exist_ok=True)
os.makedirs(CLASSIFIED_DIR3, exist_ok=True)
os.makedirs(HISTOGRAM_DR1, exist_ok=True)
os.makedirs(HISTOGRAM_DR2, exist_ok=True)
os.makedirs(HISTOGRAM_DR3, exist_ok=True)

# #thresholds -------------------------------------------->
# thresholds = {
#     "no_change": 0,
#     "very_low": (0, 0.1),
#     "low": (0.11, 0.2),
#     "medium": (0.21, 0.3),
#     "high": (0.31, 0.4),
#     "very_high": 0.4
# }


def calculate_histogram(image_array):
    # Flatten the array and calculate the histogram for the range (, 1)
    hist, bins = np.histogram(image_array, bins=256, range=(-1, 1))
    return hist, bins

# def save_histogram_plot(hist, bins, filename):
#     plt.figure(figsize=(8, 6))
    
#     # Create a smooth line plot for the histogram
#     plt.plot(bins[:-1], hist, color='gray', lw=1.5)
    
#     # Customize the grid, labels, and title
#     plt.title("Raster Histogram", fontsize=16, weight='bold')
#     plt.xlabel("Pixel Value", fontsize=12)
#     plt.ylabel("Frequency", fontsize=12)
#     plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
#     plt.legend(fontsize=10, loc="upper left")
    
#     # Save the histogram plot
#     histogram_path = os.path.join(HISTOGRAM_DR1, filename)
#     plt.savefig(histogram_path, dpi=300)
#     plt.close()  # Close the plot to avoid overlap
#     return histogram_path

def save_histogram_plot(hist, bins, filename, index):
    """
    Save a histogram plot dynamically based on the specified index.
    """
    # Define the folder mapping for each index
    histogram_folders = {
        'ndvi': HISTOGRAM_DR1,
        'ndwi': HISTOGRAM_DR2,
        'nsmi': HISTOGRAM_DR3
    }
    
    # Get the folder based on the index
    folder = histogram_folders.get(index)
    if not folder:
        raise ValueError(f"Invalid index specified: {index}")
    
    # Create the folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)
    
    # Generate the plot
    plt.figure(figsize=(8, 6))
    plt.plot(bins[:-1], hist, color='gray', lw=1.5)
    plt.title(f"{index.upper()} Histogram", fontsize=16, weight='bold')
    plt.xlabel("Pixel Value", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    
    # Save the plot in the correct folder
    histogram_path = os.path.join(folder, filename)
    plt.savefig(histogram_path, dpi=300)
    plt.close()  # Close the plot to avoid overlap
    return histogram_path


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['tif', 'tiff']


def classify_change(diff_image, index):
    # Prepare an output image with the same shape as the diff_image
    classified_image = np.zeros((diff_image.shape[0], diff_image.shape[1], 3), dtype=np.uint8)  # RGB image

    # Define color thresholds based on the index
    if index == "ndvi":
        very_low_change_color = [128, 128, 128]  # Grey for very low change
        low_change_color = [124, 252, 0]        # Vibrant green
        medium_change_color = [255, 255, 0]     # Yellow for medium change
        high_change_color = [255, 165, 0]       # Orange for high change
        very_high_change_color = [255, 0, 0]    # Red for very high change

        classified_dir = CLASSIFIED_DIR1
        output_filename = "classified_ndvi.png"

    elif index == "ndwi":
        very_low_change_color = [173, 216, 230]  # Light blue for very low change
        low_change_color = [135, 206, 250]       # Sky blue
        medium_change_color = [30, 144, 255]     # Dodger blue
        high_change_color = [0, 0, 255]          # Blue
        very_high_change_color = [0, 0, 139]     # Dark blue

        classified_dir = CLASSIFIED_DIR2
        output_filename = "classified_ndwi.png"

    elif index == "nsmi":
        very_low_change_color = [167, 167, 167]   # Peru (low moisture)
        low_change_color = [255, 69, 0]      # Tan
        medium_change_color = [20, 20, 0]    # Sandy brown
        high_change_color = [37, 221, 30]       # Dark orange
        very_high_change_color = [0, 0, 255]  # Saddle brown


        classified_dir = CLASSIFIED_DIR3
        output_filename = "classified_nsmi.png"

    else:
        raise ValueError(f"Index '{index}' is not supported for classification.")

    # Flatten the diff_image for prediction
    flat_image = diff_image.reshape(-1, 1)  # Reshape to (num_samples, 1) for model input

    # Predict classes using the RF model
    predicted_classes = rf_model.predict(flat_image)  # Predict classes for each pixel

    # Reshape predictions back to the image dimensions
    predicted_classes = predicted_classes.reshape(diff_image.shape)

    # Map the predicted classes to corresponding colors
    color_mapping = {
        0: [255, 255, 255],  
        1: very_low_change_color,
        2: low_change_color,
        3: medium_change_color,
        4: high_change_color,
        5: very_high_change_color
    }

    for i in range(diff_image.shape[0]):
        for j in range(diff_image.shape[1]):
            classified_image[i, j] = color_mapping.get(predicted_classes[i, j], [0, 0, 0])  # Default to black for invalid class

    # Save the classified image
    classified_image_path = os.path.join(classified_dir, output_filename)
    img = Image.fromarray(classified_image)
    img.save(classified_image_path)

    return classified_image_path


#funcs for calculating individual indices --------------------------------------------> 
def calculate_ndvi(red_band, nir_band):
    red = red_band.read(1).astype(np.float32)
    nir = nir_band.read(1).astype(np.float32)
    ndvi = (nir - red) / (nir + red + 1e-6)
    ndvi = np.clip(ndvi, -1, 1)
    return ndvi

def calculate_ndwi(green_band, nir_band):
    green = green_band.read(1).astype(np.float32)
    nir = nir_band.read(1).astype(np.float32)
    ndwi = (green - nir) / (green + nir + 1e-6)
    ndwi = np.clip(ndwi, -1, 1)
    return ndwi
def calculate_nsmi(swir_band, nir_band):
    swir = swir_band.read(1).astype(np.float32)
    nir = nir_band.read(1).astype(np.float32)
    nsmi = (swir - nir) / (swir + nir + 1e-6)
    nsmi = np.clip(nsmi, -1, 1)
    return nsmi

#indices calculation-------------------------------------------------------------------->
#  the 'process_files' route to handle NDVI,NDWI,NSMI indices calculation
@app.route('/process', methods=['POST'])
def process_files():
    satellite = request.form.get('satellite')
    index = request.form.get('index')

    if not satellite or satellite not in ['sentinel', 'landsat']:
        return jsonify({'error': 'Invalid satellite! Please select Sentinel or Landsat.'}), 400

    if not index or index not in ['ndvi', 'ndwi', 'nsmi']:
        return jsonify({'error': 'Invalid index! Please select NDVI, NDWI, or NSMI.'}), 400

    # Select the correct upload folder based on the index (NDVI, NDWI, or NSMI)
    if index == 'ndvi':
        upload_folder = NDVI_UPLOAD_FOLDER
    elif index == 'ndwi':
        upload_folder = NDWI_UPLOAD_FOLDER
    else:  # NSMI case
        upload_folder = NSMI_UPLOAD_FOLDER

    band1_file = request.files.get('band1')
    band2_file = request.files.get('band2')

    if not band1_file or not band2_file or not (allowed_file(band1_file.filename) and allowed_file(band2_file.filename)):
        return jsonify({'error': 'Invalid file type! Please upload TIFF files only.'}), 400

    band1_filename = secure_filename(band1_file.filename)
    band2_filename = secure_filename(band2_file.filename)
    band1_filepath = os.path.join(upload_folder, band1_filename)
    band2_filepath = os.path.join(upload_folder, band2_filename)
    band1_file.save(band1_filepath)
    band2_file.save(band2_filepath)

    try:
        with rasterio.open(band1_filepath) as band1, rasterio.open(band2_filepath) as band2:
            if index == 'ndvi':
                result_array = calculate_ndvi(band1, band2)
                result_filename = save_raster(result_array, band1.profile, 'ndvi_result.tiff')
            elif index == 'ndwi':
                result_array = calculate_ndwi(band1, band2)
                result_filename = save_raster(result_array, band1.profile, 'ndwi_result.tiff')
            elif index == 'nsmi':
                result_array = calculate_nsmi(band1, band2)
                result_filename = save_raster(result_array, band1.profile, 'nsmi_result.tiff')

        return jsonify({'download_url': f'/download/{result_filename}'}), 200
    except Exception as e:
        return jsonify({'error': f'An error occurred: {e}'}), 500

@app.route('/histogram_ndvi')
def histogram_ndvi():
    # Render the NDVI histogram page
    return render_template('histogram_ndvi.html')


@app.route('/histogram_ndwi')
def histogram_ndwi():
    # Render the NDWI histogram page
    return render_template('histogram_ndwi.html')


@app.route('/histogram_nsmi')
def histogram_nsmi():
    # Render the NSMI histogram page
    return render_template('histogram_nsmi.html')

#saving the individual indces ------------------->
@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(DOWNLOAD_FOLDER, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return jsonify({'error': 'File not found!'}), 404

#saving the raster file----------------------------------------------->
def save_raster(result_array, profile, result_filename):
    result_filepath = os.path.join(DOWNLOAD_FOLDER, result_filename)
    with rasterio.open(
        result_filepath, 'w', driver='GTiff', height=result_array.shape[0],
        width=result_array.shape[1], count=1, dtype=result_array.dtype,
        crs=profile['crs'], transform=profile['transform']
    ) as dst:
        dst.write(result_array, 1)
    return result_filename

# function for  calculation of change detection of indices like ndvi,ndwi,nsmi

# def process_tiff(file1, file2, index):
#     with rasterio.open(file1) as src1, rasterio.open(file2) as src2:
#         ndvi1 = src1.read(1)  # Read the first TIFF file
#         ndvi2 = src2.read(1)  # Read the second TIFF file
#         profile = src1.profile

#         # Calculate the NDVI change (ndvi2 - ndvi1)
#         diff_image = ndvi2 - ndvi1

#         # Determine output paths and directories based on the index
#         if index == "ndvi":
#             output_dir = OUTPUT_DIR1
#             output_tiff_name = "change_detection_ndvi.tif"
#             output_png_name = "change_detection_ndvi.png"
#         elif index == "ndwi":
#             output_dir = OUTPUT_DIR2
#             output_tiff_name = "change_detection_ndwi.tif"
#             output_png_name = "change_detection_ndwi.png"
#         elif index == "nsmi":
#             output_dir = OUTPUT_DIR3
#             output_tiff_name = "change_detection_nsmi.tif"
#             output_png_name = "change_detection_nsmi.png"
#         else:
#             raise ValueError(f"Index '{index}' is not supported.")

#         # Write the difference to a new TIFF file
#         output_tiff_path = os.path.join(output_dir, output_tiff_name)
#         with rasterio.open(output_tiff_path, 'w', **profile) as dst:
#             dst.write(diff_image, 1)

#         # Rescale and convert Float32 to 8-bit for PNG output
#         diff_image = np.nan_to_num(diff_image, nan=0.0)
#         min_val = np.nanmin(diff_image)
#         max_val = np.nanmax(diff_image)

#         # Rescale to 0-255 range for 8-bit PNG
#         if min_val == max_val:
#             scaled_image = np.zeros_like(diff_image, dtype=np.uint8)  # Set to zero if all values are the same
#         else:
#             scaled_image = ((diff_image - min_val) / (max_val - min_val) * 255).astype(np.uint8)

#         # Convert to PNG using PIL
#         output_png_path = os.path.join(output_dir, output_png_name)
#         img = Image.fromarray(scaled_image)
#         img.save(output_png_path)

#         return output_tiff_path, output_png_path, diff_image
def process_tiff(file1, file2, index):
    with rasterio.open(file1) as src1, rasterio.open(file2) as src2:
        ndvi1 = src1.read(1)  # Read the first TIFF file
        ndvi2 = src2.read(1)  # Read the second TIFF file
        profile = src1.profile

        # Calculate the NDVI change (ndvi2 - ndvi1)
        diff_image = ndvi2 - ndvi1

        # Set NaN values to NoData for transparency in TIFF
        nodata_value = -9999  # Use a special value for NoData
        diff_image = np.where(np.isnan(diff_image), nodata_value, diff_image)
        
        # Update the profile to include the NoData value
        profile.update(dtype=rasterio.float32, nodata=nodata_value)

        # Determine output paths and directories based on the index
        if index == "ndvi":
            output_dir = OUTPUT_DIR1
            output_tiff_name = "change_detection_ndvi.tif"
            output_png_name = "change_detection_ndvi.png"
        elif index == "ndwi":
            output_dir = OUTPUT_DIR2
            output_tiff_name = "change_detection_ndwi.tif"
            output_png_name = "change_detection_ndwi.png"
        elif index == "nsmi":
            output_dir = OUTPUT_DIR3
            output_tiff_name = "change_detection_nsmi.tif"
            output_png_name = "change_detection_nsmi.png"
        else:
            raise ValueError(f"Index '{index}' is not supported.")

        # Write the difference to a new TIFF file with NoData value
        output_tiff_path = os.path.join(output_dir, output_tiff_name)
        with rasterio.open(output_tiff_path, 'w', **profile) as dst:
            dst.write(diff_image, 1)

        # Create an alpha channel for PNG transparency (255 for valid, 0 for NoData)
        alpha_channel = np.where(diff_image == nodata_value, 0, 255).astype(np.uint8)

        # Rescale diff_image to 0-255 range for 8-bit PNG
        valid_mask = diff_image != nodata_value  # Ignore NoData values
        if np.any(valid_mask):
            min_val = np.nanmin(diff_image[valid_mask])
            max_val = np.nanmax(diff_image[valid_mask])
        else:
            min_val, max_val = 0, 1  # Handle edge case with no valid data
        
        if min_val == max_val:
            scaled_image = np.zeros_like(diff_image, dtype=np.uint8)  # All values the same
        else:
            scaled_image = ((diff_image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            scaled_image[diff_image == nodata_value] = 0  # Ensure NoData is 0

        # Convert to PNG using PIL, including the alpha channel for transparency
        output_png_path = os.path.join(output_dir, output_png_name)
        img = Image.fromarray(scaled_image).convert("RGBA")  # Convert to RGBA
        img.putalpha(Image.fromarray(alpha_channel))  # Apply alpha channel
        img.save(output_png_path)

        return output_tiff_path, output_png_path, diff_image


@app.route('/') #index ------------------------------------------------->
def home():
    return render_template('index.html')

@app.route('/about')  #home page --------------------------------------------->
def about():
    return render_template('about.html')

@app.route('/dashboard') #dashboard page --------------------------------------->
def dashboard():
    if 'username' in session:  # Check if the user is logged in
        return render_template('dashboard.html')  # Render the dashboard template
    else:
        flash('You need to log in first.', 'danger')
        return redirect(url_for('login'))  # Redirect to login if not logged in

 # ndvi change detection code it contains ndvi,view_ndvi,download_ndvi routes------------------->
# @app.route('/ndvi', methods=['GET', 'POST'])
# def ndvi():
#     if request.method == 'POST':
#         file1 = request.files.get('ndvi1')
#         file2 = request.files.get('ndvi2')

#         if file1 and file2 and allowed_file(file1.filename) and allowed_file(file2.filename):
#             try:
#                 output_tiff_path, output_png_path,diff_image = process_tiff(file1.stream, file2.stream,"ndvi")
#                 classified_png_path = classify_change(diff_image,"ndvi")
#                 return render_template('ndvi.html', show_view=True, output_file=output_tiff_path, png_file=output_png_path,  classified_file=classified_png_path)
#             except Exception as e:
#                 flash(f"Error processing NDVI images: {str(e)}")
#                 return redirect(url_for('ndvi'))
#         else:
#             flash("Invalid file type. Please upload TIFF files.")
#             return redirect(url_for('ndvi'))

#     return render_template('ndvi.html')
#------------------------------------------------------------------------->
@app.route('/ndvi', methods=['GET', 'POST'])
def ndvi():
    if request.method == 'POST':
        # Retrieve uploaded files
        file1 = request.files.get('ndvi1')
        file2 = request.files.get('ndvi2')
        
        if file1 and file2 and allowed_file(file1.filename) and allowed_file(file2.filename):
            try:
                index = "ndvi"  # Specify the index type dynamically (e.g., "ndvi", "ndwi", or "nsmi")
                
                # Process the TIFF files and calculate the change map
                output_tiff_path, output_png_path, diff_image = process_tiff(file1.stream, file2.stream, index)
                
                # Convert the input images to numpy arrays
                with rasterio.open(file1.stream) as band1, rasterio.open(file2.stream) as band2:
                    band1_array = band1.read(1)
                    band2_array = band2.read(1)
                
                # Calculate histograms for the input images
                hist_band1, bins_band1 = calculate_histogram(band1_array)
                hist_band2, bins_band2 = calculate_histogram(band2_array)
                
                # Save histograms as images based on the index
                hist_band1_path = save_histogram_plot(hist_band1, bins_band1, 'histogram_band1.png', index)
                hist_band2_path = save_histogram_plot(hist_band2, bins_band2, 'histogram_band2.png', index)
                
                # Calculate and save histogram for the difference image
                hist_diff, bins_diff = calculate_histogram(diff_image)
                hist_diff_path = save_histogram_plot(hist_diff, bins_diff, 'histogram_diff.png', index)
                
                # Classify the change image based on the index
                classified_png_path = classify_change(diff_image, index)

                # Render the template with the calculated paths
                return render_template(
                    'ndvi.html', 
                    show_view=True, 
                    output_file=output_tiff_path,
                    png_file=output_png_path, 
                    classified_file=classified_png_path,
                    hist_band1=hist_band1_path, 
                    hist_band2=hist_band2_path, 
                    hist_diff=hist_diff_path
                )
            except Exception as e:
                flash(f"Error processing {index.upper()} images: {str(e)}")
                return redirect(url_for('ndvi'))
        else:
            flash("Invalid file type. Please upload valid TIFF files.")
            return redirect(url_for('ndvi'))
    
    # Render the NDVI form page for GET request
    return render_template('ndvi.html')

@app.route('/view_ndvi')
def view_ndvi():
    return render_template('view.html', title='View NDVI', image_url='output_files_ndvi/change_detection_ndvi.png')

@app.route('/download_ndvi')
def download_ndvi():
    return send_file(os.path.join(OUTPUT_DIR1, 'change_detection_ndvi.tif'), as_attachment=True)


 # ndwi change detection code it contains ndwi,view_ndwi,download_ndwi routes------------------->
# @app.route('/ndwi', methods=['GET', 'POST'])
# def ndwi():
#     if request.method == 'POST':
#         file1 = request.files.get('ndwi1')
#         file2 = request.files.get('ndwi2')

#         if file1 and file2 and allowed_file(file1.filename) and allowed_file(file2.filename):
#             try:
#                 output_tiff_path, output_png_path,diff_image = process_tiff(file1.stream, file2.stream,"ndwi")
#                 classified_png_path = classify_change(diff_image,"ndwi")
#                 return render_template('ndwi.html', show_view=True, output_file=output_tiff_path, png_file=output_png_path,classified_file=classified_png_path)
#             except Exception as e:
#                 flash(f"Error processing NDWI images: {str(e)}")
#                 return redirect(url_for('ndwi'))
#         else:
#             flash("Invalid file type. Please upload TIFF files.")
#             return redirect(url_for('ndwi'))

#     return render_template('ndwi.html')
@app.route('/ndwi', methods=['GET', 'POST'])
def ndwi():
    if request.method == 'POST':
        # Retrieve uploaded files
        file1 = request.files.get('ndwi1')
        file2 = request.files.get('ndwi2')

        if file1 and file2 and allowed_file(file1.filename) and allowed_file(file2.filename):
            try:
                index = "ndwi"  # Specify the index type dynamically

                # Process the TIFF files and calculate the change map
                output_tiff_path, output_png_path, diff_image = process_tiff(file1.stream, file2.stream, index)
                
                # Convert the input images to numpy arrays
                with rasterio.open(file1.stream) as band1, rasterio.open(file2.stream) as band2:
                    band1_array = band1.read(1)
                    band2_array = band2.read(1)
                
                # Calculate histograms for the input images
                hist_band1, bins_band1 = calculate_histogram(band1_array)
                hist_band2, bins_band2 = calculate_histogram(band2_array)
                
                # Save histograms as images based on the index
                hist_band1_path = save_histogram_plot(hist_band1, bins_band1, 'histogram_band1.png', index)
                hist_band2_path = save_histogram_plot(hist_band2, bins_band2, 'histogram_band2.png', index)
                
                # Calculate and save histogram for the difference image
                hist_diff, bins_diff = calculate_histogram(diff_image)
                hist_diff_path = save_histogram_plot(hist_diff, bins_diff, 'histogram_diff.png', index)
                
                # Classify the change image based on the index
                classified_png_path = classify_change(diff_image, index)

                # Render the template with the calculated paths
                return render_template(
                    'ndwi.html', 
                    show_view=True, 
                    output_file=output_tiff_path,
                    png_file=output_png_path, 
                    classified_file=classified_png_path,
                    hist_band1=hist_band1_path, 
                    hist_band2=hist_band2_path, 
                    hist_diff=hist_diff_path
                )
            except Exception as e:
                flash(f"Error processing {index.upper()} images: {str(e)}")
                return redirect(url_for('ndwi'))
        else:
            flash("Invalid file type. Please upload valid TIFF files.")
            return redirect(url_for('ndwi'))
    
    # Render the NDWI form page for GET request
    return render_template('ndwi.html')


@app.route('/view_ndwi')
def view_ndwi():
    return render_template('view.html', title='View NDWI', image_url='output_files_ndwi/change_detection_ndwi.png')

@app.route('/download_ndwi')
def download_ndwi():
    return send_file(os.path.join(OUTPUT_DIR2, 'change_detection_ndwi.tif'), as_attachment=True)

 # nsmi change detection code it contains nsmi,view_nsmi,download_nsmi routes------------------->
# @app.route('/nsmi', methods=['GET', 'POST'])
# def nsmi():
#     if request.method == 'POST':
#         file1 = request.files.get('nsmi1')
#         file2 = request.files.get('nsmi2')

#         if file1 and file2 and allowed_file(file1.filename) and allowed_file(file2.filename):
#             try:
#                 output_tiff_path, output_png_path,diff_image = process_tiff(file1.stream, file2.stream,"nsmi")
#                 classified_png_path = classify_change(diff_image,"nsmi")
#                 return render_template('nsmi.html', show_view=True, output_file=output_tiff_path, png_file=output_png_path,classified_file=classified_png_path)
#             except Exception as e:
#                 flash(f"Error processing NSMI images: {str(e)}")
#                 return redirect(url_for('nsmi'))
#         else:
#             flash("Invalid file type. Please upload TIFF files.")
#             return redirect(url_for('nsmi'))

#     return render_template('nsmi.html')
@app.route('/nsmi', methods=['GET', 'POST'])
def nsmi():
    if request.method == 'POST':
        # Retrieve uploaded files
        file1 = request.files.get('nsmi1')
        file2 = request.files.get('nsmi2')

        if file1 and file2 and allowed_file(file1.filename) and allowed_file(file2.filename):
            try:
                index = "nsmi"  # Specify the index type dynamically

                # Process the TIFF files and calculate the change map
                output_tiff_path, output_png_path, diff_image = process_tiff(file1.stream, file2.stream, index)
                
                # Convert the input images to numpy arrays
                with rasterio.open(file1.stream) as band1, rasterio.open(file2.stream) as band2:
                    band1_array = band1.read(1)
                    band2_array = band2.read(1)
                
                # Calculate histograms for the input images
                hist_band1, bins_band1 = calculate_histogram(band1_array)
                hist_band2, bins_band2 = calculate_histogram(band2_array)
                
                # Save histograms as images based on the index
                hist_band1_path = save_histogram_plot(hist_band1, bins_band1, 'histogram_band1.png', index)
                hist_band2_path = save_histogram_plot(hist_band2, bins_band2, 'histogram_band2.png', index)
                
                # Calculate and save histogram for the difference image
                hist_diff, bins_diff = calculate_histogram(diff_image)
                hist_diff_path = save_histogram_plot(hist_diff, bins_diff, 'histogram_diff.png', index)
                
                # Classify the change image based on the index
                classified_png_path = classify_change(diff_image, index)

                # Render the template with the calculated paths
                return render_template(
                    'nsmi.html', 
                    show_view=True, 
                    output_file=output_tiff_path,
                    png_file=output_png_path, 
                    classified_file=classified_png_path,
                    hist_band1=hist_band1_path, 
                    hist_band2=hist_band2_path, 
                    hist_diff=hist_diff_path
                )
            except Exception as e:
                flash(f"Error processing {index.upper()} images: {str(e)}")
                return redirect(url_for('nsmi'))
        else:
            flash("Invalid file type. Please upload valid TIFF files.")
            return redirect(url_for('nsmi'))
    
    # Render the NSMI form page for GET request
    return render_template('nsmi.html')

@app.route('/view_nsmi')
def view_nsmi():
    return render_template('view.html', title='View NSMI', image_url='output_files_nsmi/change_detection_nsmi.png')

@app.route('/download_nsmi')
def download_nsmi():
    return send_file(os.path.join(OUTPUT_DIR3, 'change_detection_nsmi.tif'), as_attachment=True)

#viewing the classified images of all------------------------------>   
@app.route('/view_classified_ndvi')
def view_classified_ndvi():
    """Display the classified PNG image."""
    return render_template('view_classified_ndvi.html', title='Classified NDVI', image_url='classified_ndvi_files/classified_ndvi.png')
@app.route('/view_classified_ndwi')
def view_classified_ndwi():
    """Display the classified PNG image."""
    return render_template('view_classified_ndwi.html', title='Classified NDWI', image_url='classified_ndwi_files/classified_ndwi.png')
@app.route('/view_classified_nsmi')
def view_classified_nsmi():
    """Display the classified PNG image."""
    return render_template('view_classified_nsmi.html', title='Classified NSMI', image_url='classified_nsmi_files/classified_nsmi.png')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = users_collection.find_one({'username': username})
        if user and check_password_hash(user['password'], password):
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid credentials. Please try again.', 'danger')

    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        
        # Hash the password for security
        hashed_password = generate_password_hash(password)

        # Store user credentials in MongoDB
        users_collection.insert_one({
            'username': username,
            'password': hashed_password,
            'email':email
        })

        flash('Signup successful! You can now log in.', 'success')
        return redirect(url_for('login'))

    return render_template('signup.html')
@app.route('/logout')
def logout():
    session.pop('username', None)  # Remove the user from the session
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))  # Redirect to the login page

if __name__ == '__main__':
    app.run(debug=True)
