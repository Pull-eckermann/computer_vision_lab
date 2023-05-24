import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def compute_color_histogram(image):
    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Split the HSV image into individual channels
    h, s, v = cv2.split(hsv_image)
    
    # Compute histograms for each channel
    hist_h = cv2.calcHist([h], [0], None, [256], [0, 256])
    hist_s = cv2.calcHist([s], [0], None, [256], [0, 256])
    hist_v = cv2.calcHist([v], [0], None, [256], [0, 256])
    
    # Concatenate the histograms into a single feature vector
    hist = np.concatenate((hist_h, hist_s, hist_v), axis=0)
    
    # Normalize the histogram
    hist = cv2.normalize(hist, hist)
    
    return hist.flatten()

def compare_histograms(hist1, hist2):
    # Compute the intersection of two histograms
    intersection = cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)
    
    return intersection

def search_by_color(query_image, database_images, top_k=5):
    # Compute the histogram of the query image
    query_hist = compute_color_histogram(query_image)
    
    results = []
    
    for image_path in database_images:
        # Load the database image
        image = cv2.imread(image_path)
        
        # Compute the histogram of the database image
        image_hist = compute_color_histogram(image)
        
        # Compare the histograms
        similarity = compare_histograms(query_hist, image_hist)
        
        # Store the similarity score and image path
        results.append((similarity, image_path))
    
    # Sort the results based on similarity
    results.sort(reverse=True)
    
    # Return the top-k matching images
    return results[:top_k]

# diretório com as imagens
img_dir = './imagens/'

# lista de imagens
database_images_paths = []

# loop pelos arquivos no diretório
for filename in os.listdir(img_dir):
    database_images_paths.append(os.path.join(img_dir, filename))

# Example usage
query_image_path = 'proc.jpg'

query_image = cv2.imread(query_image_path)

results = search_by_color(query_image, database_images_paths)

fig, axs = plt.subplots(4, 3, figsize=(12, 8))
i = 1
for similarity, image_path in results:
    row = i // 3
    col = i % 3
    i = i + 1
    axs[row, col].imshow(cv2.imread(image_path))
    axs[row, col].set_title(f"Similarity: {similarity:.2f}")
    axs[row, col].axis('off')

plt.show()
