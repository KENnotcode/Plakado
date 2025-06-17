from ultralytics import YOLO
import os

# Folder containing the images
image_folder = r'C:\Users\kenne\Downloads\repos\Plate detection\dataset\test\images'

def run_inference(image_path):
    # Load the YOLOv8n model
    model_path = r'C:\Users\kenne\Downloads\repos\Plate detection\dataset\runs\detect\train4\weights\best.pt'
    model = YOLO(model_path)  # Use YOLOv8's new API for loading the model
    
    # Perform inference using the predict method
    results = model.predict(image_path)  # Run inference, use .predict() for YOLOv8

    # Loop through the results (if multiple images are processed)
    for result in results:
        # Show results (images with detected objects)
        result.show()  # This will display the image with bounding boxes and labels

        # Save the results to a folder
        output_path = 'runs/detect/inference_results/'
        result.save(output_path)  # Save results (bounding boxes, labels, etc.)
        print(f"Results saved to: {output_path}")
        
        # Access the detections
        detections = result.boxes.cpu().numpy()  # Convert to numpy array for easier handling
        
        # Print the detections (class, confidence, and bounding box)
        for detection in detections:
            # Each detection is a tensor, so we need to access its elements
            class_id = detection.cls.item()  # Class ID
            confidence = detection.conf.item()  # Confidence score
            bbox = detection.xyxy  # Bounding box in xyxy format (already a numpy array)
            
            print(f"Class: {class_id}, Confidence: {confidence}, Bounding Box: {bbox}")

def process_images_in_folder(image_folder):
    # Get all image files from the folder
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Run inference on each image
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        print(f"Running inference on: {image_path}")
        run_inference(image_path)

# Path to your test images folder
image_folder = r'C:\Users\kenne\Downloads\repos\Plate detection\dataset\test\images'

# Run inference on all images in the folder
process_images_in_folder(image_folder)