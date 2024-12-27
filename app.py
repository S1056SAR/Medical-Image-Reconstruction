from flask import Flask, render_template, request, jsonify, send_file
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

# Load the trained generator model
model = tf.keras.models.load_model('models/srgan_generator_full.h5')
print("Model input shape:", model.input_shape)
print("Model output shape:", model.output_shape)
def process_image(image):
    # Add this in process_image function
    print("Image mode:", image.mode)
    print("Original image size:", image.size)
    try:
        # Ensure image is in RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to low resolution (64x64)
        img = image.resize((64, 64), Image.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(img, dtype=np.float32)
        
        # Normalize to [-1, 1] range
        img_array = (img_array / 127.5) - 1.0
        
        # Ensure shape is correct (1, 64, 64, 3)
        if len(img_array.shape) == 2:
            # If grayscale, convert to RGB
            img_array = np.stack((img_array,)*3, axis=-1)
        
        # Add batch dimension if not present
        if len(img_array.shape) == 3:
            img_array = np.expand_dims(img_array, axis=0)
            
        print(f"Input shape before prediction: {img_array.shape}")  # Debug print
        
        # Ensure the shape is exactly what we want
        assert img_array.shape == (1, 64, 64, 3), f"Unexpected shape: {img_array.shape}"
        
        # Generate high-resolution image
        generated = model.predict(img_array)
        
        # Convert back to PIL Image
        generated = ((generated + 1) * 127.5).astype(np.uint8)
        generated_image = Image.fromarray(generated[0])
        
        return generated_image, None
        
    except Exception as e:
        print(f"Error in process_image: {str(e)}")  # Debug print
        return None, str(e)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/enhance', methods=['POST'])
def enhance():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        if not file:
            return jsonify({'error': 'Empty file'}), 400
        
        # Open and process the image
        image = Image.open(file.stream)
        enhanced_image, error = process_image(image)
        
        if error:
            return jsonify({'error': f'Processing error: {error}'}), 500
        
        if enhanced_image is None:
            return jsonify({'error': 'Failed to enhance image'}), 500
        
        # Save to buffer
        buffer = io.BytesIO()
        enhanced_image.save(buffer, format='PNG')
        buffer.seek(0)
        
        return send_file(
            buffer,
            mimetype='image/png',
            as_attachment=True,
            download_name='enhanced_image.png'
        )
        
    except Exception as e:
        print(f"Error in enhance endpoint: {str(e)}")  # Debug print
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)