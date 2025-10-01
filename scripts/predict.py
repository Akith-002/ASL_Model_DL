"""
Simple inference script for the ASL model.

Usage:
  python scripts/predict.py --model models/model.keras --labels models/training_set_labels.txt --image path/to/image.jpg

"""
from pathlib import Path
import argparse
import numpy as np
from PIL import Image
import tensorflow as tf


def load_image(path, img_size=(200, 200)):
    img = Image.open(path).convert('RGB')
    img = img.resize(img_size)
    arr = np.array(img).astype('float32') / 255.0
    return np.expand_dims(arr, axis=0)


def load_labels(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]


def predict(model_path, labels_path, image_path):
    model = tf.keras.models.load_model(model_path)
    labels = load_labels(labels_path)
    img = load_image(image_path, img_size=(200, 200))
    preds = model.predict(img)
    idx = int(np.argmax(preds[0]))
    conf = float(preds[0][idx])
    return labels[idx] if idx < len(labels) else str(idx), conf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to keras model (.keras)')
    parser.add_argument('--labels', required=True, help='Path to labels txt file')
    parser.add_argument('--image', required=True, help='Path to input image')
    args = parser.parse_args()

    label, conf = predict(args.model, args.labels, args.image)
    print(f'Prediction: {label}  Confidence: {conf:.2%}')


if __name__ == '__main__':
    main()
