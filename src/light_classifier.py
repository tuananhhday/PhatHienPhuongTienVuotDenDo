"""Autoware MobileNetV2 classifier, OpenCV ONNX inference on tight signal ROIs.

Source: https://huggingface.co/AutowareFoundation/traffic_light_classifier
Revision: 527c4905afd8f1bfcc03253529a105f7fa5a6d36 (Apache-2.0).
"""
from pathlib import Path
import cv2
import numpy as np


class LightClassifier:
    def __init__(self, weights, labels, confidence=0.8):
        self.net = cv2.dnn.readNetFromONNX(str(weights))
        self.labels = Path(labels).read_text(encoding='utf-8').splitlines()
        self.confidence = confidence

    def classify(self, crop, movement='straight'):
        if crop.size == 0:
            return 'UNKNOWN', 0.0, 'empty'
        image = cv2.cvtColor(cv2.resize(crop, (224, 224)), cv2.COLOR_BGR2RGB).astype(np.float32)
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        tensor = np.ascontiguousarray(((image - mean) / std).transpose(2, 0, 1)[None])
        self.net.setInput(tensor)
        probabilities = self.net.forward().reshape(-1)
        if len(probabilities) != len(self.labels):
            raise ValueError('Classifier output does not match label file')
        index = int(probabilities.argmax())
        score, label = float(probabilities[index]), self.labels[index]
        if score < self.confidence or label == 'unknown':
            return 'UNKNOWN', score, label
        if label in ('red', 'green', 'yellow'):
            return label.upper(), score, label
        # Mixed red + green arrows must be interpreted for the configured movement.
        parts = set(label.split(','))
        if 'red' in parts and parts <= {'red', 'left', 'right', 'straight'}:
            return ('GREEN' if movement in parts else 'RED'), score, label
        return 'UNKNOWN', score, label
