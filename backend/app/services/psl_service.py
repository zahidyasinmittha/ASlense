# app/services/psl_service.py
import torch
import torch.nn as nn
import cv2
import numpy as np
import asyncio
import base64
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import torchvision.transforms as transforms
from PIL import Image

class PSLModelService:
    """
    Service for handling PSL (Pakistani Sign Language) models
    Supports both MobileNet (mini) and EfficientNet (pro) architectures
    """
    
    def __init__(self):
        self.models_dir = Path(__file__).parent.parent.parent / "psl models"
        self.mobilenet_model = None
        self.efficientnet_model = None
        self.label_encoder = None
        self.loaded_models = {"mini": False, "pro": False}
        
        # Device setup - prioritize GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 PSL Service using device: {self.device}")
        
        # Model configurations
        self.input_size = 224
        self.num_classes = None  # Will be set from label encoder
        
        # Data preprocessing transforms (same as training)
        self.transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    def load_label_encoder(self):
        """Load the label encoder for PSL alphabet mapping"""
        try:
            encoder_path = self.models_dir / "label_encoder.pkl"
            import joblib
            self.label_encoder = joblib.load(encoder_path)
            
            # Update num_classes based on actual label encoder
            self.num_classes = len(self.label_encoder.classes_)
            print(f"✅ PSL Label encoder loaded from {encoder_path}")
            print(f"📊 Number of classes: {self.num_classes}")
            print(f"📝 Sample classes: {list(self.label_encoder.classes_[:10])}")
            return True
        except Exception as e:
            print(f"❌ Failed to load PSL label encoder: {e}")
            return False
    
    def create_mobilenet_model(self):
        """Create MobileNetV2 model for mini PSL recognition"""
        try:
            import torchvision.models as models
            
            # Load model architecture without pretrained weights for inference
            model = models.mobilenet_v2(weights=None)
            
            # Get number of classes from label encoder
            if self.label_encoder is None:
                print("❌ Label encoder must be loaded first for model creation")
                return None
            
            num_classes = len(self.label_encoder.classes_)
            
            # Modify classifier for PSL alphabet (same as training script)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
            
            return model
        except Exception as e:
            print(f"❌ Failed to create MobileNet model: {e}")
            return None
    
    def create_efficientnet_model(self):
        """Create EfficientNet-B0 model for pro PSL recognition"""
        try:
            import torchvision.models as models
            
            # Load model architecture without pretrained weights for inference
            model = models.efficientnet_b0(weights=None)
            
            # Get number of classes from label encoder
            if self.label_encoder is None:
                print("❌ Label encoder must be loaded first for model creation")
                return None
            
            num_classes = len(self.label_encoder.classes_)
            
            # Modify classifier for PSL alphabet (same as training script)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
            
            return model
        except Exception as e:
            print(f"❌ Failed to create EfficientNet model: {e}")
            return None
    
    def load_mini_model(self):
        """Load MobileNetV2 model for mini PSL recognition"""
        try:
            model_path = self.models_dir / "psl_mobilenet_v2_model.pth"
            if not model_path.exists():
                print(f"❌ PSL MobileNet model file not found: {model_path}")
                return False
            
            # Create model architecture (label encoder must be loaded first)
            self.mobilenet_model = self.create_mobilenet_model()
            if self.mobilenet_model is None:
                return False
            
            # Load trained weights (direct state dict as saved by training script)
            # Using weights_only=True for security and loading to device
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            self.mobilenet_model.load_state_dict(state_dict)
            
            # Move model to device and set to eval mode
            self.mobilenet_model.to(self.device)
            self.mobilenet_model.eval()
            self.loaded_models["mini"] = True
            print(f"✅ PSL MobileNet model loaded successfully from {model_path} on {self.device}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load PSL MobileNet model: {e}")
            self.loaded_models["mini"] = False
            return False
    
    def load_pro_model(self):
        """Load EfficientNet-B0 model for pro PSL recognition"""
        try:
            model_path = self.models_dir / "psl_efficientnet_b0_model.pth"
            if not model_path.exists():
                print(f"❌ PSL EfficientNet model file not found: {model_path}")
                return False
            
            # Create model architecture (label encoder must be loaded first)
            self.efficientnet_model = self.create_efficientnet_model()
            if self.efficientnet_model is None:
                return False
            
            # Load trained weights (direct state dict as saved by training script)
            # Using weights_only=True for security and loading to device
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            self.efficientnet_model.load_state_dict(state_dict)
            
            # Move model to device and set to eval mode
            self.efficientnet_model.to(self.device)
            self.efficientnet_model.eval()
            self.loaded_models["pro"] = True
            print(f"✅ PSL EfficientNet model loaded successfully from {model_path} on {self.device}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load PSL EfficientNet model: {e}")
            self.loaded_models["pro"] = False
            return False
    
    def initialize_models(self):
        """Initialize both PSL models and label encoder"""
        print("🔄 Initializing PSL models...")
        
        # Load label encoder first
        if not self.load_label_encoder():
            print("❌ Cannot proceed without PSL label encoder")
            return False
        
        # Load models
        mini_loaded = self.load_mini_model()
        pro_loaded = self.load_pro_model()
        
        if not mini_loaded and not pro_loaded:
            print("❌ No PSL models could be loaded")
            return False
        
        print(f"✅ PSL Service initialized - Mini: {mini_loaded}, Pro: {pro_loaded}")
        return True
    
    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess frame for PSL model inference"""
        try:
            # Convert BGR to RGB
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame
            
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)
            
            # Apply transforms
            input_tensor = self.transform(pil_image)
            
            # Add batch dimension and move to device
            input_batch = input_tensor.unsqueeze(0).to(self.device)
            
            return input_batch
            
        except Exception as e:
            print(f"❌ PSL preprocessing error: {e}")
            return None
    
    def predict_frame(self, frame: np.ndarray, model_type: str = "mini") -> List[Dict]:
        """
        Predict PSL sign from a single frame
        Returns top 4 predictions with confidence scores
        """
        try:
            # Validate model type and availability
            if model_type not in ["mini", "pro"]:
                raise ValueError("Model type must be 'mini' or 'pro'")
            
            if not self.loaded_models[model_type]:
                raise ValueError(f"PSL {model_type} model not loaded")
            
            # Select appropriate model
            model = self.mobilenet_model if model_type == "mini" else self.efficientnet_model
            
            # Preprocess frame
            input_tensor = self.preprocess_frame(frame)
            if input_tensor is None:
                raise ValueError("Failed to preprocess frame")
            
            # Inference
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
            # Get top 4 predictions
            top_probs, top_indices = torch.topk(probabilities, k=4)
            
            # Convert to predictions list
            predictions = []
            for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                # Map index to label using label encoder
                label = self.get_label_from_index(idx.item())
                
                predictions.append({
                    "letter": label,
                    "confidence": float(prob.item()),
                    "rank": i + 1
                })
            
            return predictions
            
        except Exception as e:
            print(f"❌ PSL prediction error: {e}")
            return [
                {"letter": "Error", "confidence": 0.0, "rank": 1},
                {"letter": "Failed", "confidence": 0.0, "rank": 2},
                {"letter": "Try", "confidence": 0.0, "rank": 3},
                {"letter": "Again", "confidence": 0.0, "rank": 4}
            ]
    
    def get_label_from_index(self, index: int) -> str:
        """Convert model output index to PSL letter using label encoder"""
        try:
            if self.label_encoder is None:
                return f"Index_{index}"
            
            # Use inverse_transform method from sklearn LabelEncoder
            if hasattr(self.label_encoder, 'inverse_transform'):
                label = self.label_encoder.inverse_transform([index])[0]
            elif hasattr(self.label_encoder, 'classes_'):
                if index < len(self.label_encoder.classes_):
                    label = self.label_encoder.classes_[index]
                else:
                    label = f"Unknown_{index}"
            else:
                label = f"Index_{index}"
            
            return str(label)
            
        except Exception as e:
            print(f"❌ Label mapping error for index {index}: {e}")
            return f"Error_{index}"
    
    def is_model_loaded(self, model_type: str) -> bool:
        """Check if a specific model is loaded"""
        return self.loaded_models.get(model_type, False)
    
    def get_model_status(self) -> Dict:
        """Get status of all PSL models"""
        return {
            "mini": {
                "name": "PSL MobileNet",
                "type": "mobilenet_v2",
                "loaded": self.loaded_models["mini"],
                "architecture": "MobileNetV2",
                "accuracy": "85%+",
                "speed": "fast",
                "description": "Fast PSL alphabet recognition using MobileNetV2"
            },
            "pro": {
                "name": "PSL EfficientNet", 
                "type": "efficientnet_b0",
                "loaded": self.loaded_models["pro"],
                "architecture": "EfficientNet-B0",
                "accuracy": "92%+",
                "speed": "medium",
                "description": "High accuracy PSL alphabet recognition using EfficientNet-B0"
            },
            "label_encoder": {
                "loaded": self.label_encoder is not None,
                "num_classes": self.num_classes
            }
        }

# Global PSL service instance
psl_service = PSLModelService()

def get_psl_service() -> PSLModelService:
    """Get the global PSL service instance"""
    return psl_service

def initialize_psl_service():
    """Initialize PSL service - call this at startup"""
    return psl_service.initialize_models()
