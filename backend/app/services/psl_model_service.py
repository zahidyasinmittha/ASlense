# app/services/psl_model_service.py
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights, efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import joblib
import numpy as np
import cv2
import base64
import io
from typing import Dict, List, Tuple, Optional
import logging
import warnings

# Suppress sklearn version warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

logger = logging.getLogger(__name__)

class PSLModelService:
    """PSL Model Service for real-time inference with multiple model support"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}  # Dictionary to store multiple models
        self.label_encoders = {}  # Dictionary to store label encoders for each model
        self.transform = None
        self.current_model = "ps_mini"  # Default model
        
        # Model configurations
        self.model_configs = {
            "ps_mini": {
                "name": "PS Mini",
                "model_path": "psl models/psl_mobilenet_v2_model.pth",
                "encoder_path": "psl models/label_encoder.pkl",
                "architecture": "mobilenet_v2",
                "accuracy": "98%",
                "speed": "Fast (400ms)",
                "processing_time": "400ms"
            },
            "ps_pro": {
                "name": "PS Pro",
                "model_path": "psl models/psl_efficientnet_b0_model.pth",
                "encoder_path": "psl models/label_encoder.pkl",  # Same encoder for both
                "architecture": "efficientnet_b0",
                "accuracy": "99%",
                "speed": "Medium (400ms)",
                "processing_time": "400ms"
            }
        }
        
        self._setup_transform()
        self._load_all_models()
    
    def _setup_transform(self):
        """Setup image preprocessing transforms"""
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _load_model(self, model_key: str):
        """Load a specific model"""
        try:
            config = self.model_configs[model_key]
            
            # Check if model files exist
            if not os.path.exists(config["model_path"]):
                logger.warning(f"Model file not found: {config['model_path']}")
                return False
                
            if not os.path.exists(config["encoder_path"]):
                logger.warning(f"Label encoder not found: {config['encoder_path']}")
                return False
            
            # Load label encoder
            label_encoder = joblib.load(config["encoder_path"])
            num_classes = len(label_encoder.classes_)
            
            # Initialize model architecture based on type
            if config["architecture"] == "mobilenet_v2":
                model = mobilenet_v2(weights=None)
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
            elif config["architecture"] == "efficientnet_b0":
                model = efficientnet_b0(weights=None)
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
            else:
                logger.error(f"Unknown architecture: {config['architecture']}")
                return False
            
            # Load trained weights
            state_dict = torch.load(config["model_path"], map_location=self.device, weights_only=True)
            model.load_state_dict(state_dict)
            model = model.to(self.device)
            model.eval()
            
            # Store model and encoder
            self.models[model_key] = model
            self.label_encoders[model_key] = label_encoder
            
            logger.info(f"✅ {config['name']} model loaded successfully with {num_classes} classes")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading {model_key} model: {str(e)}")
            return False
    
    def _load_all_models(self):
        """Load all available models"""
        loaded_count = 0
        for model_key in self.model_configs.keys():
            if self._load_model(model_key):
                loaded_count += 1
            else:
                logger.error(f"❌ Failed to load {model_key} model")
        
        logger.info(f"📦 Loaded {loaded_count}/{len(self.model_configs)} PSL models successfully")
        
        # Set default model to first successfully loaded model
        if loaded_count > 0:
            for model_key in self.model_configs.keys():
                if model_key in self.models:
                    self.current_model = model_key
                    break
    
    def set_model(self, model_key: str) -> bool:
        """Switch to a different model"""
        if model_key in self.models:
            self.current_model = model_key
            logger.info(f"Switched to {self.model_configs[model_key]['name']} model")
            return True
        else:
            logger.error(f"Model {model_key} not available")
            return False
    
    def get_available_models(self) -> Dict:
        """Get list of available models"""
        available = {}
        for key, config in self.model_configs.items():
            is_loaded = key in self.models
            available[key] = {
                "name": config["name"],
                "accuracy": config["accuracy"],
                "speed": config["speed"],
                "architecture": config["architecture"],
                "loaded": is_loaded,
                "num_classes": len(self.label_encoders[key].classes_) if is_loaded else 0
            }
        return available
    
    def preprocess_image(self, image_data: str) -> Optional[torch.Tensor]:
        """
        Preprocess base64 image data for model inference
        
        Args:
            image_data: Base64 encoded image string
            
        Returns:
            Preprocessed tensor or None if error
        """
        try:
            # Remove data URL prefix if present
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            # Apply transforms
            if self.transform:
                image_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
                return image_tensor.to(self.device)
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error preprocessing image: {str(e)}")
            return None
    
    def predict(self, image_data: str, top_k: int = 3, model_key: str = None) -> Dict:
        """
        Make prediction on image data
        
        Args:
            image_data: Base64 encoded image string
            top_k: Number of top predictions to return
            model_key: Specific model to use (optional, uses current_model if None)
            
        Returns:
            Dictionary with predictions and confidence scores
        """
        # Use specified model or current model
        model_to_use = model_key if model_key else self.current_model
        
        if model_to_use not in self.models:
            return {
                "error": f"Model {model_to_use} not loaded",
                "predictions": [],
                "confidence": 0.0,
                "model_used": model_to_use
            }

        try:
            # Get model and encoder
            model = self.models[model_to_use]
            label_encoder = self.label_encoders[model_to_use]
            config = self.model_configs[model_to_use]
            
            # Preprocess image
            image_tensor = self.preprocess_image(image_data)
            if image_tensor is None:
                return {
                    "error": "Failed to preprocess image",
                    "predictions": [],
                    "confidence": 0.0,
                    "model_used": model_to_use
                }
            
            # Make prediction
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                
                # Get top-k predictions
                top_probs, top_indices = torch.topk(probabilities, top_k)
                
                predictions = []
                for i in range(top_k):
                    class_idx = top_indices[i].item()
                    confidence = top_probs[i].item()
                    class_name = label_encoder.inverse_transform([class_idx])[0]
                    
                    predictions.append({
                        "class": class_name,
                        "confidence": float(confidence)
                    })
                
                return {
                    "predictions": predictions,
                    "confidence": float(top_probs[0].item()),
                    "processing_time": config["processing_time"],
                    "model_used": model_to_use,
                    "model_name": config["name"]
                }
                
        except Exception as e:
            logger.error(f"Error during prediction with {model_to_use}: {str(e)}")
            return {
                "error": f"Prediction failed: {str(e)}",
                "predictions": [],
                "confidence": 0.0,
                "model_used": model_to_use
            }
    
    def set_current_model(self, model_key: str) -> bool:
        """Set the current model for predictions"""
        if model_key in self.model_configs:
            self.current_model = model_key
            return True
        return False
    
    def get_model_classes(self, model_key: str = None) -> List[str]:
        """Get the list of classes/letters that the model can recognize"""
        model_to_use = model_key if model_key else self.current_model
        
        if model_to_use in self.label_encoders:
            return list(self.label_encoders[model_to_use].classes_)
        return []
    
    def test_models(self) -> Dict:
        """Test all loaded models and return their status"""
        test_results = {}
        
        for model_key, config in self.model_configs.items():
            if model_key in self.models:
                try:
                    # Create a dummy tensor for testing
                    dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
                    
                    with torch.no_grad():
                        model = self.models[model_key]
                        output = model(dummy_input)
                        
                    test_results[model_key] = {
                        "status": "✅ Working",
                        "name": config["name"],
                        "output_shape": list(output.shape),
                        "num_classes": len(self.label_encoders[model_key].classes_)
                    }
                    logger.info(f"✅ {config['name']} model test passed")
                    
                except Exception as e:
                    test_results[model_key] = {
                        "status": f"❌ Error: {str(e)}",
                        "name": config["name"]
                    }
                    logger.error(f"❌ {config['name']} model test failed: {str(e)}")
            else:
                test_results[model_key] = {
                    "status": "❌ Not loaded",
                    "name": config["name"]
                }
        
        return test_results
    
    def get_model_info(self, model_key: str = None) -> Dict:
        """Get model information"""
        model_to_use = model_key if model_key else self.current_model
        
        if model_to_use not in self.models:
            # Return info for current model config even if not loaded
            config = self.model_configs.get(model_to_use, self.model_configs["ps_mini"])
            return {
                "model_name": config["name"],
                "status": "not_loaded",
                "accuracy": config["accuracy"],
                "speed": config["speed"],
                "architecture": config["architecture"],
                "num_classes": 0,
                "device": str(self.device),
                "current_model": self.current_model,
                "available_models": list(self.model_configs.keys())
            }
        
        config = self.model_configs[model_to_use]
        return {
            "model_name": config["name"],
            "status": "loaded",
            "accuracy": config["accuracy"],
            "speed": config["speed"],
            "architecture": config["architecture"],
            "num_classes": len(self.label_encoders[model_to_use].classes_),
            "device": str(self.device),
            "current_model": self.current_model,
            "available_models": list(self.model_configs.keys())
        }

# Global model service instance
psl_model_service = PSLModelService()
