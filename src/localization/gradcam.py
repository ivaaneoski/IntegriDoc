import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

class GradCAM:
    def __init__(self, model, target_layer):
        """
        Initializes the GradCAM hook on a specific layer of the model.
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hook the target layer
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        # grad_output[0] is the gradient w.r.t the activation
        self.gradients = grad_output[0]
        
    def generate_heatmap(self, input_tensor, target_class=1):
        """
        Generates a Grad-CAM heatmap for a given input tensor and target class.
        input_tensor: [1, C, H, W]
        target_class: the class index to calculate gradients for
        """
        self.model.eval()
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Backward pass on target class
        target = output[0, target_class]
        target.backward()
        
        # Get gradients and activations
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        # Global average pooling on gradients to get weights
        weights = np.mean(gradients, axis=(1, 2))
        
        # Apply weights to activations
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
            
        # ReLU on CAM (we only care about positive influences)
        cam = np.maximum(cam, 0)
        
        # Normalize between 0 and 1
        cam = cam - np.min(cam)
        if np.max(cam) != 0:
            cam = cam / np.max(cam)
            
        # Resize to input image size
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        
        return cam

def overlay_heatmap(image_tensor, heatmap_np, colormap=cv2.COLORMAP_JET, alpha=0.5):
    """
    Overlays a numpy heatmap on top of the original RGB image tensor.
    image_tensor: [3, H, W] (normalized float tensor)
    heatmap_np: [H, W] (float numpy array 0.0-1.0)
    """
    # Unnormalize image tensor assuming ImageNet stats
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = image_tensor.cpu() * std + mean
    
    # Convert to HWC [0, 255] uint8 for cv2
    img = img.permute(1, 2, 0).numpy()
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    
    # Apply colormap to heatmap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_np), colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Superimpose
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)
    return Image.fromarray(superimposed_img)
