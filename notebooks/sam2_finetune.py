import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import json
import os
from tqdm import tqdm
from pynvml import *
import torch.nn.functional as F  

def print_gpu_utilization():
    """Prints current GPU memory usage"""
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory used: {info.used//1024**2}MB / {info.total//1024**2}MB")

def json_to_mask(json_path, image_shape):
    """Load JSON annotation and convert to binary mask"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for shape in data['shapes']:
        points = np.array(shape['points'], dtype=np.int32)
        cv2.fillPoly(mask, [points], 1)
    return mask

def load_multi_site_data(base_path="./train", sites=["site_a", "site_b"]):
    """
    Load training data from multiple sites
    Args:
        base_path: Root directory containing site folders
        sites: List of site folder names
    Returns:
        List of images, List of masks
    """
    images, masks = [], []
    
    for site in sites:
        site_path = os.path.join(base_path, site)
        if not os.path.exists(site_path):
            print(f"Warning: Site directory {site_path} not found")
            continue
            
        # Get all JPGs in site directory
        img_files = [f for f in os.listdir(site_path) if f.endswith('.jpg')]
        
        for img_file in img_files:
            # Construct paths
            img_path = os.path.join(site_path, img_file)
            mask_path = os.path.join(site_path, img_file.replace('.jpg', '.json'))
            
            if not os.path.exists(mask_path):
                print(f"Warning: Mask {mask_path} not found")
                continue
                
            # Load and preprocess
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            mask = json_to_mask(mask_path, img.shape)
            
            # Resize if needed (recommended 1024x1024 for SAM)
            if img.shape[0] > 1024 or img.shape[1] > 1024:
                img = cv2.resize(img, (1024, 1024))
                mask = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)
            
            images.append(img)
            masks.append(mask)
    
    print(f"Loaded {len(images)} images from {len(sites)} sites")
    return images, masks

def fine_tune_sam(sam, images, masks, epochs=4, lr=3e-5, batch_size=2):
    """
    Fine-tuning function specifically adapted for the provided ImageEncoder structure
    """
    device = sam.device
    # Estimate this from your dataset
    pos_weight = torch.tensor([10.0], device=device)  # tweak as needed
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.AdamW(sam.sam_mask_decoder.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler()
    accum_steps = 2
    
    # Get model components
    image_encoder = sam.image_encoder
    mask_decoder = sam.sam_mask_decoder
    prompt_encoder = sam.sam_prompt_encoder
    
    for epoch in range(epochs):
        epoch_loss = 0
        optimizer.zero_grad()
        
        for i in tqdm(range(len(images)), desc=f"Epoch {epoch+1}/{epochs}"):
            # Prepare inputs
            img = images[i]
            mask = masks[i]
            
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
            mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float().to(device)
            
            # Generate embeddings
            with torch.no_grad():
                encoder_output = image_encoder(img_tensor)

                # Extract features from the encoder output
                features = encoder_output['backbone_fpn']
                vision_features = encoder_output['vision_features']
                vision_pos_enc = encoder_output['vision_pos_enc']

                # Print shapes for debug
                print(f"Main features shape: {vision_features.shape}")
                print(f"FPN features shapes: {[f.shape for f in features]}")
                print(f"Position encoding shapes: {[p.shape for p in vision_pos_enc]}")

                # Get positional encoding for prompt encoder
                image_pe = prompt_encoder.get_dense_pe()

                mask_input = F.interpolate(
                    mask_tensor,
                    size=prompt_encoder.mask_input_size,
                    mode="bilinear",
                    align_corners=False
                )

                cv2.imwrite("prompt_mask_input.jpg", (mask_input.squeeze().cpu().numpy() * 255).astype(np.uint8))

                sparse_embeddings, dense_embeddings = prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=mask_input
                )
            
            # Forward pass with proper arguments
            with torch.cuda.amp.autocast():
                outputs = mask_decoder(
                    image_embeddings=vision_features,
                    image_pe=image_pe,
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                    repeat_image=False,
                    high_res_features=[
                        mask_decoder.conv_s0(features[0]),
                        mask_decoder.conv_s1(features[1])
                    ]

                )[0]  # Take first output

                # Resize GT mask to match output resolution (use nearest for binary mask)
                target_mask = F.interpolate(
                    mask_tensor,
                    size=outputs.shape[-2:],  # usually (256, 256)
                    mode="bilinear",
                    align_corners=False
                )

                loss = criterion(outputs, target_mask) / accum_steps

            # Backward pass and optimization
            scaler.scale(loss).backward()
            epoch_loss += loss.item() * accum_steps

            if torch.cuda.memory_allocated() > 8e9:
                torch.cuda.empty_cache()
                batch_size = max(1, batch_size - 1)
            
            if (i+1) % accum_steps == 0 or i == len(images)-1:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                if i % 2 == 0:
                    print_gpu_utilization()
        
        print(f"Epoch {epoch+1} Loss: {epoch_loss/len(images):.4f}")
    
    return sam

def predict_shoreline(model, image):
    """Prediction function adapted for the specific ImageEncoder output"""
    if isinstance(image, str):
        image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
    
    img_tensor = torch.from_numpy(image).permute(2,0,1).unsqueeze(0).float().cuda() / 255.0
    
    with torch.no_grad(), torch.cuda.amp.autocast():
        # Get features from encoder
        encoder_output = model.image_encoder(img_tensor)
        
        # Extract features
        features = encoder_output['backbone_fpn']
        vision_features = encoder_output['vision_features']

        # Apply SAM2-specific transforms for high-res features
        feat_s0 = sam.sam_mask_decoder.conv_s0(features[0])
        feat_s1 = sam.sam_mask_decoder.conv_s1(features[1])
        high_res_features = [feat_s0, feat_s1]

        # Get other components
        image_pe = model.sam_prompt_encoder.get_dense_pe()
        sparse_embeddings, dense_embeddings = model.sam_prompt_encoder(
            points=None,
            boxes=None,
            masks=None
        )
        
        # Forward pass
        outputs = model.sam_mask_decoder(
            image_embeddings=vision_features,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features  # must be 64-channel tensors
        )[0]
        
        print(f"Mask values - Min: {outputs.min().item():.3f}, Max: {outputs.max().item():.3f}, Mean: {outputs.mean().item():.3f}")

        pred_mask = (torch.sigmoid(outputs) > 0.5).cpu().numpy()[0,0]
    
    return pred_mask

if __name__ == "__main__":
    # Initialize SAM2
    from sam2.build_sam import build_sam2_object_tracker
    
    print("Initializing SAM2...")
    sam = build_sam2_object_tracker(
        num_objects=1,
        config_file="./configs/samurai/sam2.1_hiera_b+.yaml",
        ckpt_path="../checkpoints/sam2.1_hiera_base_plus.pt",
        device="cuda:0"
    )
    
    # Load multi-site data
    print("\nLoading training data...")
    train_images, train_masks = load_multi_site_data(
        base_path="./train",
        sites=["site_a", "site_b"]  # Add your site folders here
    )

    # Add this check after loading data
    img, mask = train_images[0], train_masks[0]
    print(f"Image shape: {img.shape}, Mask shape: {mask.shape}")
    print(f"Mask unique values: {np.unique(mask)}")  # Should be [0, 1]

    # Visualize a training sample
    cv2.imshow("Training Sample", np.hstack([
        img,
        cv2.cvtColor(mask*255, cv2.COLOR_GRAY2RGB)
    ]))
    cv2.waitKey(3000)
    cv2.destroyAllWindows()
    
    # Verify GPU
    print("\nPre-training GPU status:")
    print_gpu_utilization()
    
    # Run fine-tuning
    print("\nStarting fine-tuning...")
    tuned_sam = fine_tune_sam(
        sam,
        train_images,
        train_masks,
        epochs=4,
        lr=3e-5,
        batch_size=2
    )
    
    # Save and test
    print("\nSaving tuned model...")
    torch.save(tuned_sam.sam_mask_decoder.state_dict(), "tuned_shoreline_decoder.pth")
    
    # Quick test
    test_img = train_images[0]  # Test on first training image
    pred_mask = predict_shoreline(tuned_sam, test_img)
    
    pred_mask_uint8 = (pred_mask * 255).astype(np.uint8)
    pred_mask_resized = cv2.resize(pred_mask_uint8, (test_img.shape[1], test_img.shape[0]), interpolation=cv2.INTER_NEAREST)
    pred_mask_rgb = cv2.cvtColor(pred_mask_resized, cv2.COLOR_GRAY2RGB)

    # Stack and show
    combined = np.hstack([test_img, pred_mask_rgb])
    cv2.imshow("Test Prediction", combined)
    cv2.imwrite("prediction_output.jpg", cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))  # Save as JPEG

    cv2.waitKey(3000)
    cv2.destroyAllWindows()
