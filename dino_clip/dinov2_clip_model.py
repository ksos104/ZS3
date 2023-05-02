import torch
import vision_transformer as vits

def main():
    dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    img = torch.rand(5, 3, 224, 224)
    
    dino_model = vits.vit_base()
    print("done!")
    
    
    
if __name__ == '__main__':
    main()