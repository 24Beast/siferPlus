# Importing Libraries
import os
from transformers import CLIPProcessor, CLIPModel

# CLIP Class
class CLIPAnnotator:
    
    def __init__(self, model_name = "openai/clip-vit-base-patch32"):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.valid_extns = ["png", "jpg", "jpeg"]
        
    def generateLabels(self, imgs, prompts):
        """
        Parameters
        ----------
        imgs : list
            list of PIL Images.
        prompts : list
            list of mutually exclusive text prompts.

        Returns
        -------
        probs : torch.tensor
            Probabilities for each image prompt pair. 
            Returns a matrix of size (num images, num prompts).

        """
        inputs = self.processor(text=prompts, images=imgs, return_tensors="pt", padding=True)
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image  
        probs = logits_per_image.softmax(dim=1)  
        return probs
    
    def generateLabelsDir(self, img_dir, prompts):
        """
        Parameters
        ----------
        img_dir : string
            Path where all the images are.
        prompts : list
            list of mutually exclusive text prompts.

        Returns
        -------
        probs : torch.tensor
            Probabilities for each image prompt pair. 
            Returns a matrix of size (num images, num prompts).

        """
        img_names = os.listdir(img_dir)
        imgs = []
        for name in img_names:
            if not(name.split(".")[-1] in self.valid_extns):
                continue
            imgs.append(Image.open(img_dir+name))
        probs = self.generateLabels(imgs, prompts)
        return probs
    
if __name__ == "__main__":
    import time    
    from PIL import Image
    
    DATA_DIR = "../data/celeba/img_align_celeba/"
    PROMPTS = ["Blonde", "Not Blonde"]
    
    img_names = os.listdir(DATA_DIR)[:10]
    imgs = []
    for name in img_names:
        imgs.append(Image.open(DATA_DIR+name))
    
    annotator = CLIPAnnotator()
    probs = annotator.generateLabels(imgs, PROMPTS)
    
    for num, img in enumerate(imgs):
        img.show()
        print(f"Probability for img {num}: {probs[num]}")
        time.sleep(2)

