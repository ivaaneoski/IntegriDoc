import random
import hashlib
from PIL import Image, ImageDraw, ImageFont

class SyntheticDocumentGenerator:
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        self.width = 600
        self.height = 400
        
        # We'll use default PIL font for maximum compatibility across OS
        self.font = ImageFont.load_default()
        # In a real scenario, we might want to scale the font, but default is fixed size
        
    def _generate_fictional_data(self) -> dict:
        first_names = ["John", "Jane", "Alice", "Bob", "Charlie", "Diana"]
        last_names = ["Doe", "Smith", "Johnson", "Williams", "Brown", "Jones"]
        
        return {
            "name": f"{random.choice(first_names)} {random.choice(last_names)}",
            "dob": f"{random.randint(1, 28):02d}/{random.randint(1, 12):02d}/{random.randint(1970, 2005)}",
            "id_number": f"{random.randint(10000000, 99999999)}",
            "issue_date": f"{random.randint(1, 28):02d}/{random.randint(1, 12):02d}/2020",
            "expiry_date": f"{random.randint(1, 28):02d}/{random.randint(1, 12):02d}/2030",
            "organization": "Fictional Corp"
        }

    def generate_id_card(self, source_id: str) -> dict:
        """
        Generates an authentic synthetic ID card.
        """
        # Set seed specifically for this source_id to make it deterministic
        seed_val = int(hashlib.md5(source_id.encode()).hexdigest()[:8], 16)
        random.seed(seed_val)
        
        data = self._generate_fictional_data()
        
        # Create base image
        img = Image.new('RGB', (self.width, self.height), color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw header
        draw.rectangle([0, 0, self.width, 60], fill=(50, 50, 150))
        draw.text((20, 20), data["organization"], fill="white", font=self.font)
        
        # Draw portrait placeholder
        portrait_rect = [400, 100, 550, 300]
        draw.rectangle(portrait_rect, fill=(200, 200, 200), outline=(100, 100, 100))
        draw.text((440, 190), "PORTRAIT", fill=(100, 100, 100), font=self.font)
        
        # Draw fields
        # Store bounding boxes for potential tampering
        fields_bboxes = {}
        
        def draw_field(label, value, y_pos, field_name):
            draw.text((50, y_pos), label, fill="gray", font=self.font)
            val_x = 200
            # Get text bounding box for the value
            bbox = draw.textbbox((val_x, y_pos), value, font=self.font)
            draw.text((val_x, y_pos), value, fill="black", font=self.font)
            # Store normalized bounding box: [x1, y1, x2, y2]
            fields_bboxes[field_name] = [
                bbox[0] / self.width,
                bbox[1] / self.height,
                bbox[2] / self.width,
                bbox[3] / self.height
            ]
            
        draw_field("Name:", data["name"], 100, "name")
        draw_field("DOB:", data["dob"], 150, "dob")
        draw_field("ID Number:", data["id_number"], 200, "id_number")
        draw_field("Issued:", data["issue_date"], 250, "issue_date")
        draw_field("Expires:", data["expiry_date"], 300, "expiry_date")
        
        return {
            "image": img,
            "data": data,
            "bboxes": fields_bboxes,
            "source_id": source_id,
            "template_family": "identity_card"
        }
