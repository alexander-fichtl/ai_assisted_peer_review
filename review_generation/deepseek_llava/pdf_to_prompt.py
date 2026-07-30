###--- Imports ----------------------------------------------------------------------------------------###
import io
import fitz
import torch
from PIL import Image
from transformers import LlavaProcessor, LlavaForConditionalGeneration
###----------------------------------------------------------------------------------------------------###



###--- Variables --------------------------------------------------------------------------------------###
VAR_filename = "testpdf"
PATH_pdf = VAR_filename + ".pdf"
PATH_output_prompt = VAR_filename + "_prompt.txt"
###----------------------------------------------------------------------------------------------------###



###--- Image Model ------------------------------------------------------------------------------------###
model_image_ID = "llava-hf/llava-1.5-7b-hf"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor_image = LlavaProcessor.from_pretrained(model_image_ID)
model_image = LlavaForConditionalGeneration.from_pretrained(
        model_image_ID, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(device)
###----------------------------------------------------------------------------------------------------###



###--- PDF Stuff --------------------------------------------------------------------------------------###
def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text() for page in doc)

def extract_images_from_pdf(pdf_path: str) -> list:
    images = []
    doc = fitz.open(pdf_path)
    for page_num, page in enumerate(doc):
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            images.append((f"Page {page_num + 1}", image))
    return images

def describe_image(image: Image.Image) -> str:
    prompt = "<|user|>\n<image>\nDescribe this scientific figure in detail.<|assistant|>"
    inputs = processor_image(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)
    output = model_image.generate(**inputs, max_new_tokens=512)
    return processor_image.tokenizer.decode(output[0], skip_special_tokens=True)

def build_prompt_from_pdf(pdf_path: str) -> str:
    text = extract_text_from_pdf(pdf_path)
    images = extract_images_from_pdf(pdf_path)

    captions = []
    for idx, (page_info, image) in enumerate(images):
        caption = describe_image(image)
        print(f"[Image Model Output - {page_info}]:\n{caption}\n") 
        captions.append(f"[Image {idx + 1} from {page_info}]: {caption}")

    full_text = "#### Paper Content ####\n\n" + text.strip()
    if captions:
        full_text += "\n\n#### Figures ####\n\n" + "\n".join(captions)
    full_text += "\n\n#### End ####"

    return full_text
###----------------------------------------------------------------------------------------------------###



###----------------------------------------------------------------------------------------------------###
paper_prompt = build_prompt_from_pdf(PATH_pdf)
with open(PATH_output_prompt, "w", encoding="utf-8") as f:
    f.write(paper_prompt)
print("Prompt written to:", PATH_output_prompt)
###----------------------------------------------------------------------------------------------------###

print("DONE.")