import gradio as gr
from PIL import Image
from app.model import predict, CLASS_NAMES

def segment_image(image):
    if image is None:
        return None
    result = predict(image)
    return result

with gr.Blocks(title="BDD100K Segmentation", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🚗 Driving Scene Segmentation\n
    **Model:** U-Net + EfficientNet-B3 — trained on BDD100K\n
    **Classes:** 19 semantic classes (road, car, person, sky...)\n
    **Metrics:** mIoU 51.6% — Pixel Accuracy 91.8%
    """)

    with gr.Row():
        input_img = gr.Image(
            type="pil",
            label="Input — driving scene"
        )
        output_img = gr.Image(
            type="pil",
            label="Output — segmentation"
        )

    btn = gr.Button("Segment", variant="primary")
    btn.click(
        fn=segment_image,
        inputs=input_img,
        outputs=output_img
    )

    

    if __name__ == "__main__":
        demo.launch()