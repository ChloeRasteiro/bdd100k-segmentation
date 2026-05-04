import gradio as gr
from PIL import Image
from app.model import predict, CLASS_NAMES

def segment_image(image):
    if image is None:
        return None
    result = predict(image)
    return result

with gr.Blocks(title="BDD100K Segmentation") as demo:

    gr.Markdown("""
    # 🚗 Driving Scene Segmentation
    **Model:** U-Net + EfficientNet-B3 — trained on BDD100K
    **Classes:** 19 semantic classes (road, car, person, sky...)
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

    gr.Markdown("### Classes détectées")
    gr.Dataframe(
        value=[[i, name] for i, name in enumerate(CLASS_NAMES)],
        headers=["ID", "Class"],
        interactive=False
    )

    if __name__ == "__main__":
        demo.launch()