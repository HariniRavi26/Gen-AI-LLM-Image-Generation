import streamlit as st
import torch
from diffusers import AutoPipelineForText2Image

st.set_page_config(
    page_title="AI Text to Image Generator",
    page_icon="🎨"
)

st.title("🎨 AI Text to Image Generator")
st.write("Enter a text prompt and generate an AI image.")

@st.cache_resource
def load_model():
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sd-turbo",
        torch_dtype=torch.float32
    )
    pipe = pipe.to("cpu")
    return pipe

prompt = st.text_input(
    "Enter your prompt",
    placeholder="A beautiful sunset over a mountain lake"
)

if st.button("Generate Image"):
    if not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Generating image..."):
            try:
                pipe = load_model()

                image = pipe(
                    prompt=prompt,
                    num_inference_steps=1,
                    guidance_scale=0.0
                ).images[0]

                st.success("Image generated successfully!")

                st.image(
                    image,
                    caption="Generated Image",
                    use_container_width=True
                )

            except Exception as e:
                st.error("Image generation failed.")
                st.exception(e)