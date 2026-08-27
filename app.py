import streamlit as st
import torch
from diffusers import AutoPipelineForText2Image

st.set_page_config(
    page_title="AI Text to Image Generator",
    page_icon="🎨"
)

st.title("🎨 AI Text to Image Generator")
st.write("Enter a text prompt and generate an image using AI.")

@st.cache_resource
def load_model():
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sd-turbo",
        torch_dtype=torch.float32
    )
    pipe = pipe.to("cpu")
    return pipe

prompt = st.text_area(
    "Enter your prompt",
    placeholder="A cute white cat sitting in a beautiful flower garden",
    height=100
)

if st.button("🎨 Generate Image"):

    if prompt.strip() == "":
        st.warning("Please enter a prompt.")

    else:

        with st.spinner("Generating your image..."):

            try:
                pipe = load_model()

                image = pipe(
                    prompt=prompt,
                    num_inference_steps=1,
                    guidance_scale=0.0
                ).images[0]

                st.image(
                    image,
                    caption="Generated Image",
                    use_container_width=True
                )

            except Exception as e:
                st.error("An error occurred while generating the image.")
                st.exception(e)