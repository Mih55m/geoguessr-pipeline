import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
from models import GeoguessrModel

# UI Configuration
st.set_page_config(page_title="Geoguessr AI", page_icon="🌍", layout="centered")
st.title("🌍 Geoguessr AI Predictor")
st.write("Upload an image, and the AI will analyze the geography, architecture, and flora to guess where it was taken!")

CLASSES = [
    'East & Southeast Asia', 'Eastern Europe', 'North America', 
    'Oceania & Islands', 'Other Regions', 'Scandinavia & Northern Europe', 
    'South America', 'Sub-Saharan Africa', 'Western Europe'
]

# Load the Model
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GeoguessrModel(num_classes=len(CLASSES))
    # Load the weights
    model.load_state_dict(torch.load("best_efficientB3_geoguessr.pth", map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model, device

# Initialize the model
model, device = load_model()

# Image Transformation Pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# The Front-End
uploaded_file = st.file_uploader("Drop a landscape image here...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the image
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # The Inference Engine
    with st.spinner('Analyzing geographic features...'):
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            # Convert raw math into percentages
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

        # Get the top prediction
        top_prob, top_idx = torch.max(probabilities, 0)
        prediction = CLASSES[top_idx]
        confidence = top_prob.item() * 100

    # Display Results
    st.success(f"**Predicted Region:** {prediction}")
    st.info(f"**Confidence:** {confidence:.2f}%")

    # Display a beautiful bar chart of all probabilities
    st.write("### AI Confidence Breakdown")
    
    # Convert tensor to numpy for the chart
    prob_array = probabilities.cpu().numpy() * 100
    chart_data = pd.DataFrame(prob_array, index=CLASSES, columns=["Confidence (%)"])
    st.bar_chart(chart_data)