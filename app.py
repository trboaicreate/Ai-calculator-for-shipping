
import streamlit as st
from PIL import Image
import pytesseract
import requests
from bs4 import BeautifulSoup
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

# Load image classifier
model = MobileNetV2(weights='imagenet')

# Product spec database
product_data = {
    "electronics": {"weight_kg":2.5,"cbft":0.4},
    "clothing": {"weight_kg":1.0,"cbft":0.25},
    "tools": {"weight_kg":3.0,"cbft":0.6},
    "general": {"weight_kg":1.0,"cbft":0.2}
}

# Customs rates
customs_rates = {
    "electronics": {"duty":0.07,"ob":0.06},
    "clothing": {"duty":0.10,"ob":0.06},
    "tools": {"duty":0.15,"ob":0.06},
    "general": {"duty":0.06,"ob":0.06}
}

def classify_image(img):
    img = img.resize((224,224))
    x = image.img_to_array(img)
    x = preprocess_input(np.expand_dims(x,0))
    preds = model.predict(x)
    top = decode_predictions(preds,1)[0][0][1]
    return top.lower()

def scrape_title(url):
    r = requests.get(url, timeout=5)
    soup = BeautifulSoup(r.text, 'html.parser')
    title = soup.title.string if soup.title else "product"
    return title.lower()

def lookup_specs(name):
    for key,v in product_data.items():
        if key in name:
            return key, v["weight_kg"], v["cbft"]
    return "general", product_data["general"]["weight_kg"], product_data["general"]["cbft"]

def calc_shipping(weight, cbft):
    wc = weight * 10
    if cbft <=6:
        vc=50
    elif cbft<=10:
        vc=100
    else:
        vc=100 + (cbft-10)*6
    return max(wc, vc)

def calc_customs(declared, category):
    r = customs_rates.get(category, customs_rates["general"])
    duty = declared * r["duty"]
    ob = (declared + duty) * r["ob"]
    return duty, ob

st.title("📦 Curaçao Shipping & Customs Estimator")

choice = st.radio("Choose input type:", ("Image", "Website URL"))
product_name = None

if choice=="Image":
    uploaded = st.file_uploader("Upload product image", type=["png","jpg","jpeg"])
    if uploaded:
        img = Image.open(uploaded)
        product_name = classify_image(img)
        st.image(img, width=250, caption=f"Detected: {product_name}")

else:
    url = st.text_input("Paste product page URL")
    if url:
        product_name = scrape_title(url)
        st.write(f"Scraped title: **{product_name}**")

if product_name:
    cat, wt, vol = lookup_specs(product_name)
    st.write(f"Estimated category: **{cat}**")
    st.write(f"Weight (kg): **{wt}**, Volume (cbft): **{vol}**")

    declared = st.number_input("Declared value (USD)", min_value=0.0, value=0.0, step=1.0)
    if st.button("Calculate"):
        ship = calc_shipping(wt, vol)
        duty, ob = calc_customs(declared, cat)
        total = ship + duty + ob

        st.write("---")
        st.write(f"**Shipping Cost:** ${ship:.2f}")
        st.write(f"**Customs Duty:** ${duty:.2f}")
        st.write(f"**OB Sales Tax:** ${ob:.2f}")
        st.write(f"**Total Cost:** ${total:.2f}")
        st.write("---")
        st.download_button("💾 Download Invoice", data=f""" 
Product: {product_name}
Category: {cat}
Weight: {wt} kg • Volume: {vol} cbft
Declared Value: ${declared:.2f}
Shipping: ${ship:.2f}
Duty: ${duty:.2f}
OB Tax: ${ob:.2f}
Total: ${total:.2f}
""", file_name="invoice.txt")
