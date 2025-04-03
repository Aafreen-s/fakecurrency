import streamlit as st
import pandas as pd
import joblib  # For loading the trained model
from fpdf import FPDF

# Load the trained model
model = joblib.load("fake_currency_model_balanced.pkl")  # Ensure you have saved your model

# Load dataset
df = pd.read_csv("bank_notes.csv", index_col=0)

def classify_note(variance, skewness, curtosis, entropy):
    input_data = [[variance, skewness, curtosis, entropy]]
    prediction = model.predict(input_data)[0]
    return "Real Note" if prediction == 0 else "Fake Note"

def generate_pdf(serial_number, variance, skewness, curtosis, entropy, result):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Fake Currency Detection Report", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Serial Number: {serial_number}", ln=True)
    pdf.cell(200, 10, txt=f"Variance: {variance}", ln=True)
    pdf.cell(200, 10, txt=f"Skewness: {skewness}", ln=True)
    pdf.cell(200, 10, txt=f"Curtosis: {curtosis}", ln=True)
    pdf.cell(200, 10, txt=f"Entropy: {entropy}", ln=True)
    pdf.cell(200, 10, txt=f"Result: {result}", ln=True)
    pdf_filename = "currency_report.pdf"
    pdf.output(pdf_filename)
    return pdf_filename

# Streamlit UI
st.title("Fake Currency Detection System")

serial_number = st.number_input("Enter Serial Number (Row Index in Dataset):", min_value=0, max_value=len(df)-1, step=1)

if st.button("Check Note"):
    row = df.iloc[int(serial_number) - 1]  # Adjusting to match actual Serial Number
    variance, skewness, curtosis, entropy = row["variance"], row["skewness"], row["curtosis"], row["entropy"]
    result = classify_note(variance, skewness, curtosis, entropy)
    
    if result == "Real Note":
        st.success("✅ The note is REAL")
    else:
        st.error("❌ The note is FAKE")
    
    # PDF Download
    pdf_file = generate_pdf(serial_number, variance, skewness, curtosis, entropy, result)
    with open(pdf_file, "rb") as f:
        st.download_button("Download Report", f, file_name=pdf_file, mime="application/pdf")
