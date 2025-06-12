# ✅ Streamlit-Compatible AI Raman Analyzer with Gemini + Database
# Author: Your Name
# Version: Deployment Ready for GitHub + Streamlit Cloud

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import joblib
import json
import os
import requests
from typing import List, Dict, Any
from scipy.signal import find_peaks, medfilt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import google.generativeai as genai

# --- Set Page Config ---
st.set_page_config(page_title="AI Raman Analyzer", layout="wide", initial_sidebar_state="expanded")

# --- Gemini API Config ---
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found. Set it in Streamlit secrets or environment.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL_NAME = "gemini-2.0-flash"

# --- Utilities ---
def despike_spectrum(intensities):
    return medfilt(intensities, kernel_size=5)

def detect_peaks(wavenumbers, intensities):
    prominence = np.std(intensities) * 0.5
    peaks, _ = find_peaks(intensities, prominence=prominence, distance=10)
    return wavenumbers[peaks], intensities[peaks]

# --- Expert Interpreter ---
class ExpertInterpreter:
    def __init__(self, peaks, intensities, metadata):
        self.peaks = peaks
        self.intensities = intensities
        self.metadata = metadata
        self.functional_groups = []
        self.diagnostics = []

    def interpret(self):
        self._check_conditions()
        self._assign_groups()
        self._handle_complexities()
        return {"diagnostics": self.diagnostics, "functional_groups": self.functional_groups}

    def _check_conditions(self):
        if self.metadata["excitation"] in ["UV", "Visible"]:
            self.diagnostics.append("⚠️ UV/Vis excitation may induce significant fluorescence.")
        if self.metadata["excitation"] == "NIR":
            self.diagnostics.append("✅ NIR excitation is ideal for biological samples.")
        if self.metadata["crystalline"] == "No":
            self.diagnostics.append("🔍 Broad peaks suggest amorphous structure.")

    def _assign_groups(self):
        for peak in self.peaks:
            if 1600 <= peak <= 1800:
                self.functional_groups.append(("C=O Stretch", peak))
            elif 780 <= peak <= 800:
                self.functional_groups.append(("Phosphate Stretch", peak))
            elif 1000 <= peak <= 1030:
                self.functional_groups.append(("Aromatic Ring C-C Stretch", peak))
            elif 2800 <= peak <= 3000:
                self.functional_groups.append(("C-H Stretch", peak))
            elif 1081 <= peak <= 1091:
                self.functional_groups.append(("Carbonate Stretch", peak))
            elif 250 <= peak <= 500:
                self.functional_groups.append(("Inorganic Signature", peak))
            elif 3200 <= peak <= 3600:
                self.functional_groups.append(("O-H Stretch", peak))

    def _handle_complexities(self):
        if any(p < 500 for p in self.peaks):
            self.diagnostics.append("⚠️ Sub-500 cm⁻¹ peaks may indicate minerals.")
        if any(abs(self.peaks[i] - self.peaks[i+1]) < 15 for i in range(len(self.peaks) - 1)):
            self.diagnostics.append("🔍 Overlapping peaks detected – consider deconvolution.")

# --- Molecular Identifier ---
class MolecularIdentifier:
    def __init__(self, tolerance=50, min_matches=1):
        self.tolerance = tolerance
        self.min_matches = min_matches

    def identify(self, peaks, database):
        matches = []
        for category, compounds in database.items():
            for compound in compounds:
                matched = sum(1 for ref in compound.get("Peaks", []) if any(abs(p - ref.get("Wavenumber", 0)) <= self.tolerance for p in peaks))
                if matched >= self.min_matches:
                    matches.append({"Compound": compound.get("Name", "Unknown"), "Group": category, "Matched Peaks Count": matched})
        return sorted(matches, key=lambda x: x["Matched Peaks Count"], reverse=True)

# --- Raman Analyzer ---
class RamanAnalyzer:
    def __init__(self, json_paths, model_path=None):
        self.model = Pipeline([('scaler', StandardScaler()), ('mlp', MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000))])
        if model_path and os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                st.info(f"Loaded ML model from '{os.path.basename(model_path)}'.")
            except:
                st.warning("Failed to load model, using new one.")

        self.database = self._load_databases(["data/up.json"])
        try:
            self.ai_generator_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        except:
            self.ai_generator_model = None
            st.warning("Gemini model not loaded.")

        self.identifier = MolecularIdentifier()

    def _load_databases(self, paths):
        db = {}
        for path in paths:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for cat, comps in data.items():
                        db.setdefault(cat, []).extend(comps)
        return db

    def analyze(self, wavenumbers, intensities, metadata):
        intensities = despike_spectrum(intensities)
        peaks, peak_intensities = detect_peaks(wavenumbers, intensities)
        interp = ExpertInterpreter(peaks.tolist(), peak_intensities.tolist(), metadata).interpret()
        suggestions = self.identifier.identify(peaks.tolist(), self.database)
        return {
            "peaks": peaks,
            "intensities": peak_intensities,
            "functional_groups": interp["functional_groups"],
            "diagnostics": interp["diagnostics"],
            "compound_suggestions": suggestions,
            "processed_wavenumbers": wavenumbers,
            "processed_intensities": intensities
        }

    def generate_summary(self, name, group):
        if not self.ai_generator_model:
            return "AI model not loaded."
        prompt = f"Describe the Raman features of {name} in group {group}."
        try:
            response = self.ai_generator_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Failed to generate summary: {e}"

    def visualize(self, wavenumbers, intensities, peaks):
        fig, ax = plt.subplots()
        ax.plot(wavenumbers, intensities, label='Spectrum')
        ax.scatter(peaks, np.interp(peaks, wavenumbers, intensities), color='red', label='Peaks')
        ax.invert_xaxis()
        ax.set_xlabel("Raman Shift (cm⁻¹)")
        ax.set_ylabel("Intensity")
        ax.legend()
        return fig

# --- Streamlit UI ---
def main():
    st.title("🔬 AI-Powered Raman Analyzer")
    analyzer = RamanAnalyzer(["data/up.json"], None)

    uploaded_file = st.file_uploader("Upload Raman Spectrum CSV", type="csv")

    excitation = st.selectbox("Excitation Wavelength", ["UV", "Visible", "NIR"], index=2)
    sample_state = st.selectbox("Sample State", ["Solid", "Liquid", "Gas"], index=0)
    crystalline = st.selectbox("Crystalline?", ["Yes", "No"], index=0)

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if df.shape[1] < 2:
            st.error("CSV must have two columns: Wavenumber and Intensity.")
            return

        wavenumbers = df.iloc[:, 0].values
        intensities = df.iloc[:, 1].values

        metadata = {
            "excitation": excitation,
            "sample_state": sample_state,
            "crystalline": crystalline
        }

        result = analyzer.analyze(wavenumbers, intensities, metadata)

        st.pyplot(analyzer.visualize(result["processed_wavenumbers"], result["processed_intensities"], result["peaks"]))

        st.subheader("Functional Groups")
        for name, peak in result["functional_groups"]:
            st.write(f"- {name} at {peak:.1f} cm⁻¹")

        st.subheader("Diagnostics")
        for d in result["diagnostics"]:
            st.info(d)

        st.subheader("Top Compound Suggestions")
        if result["compound_suggestions"]:
            st.dataframe(pd.DataFrame(result["compound_suggestions"]))
            top = result["compound_suggestions"][0]
            st.markdown("### AI Summary")
            st.write(analyzer.generate_summary(top["Compound"], top["Group"]))
        else:
            st.warning("No matching compounds found.")

if __name__ == "__main__":
    main()
