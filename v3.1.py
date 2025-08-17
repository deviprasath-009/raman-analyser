# Advanced AI Raman Expert System
# This is a rebuilt version focusing on a core expert system complemented by AI deduction.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import streamlit as st
import json
import os
import requests
from typing import List, Dict, Any, Tuple, Optional

# Import for Google Generative AI
import google.generativeai as genai

# --- Page Configuration ---
st.set_page_config(page_title="AI Raman Expert System", layout="wide", initial_sidebar_state="expanded")

# --- Peak Shape Model Functions ---
def gaussian(x, amplitude, center, sigma):
    """Gaussian peak model."""
    return amplitude * np.exp(-(x - center)**2 / (2 * sigma**2))

def lorentzian(x, amplitude, center, gamma):
    """Lorentzian peak model."""
    return amplitude * gamma**2 / ((x - center)**2 + gamma**2)

# =============================================================================
# 1. PEAK ANALYZER MODULE
# =============================================================================
class PeakAnalyzer:
    """Handles peak detection and shape fitting."""

    def detect_peaks(self, wavenumbers: np.ndarray, intensities: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Detects significant peaks in the spectrum."""
        # Prominence is a measure of how much a peak stands out from the surrounding baseline
        prominence = np.std(intensities) * 0.75 
        peaks, properties = find_peaks(intensities, prominence=prominence, distance=15, width=3)
        return peaks, properties

    def fit_peak_shape(self, wavenumbers: np.ndarray, intensities: np.ndarray, peak_index: int) -> Dict[str, float]:
        """
        Fits a Gaussian or Lorentzian model to an individual peak to find its shape.
        Returns detailed peak parameters like FWHM.
        """
        # Define a window around the peak for fitting
        window = 15
        start = max(0, peak_index - window)
        end = min(len(wavenumbers) - 1, peak_index + window)

        x_fit = wavenumbers[start:end]
        y_fit = intensities[start:end]

        # Initial guesses for the curve fitting
        center_guess = wavenumbers[peak_index]
        amplitude_guess = intensities[peak_index]
        sigma_guess = 5.0

        try:
            # Attempt to fit a Gaussian model
            params, _ = curve_fit(gaussian, x_fit, y_fit, p0=[amplitude_guess, center_guess, sigma_guess])
            amplitude, center, sigma = params
            # Full Width at Half Maximum (FWHM) for a Gaussian
            fwhm = 2.355 * abs(sigma)
            shape = "Gaussian"
        except RuntimeError:
            # If Gaussian fails, fallback to Lorentzian or return basic info
            return {"center": center_guess, "height": amplitude_guess, "fwhm": 0, "shape": "Unknown"}

        return {"center": center, "height": amplitude, "fwhm": fwhm, "shape": shape}

    def analyze(self, wavenumbers: np.ndarray, intensities: np.ndarray) -> List[Dict]:
        """Full peak analysis workflow."""
        peak_indices, _ = self.detect_peaks(wavenumbers, intensities)
        
        detailed_peaks = []
        for p_idx in peak_indices:
            peak_details = self.fit_peak_shape(wavenumbers, intensities, p_idx)
            detailed_peaks.append(peak_details)
            
        return detailed_peaks

# =============================================================================
# 2. EXPERT SYSTEM MODULE
# =============================================================================
class ExpertSystem:
    """Correlates peaks with a Raman database to identify functional groups."""

    def __init__(self, correlation_table: List[Dict]):
        self.correlation_table = correlation_table

    def deduce_bonds(self, detailed_peaks: List[Dict]) -> List[Dict]:
        """Matches peaks against the correlation table to deduce bond details."""
        identified_bonds = []
        
        for peak in detailed_peaks:
            peak_center = peak['center']
            for rule in self.correlation_table:
                min_wn = float(rule.get("min_wavenumber", 0))
                max_wn = float(rule.get("max_wavenumber", 0))

                lower_bound = min(min_wn, max_wn)
                upper_bound = max(min_wn, max_wn)

                if lower_bound <= peak_center <= upper_bound:
                    bond_info = {
                        "peak_center": peak_center,
                        "fwhm": peak['fwhm'],
                        "group": rule.get("group", "Unknown"),
                        "description": rule.get("description", "N/A")
                    }
                    identified_bonds.append(bond_info)
                    # A peak can sometimes be assigned to multiple groups, so we don't break here
        
        # Remove duplicates if a peak is matched multiple times with the exact same rule
        unique_bonds = [dict(t) for t in {tuple(d.items()) for d in identified_bonds}]
        return sorted(unique_bonds, key=lambda x: x['peak_center'])


# =============================================================================
# 3. CORE ANALYZER & AI MODULE
# =============================================================================
class RamanAnalyzer:
    """Orchestrates the analysis workflow and integrates AI for compound deduction."""

    def __init__(self, correlation_table_path: str, gemini_model_name: str = "gemini-1.5-flash-latest"):
        self._init_messages = []
        self.peak_analyzer = PeakAnalyzer()
        
        correlation_table, success, error = self._load_json_data(correlation_table_path)
        if success: self._init_messages.append({"type": "success", "text": success})
        if error: self._init_messages.append({"type": "error", "text": error})

        self.expert_system = ExpertSystem(correlation_table)
        
        # Initialize Google Generative AI model
        try:
            self.ai_model = genai.GenerativeModel(gemini_model_name)
            self._init_messages.append({"type": "success", "text": f"Google Gemini model '{gemini_model_name}' initialized."})
        except Exception as e:
            self.ai_model = None
            self._init_messages.append({"type": "error", "text": f"Error initializing Gemini model: {e}."})

    def _load_json_data(self, path: str) -> Tuple[List[Dict], Optional[str], Optional[str]]:
        """Loads the correlation table from a local JSON file."""
        if not path or not os.path.exists(path):
            return [], None, f"Correlation table not found at: {path}"
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data, f"Successfully loaded correlation table from {os.path.basename(path)}.", None
        except Exception as e:
            return [], None, f"Error loading or parsing {os.path.basename(path)}: {e}"

    def get_init_messages(self) -> List[Dict[str, str]]:
        return self._init_messages

    def run_analysis(self, wavenumbers: np.ndarray, intensities: np.ndarray) -> Dict[str, Any]:
        """Runs the full analysis pipeline."""
        detailed_peaks = self.peak_analyzer.analyze(wavenumbers, intensities)
        identified_bonds = self.expert_system.deduce_bonds(detailed_peaks)
        
        return {
            "detailed_peaks": detailed_peaks,
            "identified_bonds": identified_bonds,
            "processed_wavenumbers": wavenumbers,
            "processed_intensities": intensities
        }

    def deduce_compounds_ai(self, identified_bonds: List[Dict]) -> Optional[Dict]:
        """
        Uses AI to deduce plausible compounds from a list of identified bonds.
        Considers both pure substance and mixture scenarios.
        """
        if not self.ai_model:
            st.error("AI model is not available. Cannot perform compound deduction.")
            return None
        if not identified_bonds:
            st.warning("No bonds were identified, cannot deduce compounds.")
            return None

        prompt_parts = [
            "You are an expert chemist specializing in Raman spectroscopy. Your task is to deduce the chemical identity of a sample based on a list of identified functional groups and their peak characteristics.",
            "You must consider two distinct possibilities: the sample is a single pure substance, or it is a mixture of simple, common compounds.",
            "\n**EVIDENCE - Identified Bonds and Functional Groups:**"
        ]
        for bond in identified_bonds:
            peak_desc = "sharp" if bond['fwhm'] < 20 else "broad"
            prompt_parts.append(f"- A {peak_desc} peak at ~{bond['peak_center']:.0f} cm⁻¹ corresponding to: **{bond['group']} ({bond['description']})**.")

        prompt_parts.append("\n**ANALYSIS TASK:**")
        prompt_parts.append("1.  **Pure Substance Scenario:** Propose the most likely single chemical compound that contains ALL the identified functional groups. Provide a clear reasoning for your choice.")
        prompt_parts.append("2.  **Mixture Scenario:** Propose a plausible, simple mixture of 2-3 common compounds that would collectively explain the presence of all the identified groups. Explain which compound in the mixture is responsible for which group(s).")
        prompt_parts.append("\nReturn your complete analysis in a single, structured JSON object. The JSON object must have two top-level keys: 'pure_substance' and 'mixture'. Each key should contain a list of objects, where each object has 'compound' and 'reasoning' keys.")

        full_prompt = "\n".join(prompt_parts)

        try:
            with st.spinner("AI is deducing chemical structures from bond evidence..."):
                response = self.ai_model.generate_content(
                    full_prompt,
                    generation_config={"response_mime_type": "application/json"}
                )
            
            # Clean the response to ensure it's valid JSON
            json_string = response.text.strip()
            if json_string.startswith("```json"):
                json_string = json_string[7:]
            if json_string.endswith("```"):
                json_string = json_string[:-3]
            json_string = json_string.strip()

            return json.loads(json_string)

        except Exception as e:
            st.error(f"AI deduction failed: {e}")
            # It's helpful to show the raw response for debugging
            if 'response' in locals() and hasattr(response, 'text'):
                st.text_area("AI Raw Response (for debugging):", response.text, height=150)
            return None

    def visualize(self, wavenumbers: np.ndarray, intensities: np.ndarray, detailed_peaks: List[Dict], label: str) -> plt.Figure:
        """Generates a plot of the spectrum and its identified peaks."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(wavenumbers, intensities, label=label, color='royalblue', linewidth=1.5)
        
        peak_centers = [p['center'] for p in detailed_peaks]
        peak_heights = [p['height'] for p in detailed_peaks]
        
        ax.scatter(peak_centers, peak_heights, color='red', s=50, zorder=5, edgecolor='black', label='Detected Peaks')

        for peak in detailed_peaks:
            ax.text(peak['center'], peak['height'], f" {peak['center']:.0f}", fontsize=9, verticalalignment='bottom')

        ax.set_title(f"Raman Spectrum: {label}", fontsize=16)
        ax.set_xlabel("Raman Shift (cm⁻¹)", fontsize=12)
        ax.set_ylabel("Intensity (Arbitrary Units)", fontsize=12)
        ax.invert_xaxis()
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        plt.tight_layout()
        return fig

# --- PubChem API Integration ---
@st.cache_data(show_spinner="Fetching data from PubChem...")
def fetch_pubchem_data(compound_name: str) -> Optional[Dict[str, Any]]:
    """Fetches chemical information for a compound from the PubChem API."""
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    try:
        # Get CID
        cid_response = requests.get(f"{base_url}/compound/name/{compound_name}/cids/JSON", timeout=10)
        cid_response.raise_for_status()
        cid = cid_response.json()['IdentifierList']['CID'][0]

        # Get Properties
        props_response = requests.get(f"{base_url}/compound/cid/{cid}/property/MolecularFormula,IUPACName,CanonicalSMILES/JSON", timeout=10)
        props_response.raise_for_status()
        properties = props_response.json()['PropertyTable']['Properties'][0]

        return {"cid": cid, **properties}
    except Exception as e:
        st.error(f"Could not fetch data for '{compound_name}' from PubChem: {e}")
        return None

# =============================================================================
# 4. STREAMLIT UI
# =============================================================================
def main():
    st.title("🔬 Advanced AI Raman Expert System")
    st.markdown("A focused tool for peak analysis, bond correlation, and AI-powered compound deduction.")

    # --- Initialization ---
    CORRELATION_TABLE_PATH = "data/raw_raman_shiifts.json"
    
    # Use a session state object to hold the analyzer and prevent re-initialization on every rerun
    if 'analyzer' not in st.session_state:
        gemini_api_key = st.secrets.get("GEMINI_API_KEY")
        if not gemini_api_key:
            st.error("GEMINI_API_KEY not found in Streamlit secrets. AI features will be disabled.")
            st.session_state.analyzer = None
        else:
            try:
                genai.configure(api_key=gemini_api_key)
                st.session_state.analyzer = RamanAnalyzer(CORRELATION_TABLE_PATH)
            except Exception as e:
                st.error(f"Failed to configure Google AI: {e}")
                st.session_state.analyzer = None
    
    analyzer = st.session_state.analyzer

    if analyzer:
        with st.expander("Show Initialization Status", expanded=False):
            for msg in analyzer.get_init_messages():
                if msg["type"] == "error": st.error(msg["text"])
                else: st.success(msg["text"])

    # --- Sidebar for Data Upload ---
    st.sidebar.header("📂 Upload Spectrum Data")
    uploaded_file = st.sidebar.file_uploader(
        "Upload a single Raman Spectrum (CSV)",
        type=["csv"],
        help="CSV file must have two columns: Wavenumber and Intensity."
    )

    if uploaded_file:
        if not analyzer:
            st.error("Analyzer is not initialized. Please check your API key and file paths.")
            return
        
        # This is the corrected block with the full try...except structure
        try:
            df = pd.read_csv(uploaded_file)
            if df.shape[1] < 2:
                st.error("Invalid CSV format. Please provide a two-column file.")
                return
            
            wavenumbers = df.iloc[:, 0].values
            intensities = df.iloc[:, 1].values
            
            # --- Run Analysis ---
            analysis_results = analyzer.run_analysis(wavenumbers, intensities)
            
            st.header("📊 Analysis Results")

            # --- Visualization ---
            st.subheader("Spectrum Plot and Detected Peaks")
            fig = analyzer.visualize(
                analysis_results['processed_wavenumbers'],
                analysis_results['processed_intensities'],
                analysis_results['detailed_peaks'],
                uploaded_file.name
            )
            st.pyplot(fig, use_container_width=True)

            # --- Expert System Output ---
            st.subheader("Expert System: Bond & Functional Group Identification")
            bonds_df = pd.DataFrame(analysis_results['identified_bonds'])
            if not bonds_df.empty:
                st.dataframe(bonds_df.style.format({
                    'peak_center': '{:.1f}',
                    'fwhm': '{:.2f}'
                }), use_container_width=True)
            else:
                st.info("No functional groups could be confidently identified from the detected peaks.")

            # --- AI Deduction Section ---
            st.header("🤖 AI-Powered Compound Deduction")
            if st.button("Deduce Plausible Compounds (AI)"):
                ai_deductions = analyzer.deduce_compounds_ai(analysis_results['identified_bonds'])
                
                if ai_deductions:
                    # --- Pure Substance Case ---
                    st.subheader("Case 1: Pure Substance Hypothesis")
                    pure_substance_results = ai_deductions.get('pure_substance', [])
                    if pure_substance_results:
                        for item in pure_substance_results:
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(f"#### {item.get('compound', 'N/A')}")
                                st.info(f"**Reasoning:** {item.get('reasoning', 'No reasoning provided.')}")
                            with col2:
                                compound_key = item.get('compound', 'N/A')
                                if st.button(f"Fetch PubChem Data for {compound_key}", key=f"pubchem_{compound_key}"):
                                    pubchem_data = fetch_pubchem_data(compound_key)
                                    if pubchem_data:
                                        st.json(pubchem_data)

                    else:
                        st.warning("The AI could not identify a single compound that explains all the evidence.")

                    # --- Mixture Case ---
                    st.subheader("Case 2: Mixture Hypothesis")
                    mixture_results = ai_deductions.get('mixture', [])
                    if mixture_results:
                         for item in mixture_results:
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(f"#### {item.get('compound', 'N/A')}")
                                st.info(f"**Reasoning:** {item.get('reasoning', 'No reasoning provided.')}")
                            with col2:
                                compound_key = item.get('compound', 'N/A')
                                if st.button(f"Fetch PubChem Data for {compound_key}", key=f"pubchem_mix_{compound_key}"):
                                    pubchem_data = fetch_pubchem_data(compound_key)
                                    if pubchem_data:
                                        st.json(pubchem_data)
                    else:
                        st.warning("The AI could not propose a plausible mixture.")
        
        except Exception as e:
            st.error(f"An error occurred during file processing or analysis: {e}")

    else:
        st.info("Please upload a Raman spectrum file to begin the analysis.")

if __name__ == "__main__":
    main()
