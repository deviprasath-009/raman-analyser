# Advanced AI Raman Expert System
# This is a rebuilt version focusing on a core expert system complemented by AI deduction.
# Version 7: Re-integrated multi-file support and stacked/overlay plotting.

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
from collections import defaultdict

# Import for Google Generative AI
import google.generativeai as genai

# --- Page Configuration ---
st.set_page_config(page_title="AI Raman Expert System", layout="wide", initial_sidebar_state="expanded")

# --- Peak Shape Model Functions ---
def gaussian(x, amplitude, center, sigma):
    """Gaussian peak model."""
    return amplitude * np.exp(-(x - center)**2 / (2 * sigma**2))

# =============================================================================
# 1. PEAK ANALYZER MODULE
# =============================================================================
class PeakAnalyzer:
    """Handles peak detection and shape fitting."""

    def detect_peaks(self, wavenumbers: np.ndarray, intensities: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Detects significant peaks in the spectrum."""
        prominence = np.std(intensities) * 0.1 # Lowered for sensitivity on normalized data
        peaks, properties = find_peaks(intensities, prominence=prominence, distance=15, width=3)
        return peaks, properties

    def fit_peak_shape(self, wavenumbers: np.ndarray, intensities: np.ndarray, peak_index: int) -> Dict[str, float]:
        """Fits a Gaussian model to an individual peak to find its shape."""
        window = 15
        start = max(0, peak_index - window)
        end = min(len(wavenumbers) - 1, peak_index + window)

        x_fit = wavenumbers[start:end]
        y_fit = intensities[start:end]

        center_guess = wavenumbers[peak_index]
        amplitude_guess = intensities[peak_index]
        sigma_guess = 5.0

        try:
            params, _ = curve_fit(gaussian, x_fit, y_fit, p0=[amplitude_guess, center_guess, sigma_guess])
            amplitude, center, sigma = params
            fwhm = 2.355 * abs(sigma)
        except RuntimeError:
            return {"center": center_guess, "height": amplitude_guess, "fwhm": 0}

        return {"center": center, "height": amplitude, "fwhm": fwhm}

    def analyze(self, wavenumbers: np.ndarray, intensities: np.ndarray) -> List[Dict]:
        """Full peak analysis workflow."""
        peak_indices, _ = self.detect_peaks(wavenumbers, intensities)
        detailed_peaks = [self.fit_peak_shape(wavenumbers, intensities, p_idx) for p_idx in peak_indices]
        return detailed_peaks

# =============================================================================
# 2. EXPERT SYSTEM MODULE
# =============================================================================
class ExpertSystem:
    """Correlates peaks with a Raman database to identify functional groups."""

    def __init__(self, correlation_table: List[Dict]):
        self.correlation_table = correlation_table

    def deduce_bonds(self, detailed_peaks: List[Dict]) -> List[Dict]:
        """Matches peaks against the correlation table, including intensity and shape."""
        identified_bonds = []
        for peak in detailed_peaks:
            peak_center = peak['center']
            for rule in self.correlation_table:
                min_wn, max_wn = float(rule.get("min_wavenumber", 0)), float(rule.get("max_wavenumber", 0))
                lower_bound, upper_bound = min(min_wn, max_wn), max(min_wn, max_wn)

                if lower_bound <= peak_center <= upper_bound:
                    bond_info = {
                        "peak_center": peak_center,
                        "relative_height": peak['height'],
                        "fwhm": peak['fwhm'],
                        "group": rule.get("group", "Unknown"),
                        "description": rule.get("description", "N/A")
                    }
                    identified_bonds.append(bond_info)
        
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
        
        try:
            self.ai_model = genai.GenerativeModel(gemini_model_name)
            self._init_messages.append({"type": "success", "text": f"Google Gemini model '{gemini_model_name}' initialized."})
        except Exception as e:
            self.ai_model = None
            self._init_messages.append({"type": "error", "text": f"Error initializing Gemini model: {e}."})

    def _load_json_data(self, path: str) -> Tuple[List[Dict], Optional[str], Optional[str]]:
        if not path or not os.path.exists(path):
            return [], None, f"Correlation table not found at: {path}"
        try:
            with open(path, 'r', encoding='utf-8') as f: data = json.load(f)
            return data, f"Successfully loaded correlation table from {os.path.basename(path)}.", None
        except Exception as e:
            return [], None, f"Error loading or parsing {os.path.basename(path)}: {e}"

    def get_init_messages(self) -> List[Dict[str, str]]:
        return self._init_messages

    def run_analysis(self, wavenumbers: np.ndarray, intensities: np.ndarray) -> Dict[str, Any]:
        """Runs the full analysis pipeline, including intensity normalization."""
        max_intensity = np.max(intensities)
        normalized_intensities = intensities / max_intensity if max_intensity > 0 else intensities

        detailed_peaks = self.peak_analyzer.analyze(wavenumbers, normalized_intensities)
        identified_bonds = self.expert_system.deduce_bonds(detailed_peaks)
        
        return {
            "detailed_peaks": detailed_peaks,
            "identified_bonds": identified_bonds,
            "processed_wavenumbers": wavenumbers,
            "processed_intensities": normalized_intensities
        }

    def _describe_peak_for_ai(self, bond: Dict) -> str:
        """Converts numerical peak data into a descriptive string for AI prompts."""
        height = bond.get('relative_height', 0)
        if height > 0.7: intensity = "very strong"
        elif height > 0.4: intensity = "strong"
        elif height > 0.1: intensity = "medium"
        else: intensity = "weak"
        
        fwhm = bond.get('fwhm', 0)
        shape = "sharp" if 0 < fwhm < 25 else "broad" if fwhm >= 25 else ""
        
        return f"A {intensity}{' ' + shape if shape else ''} peak"

    def refine_bonds_with_ai(self, identified_bonds: List[Dict]) -> Optional[List[Dict]]:
        """Uses AI to resolve ambiguities using peak intensity and shape as context."""
        if not self.ai_model: return None

        peak_assignments = defaultdict(list)
        for bond in identified_bonds:
            peak_assignments[f"{bond['peak_center']:.1f}"].append(bond)

        ambiguous_peaks = {p: bonds for p, bonds in peak_assignments.items() if len(bonds) > 1}
        unambiguous_peaks = {p: bonds[0] for p, bonds in peak_assignments.items() if len(bonds) == 1}

        if not ambiguous_peaks:
            st.success("No ambiguities found.")
            return identified_bonds

        prompt_parts = [
            "You are an expert chemist resolving ambiguities in a Raman spectrum. Use the context of the unambiguous peaks to determine the most likely assignment for each ambiguous peak. Consider peak intensity and shape.",
            "\n**Unambiguous Evidence (High Confidence):**"
        ]
        for peak, bond in unambiguous_peaks.items():
            desc = self._describe_peak_for_ai(bond)
            prompt_parts.append(f"- {desc} at ~{peak} cm⁻¹ is **{bond['group']} ({bond['description']})**.")

        prompt_parts.append("\n**Ambiguous Peaks to Resolve:**")
        for peak, bonds in ambiguous_peaks.items():
            desc = self._describe_peak_for_ai(bonds[0])
            options = "' OR '".join([f"{b['group']} ({b['description']})" for b in bonds])
            prompt_parts.append(f"- {desc} at ~{peak} cm⁻¹ could be: '**{options}**'.")
        
        prompt_parts.append("\n**Your Task:** For each ambiguous peak, choose the single most plausible assignment. Return a JSON array of objects, each with 'peak_center' (string), 'chosen_assignment' (string), and 'reasoning' (string).")

        try:
            with st.spinner("AI is resolving bond ambiguities..."):
                response = self.ai_model.generate_content("\n".join(prompt_parts), generation_config={"response_mime_type": "application/json"})
            ai_choices = json.loads(response.text.strip().replace("```json", "").replace("```", "").strip())

            refined_bonds = list(unambiguous_peaks.values())
            for choice in ai_choices:
                peak_center_str = str(choice['peak_center']).split(' ')[0]
                peak_center_float = float(peak_center_str)
                for original_bond in identified_bonds:
                    if np.isclose(original_bond['peak_center'], peak_center_float) and choice['chosen_assignment'] == f"{original_bond['group']} ({original_bond['description']})":
                        refined_bonds.append(original_bond)
                        break
            return sorted(refined_bonds, key=lambda x: x['peak_center'])
        except Exception as e:
            st.error(f"AI refinement failed: {e}")
            if 'response' in locals(): st.text_area("AI Raw Response:", response.text)
            return None

    def deduce_compounds_ai(self, identified_bonds: List[Dict]) -> Optional[Dict]:
        """Uses AI to deduce compounds from a list of bonds, considering intensity and shape."""
        if not self.ai_model: return None
        if not identified_bonds: return None

        prompt_parts = [
            "You are an expert chemist. Based on the following high-confidence list of identified functional groups, deduce the chemical identity of the sample. Pay close attention to the relative intensities and shapes of the peaks.",
            "\n**EVIDENCE - Identified Bonds and Functional Groups:**"
        ]
        for bond in identified_bonds:
            desc = self._describe_peak_for_ai(bond)
            prompt_parts.append(f"- {desc} at ~{bond['peak_center']:.0f} cm⁻¹ assigned to: **{bond['group']} ({bond['description']})**.")

        prompt_parts.append("\n**ANALYSIS TASK:** Propose the most likely pure substance and a plausible simple mixture that explains this evidence. Return your analysis in a single JSON object with 'pure_substance' and 'mixture' keys, each containing a list of objects with 'compound' and 'reasoning' keys.")

        try:
            with st.spinner("AI is deducing chemical structures from refined evidence..."):
                response = self.ai_model.generate_content("\n".join(prompt_parts), generation_config={"response_mime_type": "application/json"})
            return json.loads(response.text.strip().replace("```json", "").replace("```", "").strip())
        except Exception as e:
            st.error(f"AI deduction failed: {e}")
            if 'response' in locals(): st.text_area("AI Raw Response:", response.text)
            return None

    # =============================================================================
    # START: REBUILT VISUALIZE FUNCTION FOR MULTI-FILE SUPPORT
    # =============================================================================
    def visualize(self, spectra_data: List[Dict[str, Any]], plot_type: str) -> plt.Figure:
        """Generates a plot of multiple spectra with overlay or stacked options."""
        plt.style.use('seaborn-v0_8-darkgrid')

        if plot_type == "overlay":
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.set_title("Raman Spectra (Overlay)", fontsize=16)
            for i, data in enumerate(spectra_data):
                color = plt.cm.viridis(i / len(spectra_data))
                ax.plot(data['processed_wavenumbers'], data['processed_intensities'], label=data['label'], color=color, linewidth=1.5)
                peak_centers = [p['center'] for p in data['detailed_peaks']]
                peak_heights = [p['height'] for p in data['detailed_peaks']]
                ax.scatter(peak_centers, peak_heights, color=color, s=40, zorder=5, edgecolor='black')
            ax.legend()
            ax.set_ylabel("Normalized Intensity", fontsize=12)

        elif plot_type == "stacked":
            fig, axes = plt.subplots(nrows=len(spectra_data), figsize=(12, 2 * len(spectra_data)), sharex=True, squeeze=False)
            axes = axes.flatten()
            fig.suptitle("Raman Spectra (Stacked)", fontsize=16, y=0.99)
            
            for i, data in enumerate(spectra_data):
                ax = axes[i]
                offset = i * 1.1 # Offset for stacking
                ax.plot(data['processed_wavenumbers'], data['processed_intensities'] + offset, label=data['label'], color='royalblue', linewidth=1.5)
                peak_centers = [p['center'] for p in data['detailed_peaks']]
                peak_heights = [p['height'] for p in data['detailed_peaks']]
                ax.scatter(peak_centers, [h + offset for h in peak_heights], color='red', s=40, zorder=5, edgecolor='black')
                ax.set_ylabel(data['label'], rotation=0, labelpad=40, ha='right', va='center')
                ax.set_yticks([])
            fig.text(0.06, 0.5, 'Normalized Intensity (Offset)', va='center', rotation='vertical', fontsize=12)
            ax = axes[-1] # Use the last axis for shared properties

        ax.set_xlabel("Raman Shift (cm⁻¹)", fontsize=12)
        ax.invert_xaxis()
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        return fig
    # =============================================================================
    # END: REBUILT VISUALIZE FUNCTION
    # =============================================================================

# --- PubChem API Integration ---
@st.cache_data(show_spinner="Fetching data from PubChem...")
def fetch_pubchem_data(compound_name: str) -> Optional[Dict[str, Any]]:
    """Fetches chemical information for a compound from the PubChem API."""
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    try:
        cid_response = requests.get(f"{base_url}/compound/name/{compound_name}/cids/JSON", timeout=10)
        cid_response.raise_for_status()
        cid = cid_response.json()['IdentifierList']['CID'][0]
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

    if 'analyzer' not in st.session_state:
        gemini_api_key = st.secrets.get("GEMINI_API_KEY")
        if not gemini_api_key:
            st.error("GEMINI_API_KEY not found in Streamlit secrets.")
            st.session_state.analyzer = None
        else:
            try:
                genai.configure(api_key=gemini_api_key)
                st.session_state.analyzer = RamanAnalyzer("data/raw_raman_shiifts.json")
            except Exception as e:
                st.error(f"Failed to configure Google AI: {e}")
                st.session_state.analyzer = None
    
    analyzer = st.session_state.analyzer
    if analyzer:
        with st.expander("Show Initialization Status"):
            for msg in analyzer.get_init_messages():
                st.info(msg["text"]) if msg["type"] != "error" else st.error(msg["text"])

    st.sidebar.header("📂 Upload Spectrum Data")
    uploaded_files = st.sidebar.file_uploader(
        "Upload one or more Raman Spectra (CSV)", 
        type=["csv"],
        accept_multiple_files=True
    )

    if uploaded_files and analyzer:
        all_results = []
        for uploaded_file in uploaded_files:
            try:
                df = pd.read_csv(uploaded_file)
                wavenumbers, intensities = df.iloc[:, 0].values, df.iloc[:, 1].values
                analysis_results = analyzer.run_analysis(wavenumbers, intensities)
                analysis_results['label'] = uploaded_file.name
                all_results.append(analysis_results)
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {e}")
        
        st.session_state.all_results = all_results

    if 'all_results' in st.session_state:
        results = st.session_state.all_results
        
        st.header("📊 Analysis Results")
        
        plot_type = st.radio("Select Plot Type", ("Overlay", "Stacked"), horizontal=True)
        st.pyplot(analyzer.visualize(results, plot_type.lower()))

        # Aggregate all identified bonds for analysis
        all_identified_bonds = []
        for res in results:
            all_identified_bonds.extend(res['identified_bonds'])
        # Remove duplicates from aggregation
        all_identified_bonds = [dict(t) for t in {tuple(d.items()) for d in all_identified_bonds}]
        all_identified_bonds = sorted(all_identified_bonds, key=lambda x: x['peak_center'])
        
        st.session_state.all_bonds = all_identified_bonds

        st.subheader("Step 1: Expert System - Aggregated Bond Identification")
        if all_identified_bonds:
            st.dataframe(pd.DataFrame(all_identified_bonds).style.format({'peak_center': '{:.1f}', 'relative_height': '{:.2f}', 'fwhm': '{:.2f}'}))

        st.subheader("Step 2: AI-Powered Bond Refinement")
        if st.button("Refine All Bond Assignments (AI)"):
            st.session_state.refined_bonds = analyzer.refine_bonds_with_ai(st.session_state.all_bonds)
        
        if 'refined_bonds' in st.session_state and st.session_state.refined_bonds is not None:
            st.success("Bond assignments have been refined by the AI.")
            st.dataframe(pd.DataFrame(st.session_state.refined_bonds).style.format({'peak_center': '{:.1f}', 'relative_height': '{:.2f}', 'fwhm': '{:.2f}'}))

        st.header("Step 3: AI-Powered Compound Deduction")
        bonds_for_deduction = st.session_state.get('refined_bonds', st.session_state.get('all_bonds'))
        if st.button("Deduce Plausible Compounds (AI)"):
            ai_deductions = analyzer.deduce_compounds_ai(bonds_for_deduction)
            if ai_deductions:
                st.session_state.ai_deductions = ai_deductions
        
        if 'ai_deductions' in st.session_state:
            st.subheader("Case 1: Pure Substance Hypothesis")
            for item in st.session_state.ai_deductions.get('pure_substance', []): display_deduction(item)
            st.subheader("Case 2: Mixture Hypothesis")
            for item in st.session_state.ai_deductions.get('mixture', []): display_deduction(item, is_mixture=True)

def display_deduction(item: Dict, is_mixture: bool = False):
    """Helper function to display AI deduction results and PubChem button."""
    col1, col2 = st.columns([3, 1])
    compound_key = item.get('compound', 'N/A')
    with col1:
        st.markdown(f"#### {compound_key}")
        st.info(f"**Reasoning:** {item.get('reasoning', 'No reasoning provided.')}")
    with col2:
        button_key = f"pubchem_{'mix' if is_mixture else 'pure'}_{compound_key.replace(' ', '_')}"
        if st.button(f"Fetch PubChem Data", key=button_key):
            with st.spinner(f"Fetching data for {compound_key}..."):
                pubchem_data = fetch_pubchem_data(compound_key)
                if pubchem_data: st.json(pubchem_data)

if __name__ == "__main__":
    main()
