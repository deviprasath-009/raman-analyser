# Combined AI Raman Analyzer with Expert Rules + ML + Database Matching + GPT Description + PubChem API

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, medfilt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import streamlit as st
import json
import joblib
import os
import requests
from typing import List, Dict, Any

# Import for Google Generative AI
import google.generativeai as genai

# --- SET PAGE CONFIG FIRST ---
# This MUST be the first Streamlit command that runs in your script.
st.set_page_config(page_title="AI Raman Analyzer", layout="wide", initial_sidebar_state="expanded")

# --- Configuration for Google Gemini API ---
# IMPORTANT: Store your API key securely, e.g., in Streamlit secrets or environment variables.
# For demonstration, we'll read from Streamlit secrets.
# In your Streamlit app, go to Settings -> Secrets and add:
# GEMINI_API_KEY="YOUR_ACTUAL_GEMINI_API_KEY_HERE"
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found. Please set it in Streamlit secrets or as an environment variable.")
    st.stop() # Stop the app if API key is not found

genai.configure(api_key=GEMINI_API_KEY)
# Choose a model, 'gemini-pro' is a good general-purpose model.
# 'gemini-1.5-flash' or 'gemini-1.5-pro' are newer and potentially better, check pricing/free tier limits.
GEMINI_MODEL_NAME = "gemini-2.0-flash"


# ------------------------ Utility Functions ------------------------
def despike_spectrum(intensities: np.ndarray) -> np.ndarray:
    """Applies a median filter to remove spikes from the spectrum."""
    return medfilt(intensities, kernel_size=5)

def detect_peaks(wavenumbers: np.ndarray, intensities: np.ndarray) -> np.ndarray:
    """Detects peaks in the spectrum based on prominence."""
    prominence = np.std(intensities) * 0.5
    peaks, _ = find_peaks(intensities, prominence=prominence, distance=10)
    return wavenumbers[peaks], intensities[peaks]

# ------------------------ Expert Interpreter ------------------------
class ExpertInterpreter:
    """Interprets Raman spectrum features based on predefined expert rules."""
    def __init__(self, peaks: List[float], intensities: List[float], metadata: Dict):
        self.peaks = peaks
        self.intensities = intensities
        self.metadata = metadata
        self.functional_groups = []
        self.diagnostics = []

    def interpret(self):
        """Runs the expert interpretation logic."""
        self._check_conditions()
        self._assign_groups()
        self._handle_complexities()
        return {
            "diagnostics": self.diagnostics,
            "functional_groups": self.functional_groups
        }

    def _check_conditions(self):
        """Checks general sample conditions based on metadata."""
        if self.metadata["excitation"] in ["UV", "Visible"]:
            self.diagnostics.append("⚠️ UV/Vis excitation may induce significant fluorescence.")
        if self.metadata["excitation"] == "NIR":
            self.diagnostics.append("✅ NIR excitation is often ideal for biological samples due to reduced fluorescence.")
        if self.metadata["crystalline"] == "No":
            self.diagnostics.append("🔍 Broad peaks typically indicate an amorphous or disordered structure.")
        if self.metadata["sample_state"] == "Liquid":
            self.diagnostics.append("💧 Liquid samples may show broad solvent peaks.")


    def _assign_groups(self):
        """Assigns common functional groups based on peak positions."""
        for peak in self.peaks:
            if 1600 <= peak <= 1800:
                self.functional_groups.append(("C=O Stretch", peak))
            elif 780 <= peak <= 800:
                self.functional_groups.append(("Phosphate Stretch", peak))
            elif 1000 <= peak <= 1030:
                self.functional_groups.append(("Aromatic Ring C-C Stretch", peak))
            elif 2800 <= peak <= 3000:
                self.functional_groups.append(("C-H Stretch", peak))
            elif 1086 - 5 <= peak <= 1086 + 5:
                self.functional_groups.append(("Carbonate Stretch (CO3^2-)", peak))
            elif 250 <= peak <= 500:
                self.functional_groups.append(("Mineral/Inorganic Signature", peak))
            elif 3200 <= peak <= 3600:
                self.functional_groups.append(("O-H Stretch (Water/Hydroxyl)", peak))

    def _handle_complexities(self):
        """Identifies more complex spectral features or issues."""
        if any(p < 500 for p in self.peaks):
            self.diagnostics.append("⚠️ Peaks below 500 cm⁻¹ often suggest the presence of minerals or inorganic compounds.")
        
        peak_list = sorted(self.peaks)
        for i in range(len(peak_list) - 1):
            if abs(peak_list[i] - peak_list[i+1]) < 15:
                self.diagnostics.append("🔍 Close or overlapping peaks detected – consider deconvolution for better resolution.")
                break

# ------------------------ Molecular Identifier ------------------------
class MolecularIdentifier:
    """Identifies potential compounds by matching detected peaks against a database."""
    def __init__(self, tolerance: float = 50, min_matches: int = 1):
        self.tolerance = tolerance
        self.min_matches = min_matches

    def identify(self, peaks: List[float], database: Dict) -> List[Dict]:
        """
        Matches observed peaks against a database of reference compounds.
        """
        matches = []
        for category, compounds in database.items():
            for compound in compounds:
                matched_count = 0
                for ref_peak_data in compound.get("Peaks", []):
                    ref_wavenumber = ref_peak_data.get("Wavenumber")
                    if ref_wavenumber is not None:
                        for obs_peak in peaks:
                            if abs(obs_peak - ref_wavenumber) <= self.tolerance:
                                matched_count += 1
                                break
                
                if matched_count >= self.min_matches:
                    matches.append({
                        "Compound": compound.get("Name", "Unknown"),
                        "Group": category,
                        "Matched Peaks Count": matched_count
                    })
        return sorted(matches, key=lambda x: x["Matched Peaks Count"], reverse=True)

# ------------------------ Analyzer Core ------------------------
class RamanAnalyzer:
    """Core class for performing Raman spectrum analysis, including ML and AI generation."""
    def __init__(self, json_paths: List[str] = None, model_path: str = None):
        self.model = Pipeline([('scaler', StandardScaler()), ('mlp', MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42))])
        if model_path and os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                st.info(f"ML model loaded successfully from '{os.path.basename(model_path)}'.")
            except Exception as e:
                st.warning(f"Could not load ML model from '{os.path.basename(model_path)}': {e}. A new (untrained) model will be used.")
        else:
            st.info("No pre-trained ML model found or path invalid. A new (untrained) model will be used.")

        self.identifier = MolecularIdentifier()
        # Corrected file path using a raw string to avoid unicode escape errors
        # Note: For Streamlit Cloud, this path needs to be relative to your project root, not an absolute Windows path.
        # Ensure 'raman data 1 .json' is in the same directory as this script or a subfolder within your repo.
        self.database = self._load_databases([r"C:\\Users\\dpras\\OneDrive\\Desktop\\raman data\\up.json"]) 
        
        # Initialize Google Generative AI model
        try:
            self.ai_generator_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
            st.success(f"AI text generation model '{GEMINI_MODEL_NAME}' loaded!")
        except Exception as e:
            st.error(f"Could not load Google Gemini model: {e}. AI summary generation will not work.")
            self.ai_generator_model = None

    def _load_databases(self, paths: List[str]) -> Dict:
        """Loads Raman spectral databases from JSON files."""
        db = {}
        if not paths:
            st.warning("No database JSON paths provided. Database matching will be empty.")
            return db

        for path in paths:
            try:
                full_path = path
                if not os.path.isabs(path):
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    full_path = os.path.join(script_dir, path)

                if os.path.exists(full_path):
                    with open(full_path, 'r',encoding='utf-8') as f:
                        data = json.load(f)
                        for cat, compounds in data.items():
                            db.setdefault(cat, []).extend(compounds)
                    st.success(f"Successfully loaded database from '{os.path.basename(path)}'.")
                else:
                    st.warning(f"Database file not found at '{full_path}'. Skipping '{os.path.basename(path)}'.")

            except json.JSONDecodeError:
                st.error(f"Error: Invalid JSON format in '{os.path.basename(path)}'. Please check the file.")
            except Exception as e:
                st.error(f"An unexpected error occurred loading '{os.path.basename(path)}': {e}")
        return db

    def analyze(self, wavenumbers: np.ndarray, intensities: np.ndarray, metadata: Dict) -> Dict:
        """Performs the full Raman analysis workflow."""
        intensities_despiked = despike_spectrum(intensities)
        peaks, peak_intensities = detect_peaks(wavenumbers, intensities_despiked)

        interpreter = ExpertInterpreter(peaks.tolist(), peak_intensities.tolist(), metadata)
        interpretation = interpreter.interpret()

        suggestions = self.identifier.identify(peaks.tolist(), self.database)
        
        return {
            "peaks": peaks,
            "intensities": peak_intensities,
            "functional_groups": interpretation["functional_groups"],
            "diagnostics": interpretation["diagnostics"],
            "compound_suggestions": suggestions,
            "processed_wavenumbers": wavenumbers,
            "processed_intensities": intensities_despiked
        }

    def generate_summary(self, name: str, group: str) -> str:
        """Generates an AI-powered summary for a given compound using Google Gemini."""
        if not self.ai_generator_model:
            return "AI summary generation model not loaded. Please check API key and model setup."
            
        prompt = f"Describe the compound '{name}' in Raman spectroscopy, focusing on its common Raman features and typical applications. It belongs to the {group} group."
        try:
            # Use the generate_content method for the Gemini model
            response = self.ai_generator_model.generate_content(prompt)
            # Access the text from the Candidates attribute of the response
            summary = response.text.strip()
            
            # Basic cleaning (similar to the previous GPT-2 logic, can be adapted)
            clean_summary = summary.replace(prompt, "").strip()
            if len(clean_summary) > 150 and '.' in clean_summary[:150]:
                clean_summary = clean_summary.rsplit('.', 1)[0] + '.'
            elif len(clean_summary) > 150:
                clean_summary = clean_summary[:150] + "..."
            
            return clean_summary if clean_summary else "AI summary could not be generated clearly."
        except Exception as e:
            return f"AI summary generation failed: {e}. Please check your API key, model access, and prompt."

    def visualize(self, wavenumbers: np.ndarray, intensities: np.ndarray, peaks: np.ndarray):
        """Generates a matplotlib plot of the Raman spectrum with detected peaks."""
        plt.style.use('seaborn-v0_8-darkgrid')

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(wavenumbers, intensities, label='Spectrum', color='#1f77b4', linewidth=1.5)
        
        peak_indices = np.searchsorted(wavenumbers, peaks)
        peak_indices = peak_indices[(peak_indices >= 0) & (peak_indices < len(wavenumbers))]
        
        ax.scatter(peaks, intensities[peak_indices],
                   color='red', s=50, zorder=5, label='Detected Peaks', edgecolor='k', alpha=0.8)

        ax.set_title("Raman Spectrum", fontsize=16, pad=15)
        ax.set_xlabel("Raman Shift (cm⁻¹)", fontsize=12)
        ax.set_ylabel("Intensity (Arb. Units)", fontsize=12)
        
        ax.invert_xaxis()

        ax.legend(loc='best', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        ax.tick_params(axis='both', which='major', labelsize=10)

        plt.tight_layout()
        return fig

# Instantiate Analyzer Core once using st.cache_resource
@st.cache_resource
def get_analyzer_instance(json_db_paths: List[str], ml_model_path: str = None) -> RamanAnalyzer:
    """Initializes and caches the RamanAnalyzer instance."""
    return RamanAnalyzer(json_db_paths, ml_model_path)

# --- PubChem API Integration ---
@st.cache_data(show_spinner="Fetching data from PubChem...")
def fetch_pubchem_data(compound_name: str) -> Dict[str, Any]:
    """Fetches chemical information for a compound from PubChem PUG-REST API."""
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    
    # 1. Get CID from compound name
    cid_url = f"{base_url}/compound/name/{compound_name}/cids/JSON"
    try:
        response = requests.get(cid_url, timeout=10)
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        cid_data = response.json()
        cids = cid_data.get("IdentifierList", {}).get("CID")
        if not cids:
            return {"error": f"No CID found for '{compound_name}' on PubChem."}
        cid = cids[0] # Take the first CID if multiple are returned
    except requests.exceptions.RequestException as e:
        return {"error": f"Error fetching CID for '{compound_name}': {e}"}
    except json.JSONDecodeError:
        return {"error": f"Could not decode JSON from PubChem CID response for '{compound_name}'."}

    # 2. Get properties (MolecularFormula, IUPACName, CanonicalSMILES)
    properties_url = f"{base_url}/compound/cid/{cid}/property/MolecularFormula,IUPACName,CanonicalSMILES/JSON"
    properties = {}
    try:
        response = requests.get(properties_url, timeout=10)
        response.raise_for_status()
        prop_data = response.json()
        props_list = prop_data.get("PropertyTable", {}).get("Properties")
        if props_list:
            properties = props_list[0] # Get the first (and usually only) set of properties
            # Remove CID from properties if it's there
            properties.pop('CID', None)
    except requests.exceptions.RequestException as e:
        properties["error"] = f"Error fetching properties: {e}"
    except json.JSONDecodeError:
        properties["error"] = "Could not decode JSON from PubChem properties response."

    # 3. Get description
    description = "No description available."
    description_url = f"{base_url}/compound/cid/{cid}/description/JSON"
    try:
        response = requests.get(description_url, timeout=10)
        response.raise_for_status()
        desc_data = response.json()
        # PubChem description structure can be nested;
        # look for 'Description' or 'Markup' in different levels
        description_sections = desc_data.get("InformationList", {}).get("Information", [])
        for info_item in description_sections:
            if info_item.get("Title") == "Description" and "Description" in info_item:
                description = info_item["Description"]
                break
            elif "Description" in info_item: # Fallback if Title isn't 'Description'
                description = info_item["Description"]
                break
            elif "Markup" in info_item: # Sometimes it's in Markup
                description = info_item["Markup"]
                break
    except requests.exceptions.RequestException as e:
        description = f"Error fetching description: {e}"
    except json.JSONDecodeError:
        description = "Could not decode JSON from PubChem description response."

    return {
        "cid": cid,
        "properties": properties,
        "description": description
    }


# ------------------------ Streamlit Interface ------------------------
def main():
    # REMOVED st.set_page_config from here, it's now at the top of the script.
    st.title("🔬 Advanced AI-Powered Raman Analyzer")
    st.markdown("---")

    script_directory = os.path.dirname(os.path.abspath(__file__))
    DB_JSON_PATHS = [os.path.join(script_directory, "my_database.json")]
    ML_MODEL_PATH = os.path.join(script_directory, "raman_mlp_model.joblib")

    analyzer = get_analyzer_instance(DB_JSON_PATHS, ML_MODEL_PATH)

    st.sidebar.header("📂 Data & Sample Information")

    uploaded_file = st.sidebar.file_uploader(
        "Upload Raman Spectrum (CSV)", 
        type=["csv"], 
        help="Upload a two-column CSV file where the first column is Wavenumber (cm⁻¹) and the second is Intensity (Arb. Units).",
        accept_multiple_files=False
    )

    st.sidebar.subheader("🧪 Sample Metadata")
    excitation = st.sidebar.selectbox(
        "Excitation Wavelength", 
        ["UV", "Visible", "NIR"], 
        index=2, 
        help="Select the laser excitation wavelength used during measurement."
    )
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        sample_state = st.selectbox(
            "Sample State", 
            ["Solid", "Liquid", "Gas"], 
            index=0, 
            help="Physical state of the sample."
        )
    with col2:
        crystalline = st.selectbox(
            "Crystalline?", 
            ["Yes", "No"], 
            index=0, 
            help="Indicate if the sample has a crystalline or amorphous structure."
        )
    
    col3, col4 = st.sidebar.columns(2)
    with col3:
        polarized = st.selectbox(
            "Polarized?", 
            ["Yes", "No"], 
            index=1, 
            help="Was polarized light used for the Raman measurement?"
        )
    with col4:
        sample_origin = st.text_input(
            "Sample Origin", 
            value="biological", 
            help="e.g., polymer, pharmaceutical, geological, biological, environmental."
        )

    st.sidebar.subheader("⚡ Measurement Parameters")
    laser_power = st.sidebar.slider(
        "Laser Power (mW)", 
        1, 500, 50, 
        help="Adjust the laser power used for the measurement."
    )
    integration_time = st.sidebar.slider(
        "Integration Time (ms)", 
        10, 10000, 1000, 
        help="Set the signal integration time per data point."
    )

    if uploaded_file:
        with st.status("Processing spectrum and running analysis...", expanded=True) as status:
            try:
                status.write("Reading uploaded CSV file...")
                df = pd.read_csv(uploaded_file)
                
                if df.shape[1] < 2:
                    status.error("Invalid CSV format. Please ensure your file has at least two columns (Wavenumber, Intensity).")
                    st.error("Please upload a valid CSV file with at least two columns.")
                    status.update(label="Analysis failed!", state="error", expanded=False)
                    return
                
                wavenumbers = df.iloc[:, 0].values
                intensities = df.iloc[:, 1].values

                meta = {
                    "excitation": excitation,
                    "sample_state": sample_state,
                    "origin": sample_origin,
                    "crystalline": crystalline,
                    "polarized": polarized,
                    "laser_power": laser_power,
                    "integration_time": integration_time
                }
                
                status.write("Performing Raman analysis (despiking, peak detection, expert interpretation)...")
                results = analyzer.analyze(wavenumbers, intensities, meta)
                
                status.update(label="Analysis complete!", state="complete", expanded=False)
                st.success("Analysis successfully completed!")

                st.subheader("📈 Raman Spectrum Visualization")
                st.pyplot(analyzer.visualize(results["processed_wavenumbers"], results["processed_intensities"], results["peaks"]), use_container_width=True)
                st.markdown("---")

                st.subheader("📊 Analysis Results")

                st.markdown("#### 🔍 Diagnostics")
                if results["diagnostics"]:
                    for d in results["diagnostics"]:
                        st.info(f"- {d}") 
                else:
                    st.info("No specific diagnostics identified for this spectrum based on expert rules.")
                st.markdown("---")

                st.markdown("#### 📚 Functional Groups Identified")
                if results["functional_groups"]:
                    for name, peak in results["functional_groups"]:
                        st.success(f"- **{name}** at {peak:.1f} cm⁻¹") 
                else:
                    st.info("No common functional groups detected based on expert rules.")
                st.markdown("---")

                st.markdown("#### 🧪 Top Compound Suggestions from Database")
                if not results["compound_suggestions"]:
                    st.warning("No matching compounds found in the loaded database with the given tolerance and minimum peak matches.")
                    st.info("Consider adjusting the 'tolerance' or 'min_matches' parameters in the MolecularIdentifier class if you expect matches.")
                else:
                    df_match = pd.DataFrame(results["compound_suggestions"])
                    st.dataframe(df_match, use_container_width=True, hide_index=True)

                    top_match = results["compound_suggestions"][0]
                    st.markdown("#### 🧠 AI-Generated Summary for Top Match")
                    # Check if the AI model was successfully loaded before trying to generate summary
                    if analyzer.ai_generator_model:
                        with st.spinner(f"Generating AI summary for {top_match['Compound']} using {GEMINI_MODEL_NAME}..."):
                            ai_summary = analyzer.generate_summary(top_match['Compound'], top_match['Group'])
                        st.markdown(ai_summary)
                    else:
                        st.warning("AI summary generation skipped because the model could not be loaded.")

                    # --- PubChem Details Integration ---
                    st.markdown("#### 🌐 Fetch Details from PubChem (Top Match)")
                    if st.button(f"Get PubChem Details for {top_match['Compound']}"):
                        pubchem_info = fetch_pubchem_data(top_match['Compound'])
                        if "error" in pubchem_info:
                            st.error(pubchem_info["error"])
                        else:
                            with st.expander(f"**Details for {top_match['Compound']} (CID: {pubchem_info.get('cid', 'N/A')})**", expanded=True):
                                st.markdown(f"**Molecular Formula:** {pubchem_info['properties'].get('MolecularFormula', 'N/A')}")
                                st.markdown(f"**IUPAC Name:** {pubchem_info['properties'].get('IUPACName', 'N/A')}")
                                st.markdown(f"**Canonical SMILES:** `{pubchem_info['properties'].get('CanonicalSMILES', 'N/A')}`")
                                st.markdown("---")
                                st.markdown("**Description:**")
                                st.write(pubchem_info.get('description', 'No detailed description available.'))
                                st.markdown(f"[View on PubChem](https://pubchem.ncbi.nlm.nih.gov/compound/{pubchem_info.get('cid')})")
                    st.markdown("---")

            except Exception as e:
                status.error(f"An error occurred during analysis: {e}")
                st.error(f"Failed to process the spectrum. Error: {e}")
                st.exception(e)
                status.update(label="Analysis failed!", state="error", expanded=False)

    else:
        st.info("Please upload a Raman spectrum CSV file to begin analysis.")
        st.markdown("""
        **Getting Started:**
        1.  **Upload your Raman spectrum data** as a two-column CSV (Wavenumber, Intensity) using the uploader in the sidebar.
        2.  **Adjust the sample metadata** and measurement parameters in the sidebar.
        3.  The analysis results, including functional groups, compound suggestions, and an AI summary, will appear here.
        """)
        st.markdown("---")
        st.subheader("App Features:")
        st.markdown("""
        -   **Automated Pre-processing:** Despiking and peak detection.
        -   **Expert Rules:** Identifies common functional groups and provides diagnostic insights.
        -   **Database Matching:** Suggests compounds by comparing your spectrum's peaks to a custom database.
        -   **AI-Powered Summary:** Generates concise descriptions of suggested compounds using **Google Gemini API**.
        -   **PubChem Integration:** Fetches additional chemical details (formula, IUPAC name, SMILES, description) from the PubChem database.
        """)

if __name__ == "__main__":
    main()
