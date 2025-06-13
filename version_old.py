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
from typing import List, Dict, Any, Tuple

# Import for Google Generative AI
import google.generativeai as genai

# --- SET PAGE CONFIG FIRST ---
# This line MUST be the very first Streamlit command in your script.
st.set_page_config(page_title="AI Raman Analyzer", layout="wide", initial_sidebar_state="expanded")

# --- Configuration for Google Gemini API and Model ---
# API key retrieval and genai.configure are now handled within get_analyzer_instance
# to ensure Streamlit's environment is fully initialized.


# ------------------------ Utility Functions ------------------------
def despike_spectrum(intensities: np.ndarray) -> np.ndarray:
    """Applies a median filter to remove spikes from the spectrum."""
    return medfilt(intensities, kernel_size=5)

def detect_peaks(wavenumbers: np.ndarray, intensities: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Detects peaks in the spectrum based on prominence."""
    prominence = np.std(intensities) * 0.5 # Prominence based on standard deviation
    peaks, _ = find_peaks(intensities, prominence=prominence, distance=10) # Minimum distance between peaks
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
        # Note: Peak ranges and assignments are simplified for demonstration
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
    def __init__(self, tolerance: float = 30, min_matches: int = 1):
        self.tolerance = tolerance
        self.min_matches = min_matches

    def identify(self, peaks: List[float], database: Dict) -> List[Dict]:
        """Matches observed peaks against a reference database of compounds."""
        matches = []

        if not isinstance(database, dict):
            return []

        for category, compounds in database.items():
            if not isinstance(compounds, list):
                continue
            for compound in compounds:
                matched_count = 0
                for ref_peak in compound.get("Peaks", []):
                    ref_wavenumber = ref_peak.get("Wavenumber")
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
    def __init__(self, json_paths: List[str] = None, model_path: str = None, 
                 gemini_model_name: str = "gemini-pro", gemini_api_key: str = None): # Added gemini_api_key
        self.loading_messages = []

        self.model = Pipeline([('scaler', StandardScaler()), ('mlp', MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42))])
        if model_path and os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                self.loading_messages.append(f"ML_INFO: ML model loaded successfully from '{os.path.basename(model_path)}'.")
            except Exception as e:
                self.loading_messages.append(f"ML_WARNING: Could not load ML model from '{os.path.basename(model_path)}': {e}. A new (untrained) model will be used.")
        else:
            self.loading_messages.append("ML_INFO: No pre-trained ML model found or path invalid. A new (untrained) model will be used.")

        self.identifier = MolecularIdentifier()
        self.database, db_messages = self._load_databases(json_paths)
        self.loading_messages.extend(db_messages)
            
        self.ai_generator_model = None
        if gemini_api_key: # Check if API key is provided
            try:
                genai.configure(api_key=gemini_api_key)
                self.ai_generator_model = genai.GenerativeModel(gemini_model_name)
                self.loading_messages.append("API_INFO: Google Gemini API configured successfully.")
            except Exception as e:
                self.loading_messages.append(f"AI_ERROR: Failed to load AI model '{gemini_model_name}' or configure API: {e}. AI summary generation will not work.")
        else:
            self.loading_messages.append("API_ERROR: GEMINI_API_KEY not provided. AI summary generation will not work.")


    def _load_databases(self, paths: List[str]) -> Tuple[Dict, List[str]]:
        """Loads Raman spectral databases from JSON files or URLs and returns status messages."""
        db = {}
        messages = []

        if not paths:
            messages.append("DB_WARNING: No database paths provided. Database matching will be empty.")
            return db, messages

        for path in paths:
            try:
                if path.startswith(('http://', 'https://')):
                    response = requests.get(path)
                    response.raise_for_status()
                    data = response.json()
                    messages.append(f"DB_INFO: Loaded database from URL: {path}")
                else:
                    if not os.path.isabs(path):
                        script_dir = os.path.dirname(os.path.abspath(__file__))
                        full_path = os.path.join(script_dir, path)
                    else:
                        full_path = path

                    if os.path.exists(full_path):
                        with open(full_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        messages.append(f"DB_INFO: Loaded database from file: {os.path.basename(path)}")
                    else:
                        messages.append(f"DB_WARNING: Database file not found at: {full_path}")
                        continue

                if isinstance(data, dict):
                    for category, compounds in data.items():
                        if isinstance(compounds, list):
                            db.setdefault(category, []).extend(compounds)
                        else:
                            messages.append(f"DB_WARNING: Category '{category}' in {path} is not a list, skipping.")
                elif isinstance(data, list):
                    db.setdefault("Uncategorized", []).extend(data)
                    messages.append(f"DB_WARNING: JSON file {os.path.basename(path)} is a list. Wrapped under 'Uncategorized'.")
                else:
                    messages.append(f"DB_ERROR: Unsupported JSON structure in {path}. Skipped.")

            except json.JSONDecodeError as e:
                messages.append(f"DB_ERROR: JSON parsing error in {path}: {e}")
            except requests.exceptions.RequestException as e:
                messages.append(f"DB_ERROR: Request error loading from {path}: {e}")
            except Exception as e:
                messages.append(f"DB_ERROR: Unexpected error loading {path}: {e}")

        return db, messages

    def analyze(self, wavenumbers: np.ndarray, intensities: np.ndarray, metadata: Dict) -> Dict:
        """Performs the full Raman analysis workflow."""
        intensities_despiked = despike_spectrum(intensities)
        peaks, peak_intensities = detect_peaks(wavenumbers, intensities_despiked)

        interpreter = ExpertInterpreter(peaks.tolist(), peak_intensities.tolist(), metadata)
        interpretation = interpreter.interpret()

        suggestions = self.identifier.identify(peaks.tolist(), self.database)
            
        return {
            "peaks": peaks,
            "peak_intensities": peak_intensities,
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
            response = self.ai_generator_model.generate_content(prompt)
            summary = response.text.strip()
            
            clean_summary = summary.replace(prompt, "").strip()
            if len(clean_summary) > 150 and '.' in clean_summary[:150]:
                clean_summary = clean_summary.rsplit('.', 1)[0] + '.'
            elif len(clean_summary) > 150:
                clean_summary = clean_summary[:150] + "..."
            
            return clean_summary if clean_summary else "AI summary could not be generated clearly."
        except Exception as e:
            return f"AI summary generation failed: {e}. Please check your API key, model access, and prompt."

    def visualize(self, spectra_data: List[Dict[str, Any]], plot_type: str = "overlay"):
        """
        Generates a matplotlib plot of one or more Raman spectra.
        spectra_data: List of dicts, each containing 'wavenumbers', 'intensities', 'peaks', 'label'.
        plot_type: 'overlay' or 'stacked'.
        """
        plt.style.use('seaborn-v0_8-darkgrid')

        if plot_type == "overlay":
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.set_title("Raman Spectra Overlay", fontsize=16, pad=15)
            for i, data in enumerate(spectra_data):
                color = plt.cm.get_cmap('viridis', len(spectra_data))(i)
                ax.plot(data['wavenumbers'], data['intensities'], label=data['label'], color=color, linewidth=1.5)
                peak_indices = np.searchsorted(data['wavenumbers'], data['peaks'])
                peak_indices = peak_indices[(peak_indices >= 0) & (peak_indices < len(data['wavenumbers']))]
                ax.scatter(data['peaks'], data['intensities'][peak_indices],
                           color=color, s=50, zorder=5, edgecolor='k', alpha=0.8)
            ax.legend(loc='best', fontsize=10)
            ax.set_ylabel("Intensity (Arb. Units)", fontsize=12)

        elif plot_type == "stacked":
            nrows = len(spectra_data)
            fig, axes = plt.subplots(nrows=nrows, figsize=(10, 2 * nrows), sharex=True)
            if nrows == 1:
                axes = [axes]

            fig.suptitle("Raman Spectra Stacked View", fontsize=16, y=1.02)
            
            max_intensity_overall = max(np.max(d['intensities']) for d in spectra_data)
            offset_factor = max_intensity_overall * 1.2

            for i, data in enumerate(spectra_data):
                ax = axes[i]
                current_offset = i * offset_factor
                
                ax.plot(data['wavenumbers'], data['intensities'] + current_offset, 
                         label=data['label'], color='#1f77b4', linewidth=1.5)
                
                peak_indices = np.searchsorted(data['wavenumbers'], data['peaks'])
                peak_indices = peak_indices[(peak_indices >= 0) & (peak_indices < len(data['wavenumbers']))]
                ax.scatter(data['peaks'], data['intensities'][peak_indices] + current_offset,
                           color='red', s=50, zorder=5, edgecolor='k', alpha=0.8)
                
                ax.set_ylabel(data['label'], fontsize=10)
                ax.set_yticks([])
                ax.tick_params(axis='both', which='major', labelsize=10)
                ax.grid(True, linestyle='--', alpha=0.6)
                
            axes[-1].set_xlabel("Raman Shift (cm⁻¹)", fontsize=12)

        ax.invert_xaxis()

        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        return fig

@st.cache_resource
def get_analyzer_instance(json_db_paths: List[str], ml_model_path: str = None, gemini_api_key: str = None) -> Tuple[RamanAnalyzer, List[str]]: # Added gemini_api_key
    """
    Initializes and caches the RamanAnalyzer instance.
    Handles API key retrieval and Gemini configuration here, preventing early Streamlit errors.
    Returns the analyzer instance and a list of setup messages.
    """
    setup_messages = []

    # genai.configure is now handled within the RamanAnalyzer constructor
    # based on the provided gemini_api_key
    gemini_model_name_for_analyzer = "gemini-2.0-flash"

    analyzer = RamanAnalyzer(json_db_paths, ml_model_path, gemini_model_name=gemini_model_name_for_analyzer, gemini_api_key=gemini_api_key) # Pass the API key
    setup_messages.extend(analyzer.loading_messages)

    return analyzer, setup_messages


# --- PubChem API Integration ---
@st.cache_data(show_spinner="Fetching data from PubChem...")
def fetch_pubchem_data(compound_name: str) -> Dict[str, Any]:
    """Fetches chemical information for a compound from PubChem PUG-REST API."""
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    
    cid_url = f"{base_url}/compound/name/{compound_name}/cids/JSON"
    try:
        response = requests.get(cid_url, timeout=10)
        response.raise_for_status()
        cid_data = response.json()
        cids = cid_data.get("IdentifierList", {}).get("CID")
        if not cids:
            return {"error": f"No CID found for '{compound_name}' on PubChem."}
        cid = cids[0]
    except requests.exceptions.RequestException as e:
        return {"error": f"Error fetching CID for '{compound_name}': {e}"}
    except json.JSONDecodeError:
        return {"error": f"Could not decode JSON from PubChem CID response for '{compound_name}'."}

    properties_url = f"{base_url}/compound/cid/{cid}/property/MolecularFormula,IUPACName,CanonicalSMILES/JSON"
    properties = {}
    try:
        response = requests.get(properties_url, timeout=10)
        response.raise_for_status()
        prop_data = response.json()
        props_list = prop_data.get("PropertyTable", {}).get("Properties")
        if props_list:
            properties = props_list[0]
            properties.pop('CID', None)
    except requests.exceptions.RequestException as e:
        properties["error"] = f"Error fetching properties: {e}"
    except json.JSONDecodeError:
        properties["error"] = "Could not decode JSON from PubChem properties response."

    description = "No description available."
    description_url = f"{base_url}/compound/cid/{cid}/description/JSON"
    try:
        response = requests.get(description_url, timeout=10)
        response.raise_status()
        desc_data = response.json()
        description_sections = desc_data.get("InformationList", {}).get("Information", [])
        for info_item in description_sections:
            if info_item.get("Title") == "Description" and "Description" in info_item:
                description = info_item["Description"]
                break
            elif "Description" in info_item:
                description = info_item["Description"]
                break
            elif "Markup" in info_item:
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
    st.title("rudra's Raman Analyzer")
    st.markdown("---")

    script_directory = os.path.dirname(os.path.abspath(__file__))
    
    GITHUB_DB_URL = "https://raw.githubusercontent.com/deviprasath-009/raman-analyser/main/data/up.json"
    LOCAL_DB_PATH = os.path.join(script_directory, "raman data 1 .json")

    ML_MODEL_PATH = os.path.join(script_directory, "raman_mlp_model.joblib")
    
    DB_JSON_PATHS = [GITHUB_DB_URL, LOCAL_DB_PATH]

    # Retrieve API key BEFORE calling the cached function
    gemini_api_key = st.secrets.get("GEMINI_API_KEY")

    analyzer, setup_messages = get_analyzer_instance(DB_JSON_PATHS, ML_MODEL_PATH, gemini_api_key)

    # --- Display setup messages in a collapsible expander ---
    # Determine if any important messages (warnings/errors) exist to decide default expander state
    has_critical_messages = any(
        msg.startswith("API_ERROR:") or msg.startswith("DB_ERROR:") or msg.startswith("ML_WARNING:") or msg.startswith("AI_ERROR:")
        for msg in setup_messages
    )
    
    with st.expander("Show Setup & Initialization Messages", expanded=has_critical_messages):
        if setup_messages:
            for msg in setup_messages:
                if msg.startswith("API_ERROR:") or msg.startswith("DB_ERROR:") or msg.startswith("AI_ERROR:"):
                    st.error(msg.replace("API_ERROR: ", "").replace("DB_ERROR: ", "").replace("AI_ERROR: ", ""))
                elif msg.startswith("API_WARNING:") or msg.startswith("DB_WARNING:") or msg.startswith("ML_WARNING:"):
                    st.warning(msg.replace("API_WARNING: ", "").replace("DB_WARNING: ", "").replace("ML_WARNING: ", ""))
                elif msg.startswith("API_INFO:") or msg.startswith("DB_INFO:") or msg.startswith("ML_INFO:"):
                    st.info(msg.replace("API_INFO: ", "").replace("DB_INFO: ", "").replace("ML_INFO: ", ""))
        else:
            st.info("No specific setup messages to display.")
    # --- End of setup messages display ---

    st.sidebar.header("📂 Data & Sample Information")

    uploaded_files = st.sidebar.file_uploader(
        "Upload Raman Spectrum(s) (CSV)", 
        type=["csv"], 
        help="Upload one or more two-column CSV files (Wavenumber, Intensity) OR single CSV with multiple intensity columns.",
        accept_multiple_files=True
    )

    st.sidebar.subheader("🧪 Sample Metadata (Applies to all uploaded files)")
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

    if uploaded_files:
        all_processed_spectra = []
        all_compound_suggestions = {}
        all_functional_groups = []
        all_diagnostics = []

        with st.status("Processing spectrum(s) and running analysis...", expanded=True) as status:
            meta = {
                "excitation": excitation,
                "sample_state": sample_state,
                "origin": sample_origin,
                "crystalline": crystalline,
                "polarized": polarized,
                "laser_power": laser_power,
                "integration_time": integration_time
            }

            for file_idx, uploaded_file in enumerate(uploaded_files):
                st.write(f"Processing file {file_idx + 1}: {uploaded_file.name}")
                try:
                    df = pd.read_csv(uploaded_file)
                    
                    if df.shape[1] < 2:
                        st.warning(f"Skipping {uploaded_file.name}: Invalid CSV format. Requires at least two columns.")
                        continue
                    
                    wavenumbers = df.iloc[:, 0].values

                    intensity_columns_count = df.shape[1] - 1
                    for col_idx in range(intensity_columns_count):
                        intensity_label = f"{uploaded_file.name} - Spectrum {col_idx + 1}"
                        intensities = df.iloc[:, col_idx + 1].values

                        status.write(f"Analyzing {intensity_label}...")
                        results = analyzer.analyze(wavenumbers, intensities, meta)
                        
                        all_processed_spectra.append({
                            'wavenumbers': results['processed_wavenumbers'],
                            'intensities': results['processed_intensities'],
                            'peaks': results['peaks'],
                            'label': intensity_label
                        })

                        for diag in results['diagnostics']:
                            if diag not in all_diagnostics:
                                all_diagnostics.append(diag)
                        for fg_name, fg_peak in results['functional_groups']:
                            if (fg_name, fg_peak) not in all_functional_groups:
                                all_functional_groups.append((fg_name, fg_peak))
                        
                        for suggestion in results['compound_suggestions']:
                            compound_name = suggestion['Compound']
                            if compound_name not in all_compound_suggestions:
                                all_compound_suggestions[compound_name] = {
                                    'Group': suggestion['Group'],
                                    'Total Matched Peaks': suggestion['Matched Peaks Count'],
                                    'Occurrences': 1,
                                    'Source Spectra': [intensity_label]
                                }
                            else:
                                all_compound_suggestions[compound_name]['Total Matched Peaks'] += suggestion['Matched Peaks Count']
                                all_compound_suggestions[compound_name]['Occurrences'] += 1
                                if intensity_label not in all_compound_suggestions[compound_name]['Source Spectra']:
                                    all_compound_suggestions[compound_name]['Source Spectra'].append(intensity_label)

                except Exception as e:
                    st.error(f"An error occurred processing {uploaded_file.name}: {e}")
                    st.exception(e)

            status.update(label="Analysis complete!", state="complete", expanded=False)
            st.success("Analysis successfully completed for all uploaded spectra!")

            st.subheader("📈 Raman Spectra Visualization")
            plot_type = st.radio("Select Plot Type:", ("Overlay View", "Stacked View"), horizontal=True)
            
            if all_processed_spectra:
                st.pyplot(analyzer.visualize(all_processed_spectra, plot_type.split()[0].lower()), use_container_width=True)
            else:
                st.warning("No valid spectra were processed for visualization.")
            st.markdown("---")

            st.subheader("📊 Consolidated Analysis Results")

            st.markdown("#### 🔍 Diagnostics from All Spectra")
            if all_diagnostics:
                for d in all_diagnostics:
                    st.info(f"- {d}")
            else:
                st.info("No specific diagnostics identified across all spectra based on expert rules.")
            st.markdown("---")

            st.markdown("#### 📚 Functional Groups Identified Across All Spectra")
            if all_functional_groups:
                for name, peak in sorted(all_functional_groups, key=lambda x: x[1]):
                    st.success(f"- **{name}** at {peak:.1f} cm⁻¹")
            else:
                st.info("No common functional groups detected across all spectra based on expert rules.")
            st.markdown("---")

            st.markdown("#### 🧪 Top Compound Suggestions (Aggregated)")
            if not all_compound_suggestions:
                st.warning("No matching compounds found in the loaded database with the given tolerance and minimum peak matches for any spectrum.")
                st.info("Consider adjusting the 'tolerance' or 'min_matches' parameters in the MolecularIdentifier class if you expect matches.")
            else:
                agg_suggestions_list = []
                for compound, details in all_compound_suggestions.items():
                    agg_suggestions_list.append({
                        "Compound": compound,
                        "Group": details['Group'],
                        "Total Matched Peaks": details['Total Matched Peaks'],
                        "Occurrences Across Spectra": details['Occurrences'],
                        "Source Spectra": ", ".join(details['Source Spectra'])
                    })
                
                df_agg_match = pd.DataFrame(agg_suggestions_list).sort_values(
                    by=["Occurrences Across Spectra", "Total Matched Peaks"], 
                    ascending=[False, False]
                ).reset_index(drop=True)
                st.dataframe(df_agg_match, use_container_width=True, hide_index=True)

                if not df_agg_match.empty:
                    top_match_compound = df_agg_match.iloc[0]["Compound"]
                    top_match_group = df_agg_match.iloc[0]["Group"]
                    st.markdown("#### 🧠 AI-Generated Summary for Highest Confidence Compound")
                    if analyzer.ai_generator_model:
                        with st.spinner(f"Generating AI summary for {top_match_compound} using gemini-2.0-flash..."):
                            ai_summary = analyzer.generate_summary(top_match_compound, top_match_group)
                        st.markdown(ai_summary)
                    else:
                        st.warning("AI summary generation skipped because the model could not be loaded.")

                if not df_agg_match.empty:
                    st.markdown("#### 🌐 Fetch Details from PubChem (Highest Confidence Compound)")
                    if st.button(f"Get PubChem Details for {top_match_compound}"):
                        pubchem_info = fetch_pubchem_data(top_match_compound)
                        if "error" in pubchem_info:
                            st.error(pubchem_info["error"])
                        else:
                            st.subheader(f"PubChem Details for {top_match_compound} (CID: {pubchem_info.get('cid', 'N/A')})")
                            st.json(pubchem_info['properties'])
                            st.markdown(f"**Description:** {pubchem_info['description']}")
    else:
        st.info("Please upload one or more Raman spectrum CSV files to begin analysis.")
        st.markdown("""
        **Getting Started:**
        1. Upload a CSV file with Raman Wavenumbers in the first column and Intensities in the second (or subsequent) columns.
        2. Adjust sample metadata and measurement parameters in the sidebar.
        3. The application will automatically process, analyze, and visualize your spectra.
        """)

if __name__ == "__main__":
    main()
