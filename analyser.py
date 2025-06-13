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

def detect_peaks(wavenumbers: np.ndarray, intensities: np.ndarray) -> np.ndarray:
    """Detects peaks in the spectrum based on prominence."""
    prominence = np.std(intensities) * 0.5 # Prominence based on standard deviation
    peaks, _ = find_peaks(intensities, prominence=prominence, distance=10) # Minimum distance between peaks
    return wavenumbers[peaks], intensities[peaks]

# ------------------------ Expert Interpreter ------------------------
class ExpertInterpreter:
    """Interprets Raman spectrum features based on predefined expert rules."""
    # Modified __init__ to accept functional_group_rules
    def __init__(self, peaks: List[float], intensities: List[float], metadata: Dict, functional_group_rules: List[Dict]):
        self.peaks = peaks
        self.intensities = intensities
        self.metadata = metadata
        self.functional_groups = []
        self.diagnostics = []
        self.functional_group_rules = functional_group_rules # Store the loaded rules

    def interpret(self):
        """Runs the expert interpretation logic."""
        self._check_conditions()
        self._assign_groups() # This will now use the external rules
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
        """Assigns common functional groups based on peak positions using external rules."""
        for peak in self.peaks:
            for rule in self.functional_group_rules:
                # The functional group rules are in a list of dictionaries.
                # Each dictionary has "wavenumber_range_cm-1", "vibrational_mode", "compound_functionality".
                # We need to parse the range string.
                wavenumber_range_str = rule.get("wavenumber_range_cm-1")
                if wavenumber_range_str:
                    try:
                        # Split the range string by '–' (en dash) or '-' (hyphen)
                        if '–' in wavenumber_range_range_str:
                            start_str, end_str = wavenumber_range_str.split('–')
                        elif '-' in wavenumber_range_str:
                            start_str, end_str = wavenumber_range_str.split('-')
                        else: # Handle single peak values or specific points if needed
                            start_str = end_str = wavenumber_range_str

                        range_start = float(start_str.strip())
                        range_end = float(end_str.strip())

                        if range_start <= peak <= range_end:
                            self.functional_groups.append((f"{rule['vibrational_mode']} ({rule['compound_functionality']})", peak))
                            break # Move to next observed peak
                    except ValueError:
                        # Handle cases where range parsing fails (e.g., malformed strings)
                        # Optionally log a warning or error, but for simplicity, we'll just skip
                        pass

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
            st.error("❌ Database format error: Expected a dictionary of categories.")
            return matches

        for category, compounds in database.items():
            if not isinstance(compounds, list):
                continue  # skip malformed categories
            for compound in compounds:
                matched_count = 0
                for ref_peak in compound.get("Peaks", []):
                    ref_wavenumber = ref_peak.get("Wavenumber")
                    if ref_wavenumber is not None:
                        for obs_peak in peaks:
                            if abs(obs_peak - ref_wavenumber) <= self.tolerance:
                                matched_count += 1
                                break # Move to the next ref_peak once a match is found
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
    # Modified __init__ to accept both functional_group_database and ai_raw_raman_shifts_data
    def __init__(self, json_paths: List[str] = None, model_path: str = None, 
                 gemini_model_name: str = "gemini-pro", 
                 functional_group_database: List[Dict] = None, # For ExpertInterpreter
                 ai_raw_raman_shifts_data: List[Dict] = None): # For AI prompt
        
        self.model = Pipeline([('scaler', StandardScaler()), ('mlp', MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42))])
        if model_path and os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                st.toast(f"ML model loaded successfully from '{os.path.basename(model_path)}'.")
            except Exception as e:
                st.warning(f"Could not load ML model from '{os.path.basename(model_path)}': {e}. A new (untrained) model will be used.")
        else:
            st.toast("No pre-trained ML model found or path invalid. A new (untrained) model will be used.")

        self.identifier = MolecularIdentifier()
        self.database = self._load_json_data(json_paths, is_compound_db=True) # Load compound database
        self.functional_group_database = functional_group_database if functional_group_database is not None else [] # Store functional group rules for ExpertInterpreter
        self.ai_raw_raman_shifts_data = ai_raw_raman_shifts_data if ai_raw_raman_shifts_data is not None else [] # Store raw Raman shifts for AI prompt
        
        # Initialize Google Generative AI model using the passed model name
        try:
            self.ai_generator_model = genai.GenerativeModel(gemini_model_name)
        except Exception as e:
            self.ai_generator_model = None # Set to None if model loading fails

    def _load_json_data(self, paths: List[str], is_compound_db: bool = False) -> Any:
        """
        Loads JSON data from a list of paths (URL or local file).
        Can be used for compound database (dict) or functional group rules (list).
        """
        if is_compound_db:
            db = {} # Compound database is a dict
        else:
            db = [] # Functional group rules is a list

        if not paths:
            st.warning(f"No database paths provided for {'compound' if is_compound_db else 'functional group'} data. It will be empty.")
            return db

        for path in paths:
            try:
                if path is None: # Skip None paths
                    continue

                if path.startswith(('http://', 'https://')):
                    response = requests.get(path)
                    response.raise_for_status()
                    data = response.json()
                    #st.toast(f"✅ Loaded {'compound' if is_compound_db else 'functional group'} data from URL: {path}")
                else:
                    if not os.path.isabs(path):
                        script_dir = os.path.dirname(os.path.abspath(__file__))
                        full_path = os.path.join(script_dir, path)
                    else:
                        full_path = path

                    if os.path.exists(full_path):
                        with open(full_path, 'r', encoding='utf-8') as f: # Use utf-8 encoding
                            data = json.load(f)
                        st.toast(f"✅ Loaded {'compound' if is_compound_db else 'functional group'} data from file: {os.path.basename(path)}")
                    else:
                        st.warning(f"⚠️ {'Compound' if is_compound_db else 'Functional group'} file not found at: {full_path}")
                        continue

                # Handle different JSON structures based on expected type
                if is_compound_db:
                    if isinstance(data, dict):
                        for category, compounds in data.items():
                            if isinstance(compounds, list):
                                db.setdefault(category, []).extend(compounds)
                            else:
                                st.warning(f"⚠️ Category '{category}' in {path} is not a list, skipping.")
                    elif isinstance(data, list):
                        db.setdefault("Uncategorized", []).extend(data)
                        st.warning(f"⚠️ JSON file {os.path.basename(path)} is a list. Wrapped under 'Uncategorized'.")
                    else:
                        st.error(f"❌ Unsupported JSON structure in {path} for compound database. Skipped.")
                else: # For functional group rules (expected to be a list)
                    if isinstance(data, list):
                        db.extend(data)
                    else:
                        st.error(f"❌ Functional group data in {path} is not a list. Skipped.")

            except json.JSONDecodeError as e:
                st.error(f"❌ JSON parsing error in {path} for {'compound' if is_compound_db else 'functional group'} data: {e}")
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Request error loading from {path} for {'compound' if is_compound_db else 'functional group'} data: {e}")
            except Exception as e:
                st.error(f"❌ Unexpected error loading {path} for {'compound' if is_compound_db else 'functional group'} data: {e}")

        return db

    def analyze(self, wavenumbers: np.ndarray, intensities: np.ndarray, metadata: Dict) -> Dict:
        """Performs the full Raman analysis workflow."""
        intensities_despiked = despike_spectrum(intensities)
        peaks, peak_intensities = detect_peaks(wavenumbers, intensities_despiked)

        # Pass the functional_group_database (for expert rules) to ExpertInterpreter
        interpreter = ExpertInterpreter(peaks.tolist(), peak_intensities.tolist(), metadata, self.functional_group_database)
        interpretation = interpreter.interpret()

        suggestions = self.identifier.identify(peaks.tolist(), self.database)
        
        return {
            "peaks": peaks,
            "peak_intensities": peak_intensities, # Store detected peak intensities
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
            
            # Basic cleaning
            clean_summary = summary.replace(prompt, "").strip()
            # Truncate if too long and ends with a sentence, or just truncate and with ellipsis
            if len(clean_summary) > 150 and '.' in clean_summary[:150]:
                clean_summary = clean_summary.rsplit('.', 1)[0] + '.'
            elif len(clean_summary) > 150:
                clean_summary = clean_summary[:150] + "..."
            
            return clean_summary if clean_summary else "AI summary could not be generated clearly."
        except Exception as e:
            return f"AI summary generation failed: {e}. Please check your API key, model access, and prompt."

    def predict_compounds_with_ai(self, peaks: List[float], functional_groups: List[Tuple[str, float]], diagnostics: List[str], metadata: Dict) -> List[Dict]:
        """
        Predicts plausible compounds using the AI model based on detected peaks,
        functional groups, diagnostics, metadata, and the raw Raman shifts data.
        Requests a structured JSON response from the AI.
        """
        if not self.ai_generator_model:
            st.warning("AI model not available for compound prediction.")
            return []

        # Construct a detailed prompt for the AI
        prompt_parts = [
            "Based on the following Raman spectral analysis:",
            f"Detected Peak Positions (cm⁻¹): {sorted(peaks)}",
            "Identified Functional Groups (and their approximate peak positions):"
        ]
        if functional_groups:
            for fg, p in functional_groups:
                prompt_parts.append(f"- {fg} (at {p:.1f} cm⁻¹)")
        else:
            prompt_parts.append("- None explicitly identified by expert rules.")

        if diagnostics:
            prompt_parts.append("Diagnostics/Observations:")
            for diag in diagnostics:
                prompt_parts.append(f"- {diag}")
        
        # Add metadata for context, but keep it concise for the AI
        prompt_parts.append(f"Sample State: {metadata.get('sample_state')}, Crystalline: {metadata.get('crystalline')}, Excitation: {metadata.get('excitation')}.")
        
        # NEW: Inject the content of raw_raman_shifts_data into the prompt for AI's context
        if self.ai_raw_raman_shifts_data:
            prompt_parts.append("\nHere is additional raw Raman shift data (wavenumber range, vibrational mode, compound functionality) that can be considered for more nuanced predictions:")
            for entry in self.ai_raw_raman_shifts_data:
                range_str = entry.get("wavenumber_range_cm-1", "N/A")
                mode = entry.get("vibrational_mode", "N/A")
                functionality = entry.get("compound_functionality", "N/A")
                prompt_parts.append(f"- Range: {range_str}, Mode: {mode}, Functionality: {functionality}")
        
        prompt_parts.append("\nConsidering these features, identify the most plausible chemical compounds that could be present in this sample. For each compound, provide a concise explanation of why it is suggested, linking it to the provided peaks and functional groups. Focus on compounds strongly indicated by the data,also give biological, minerals , biochemical and chemical with different combination of  peak to detect mixture of compound in the sample.")
        prompt_parts.append("Return your response as a JSON array of objects. Each object MUST have two keys: 'Compound' (string, the name of the chemical compound) and 'Reasoning' (string, a brief explanation). Do not include any introductory or concluding text outside the JSON array.")

        full_prompt = "\n".join(prompt_parts)

        # Define the expected JSON schema for the AI response
        response_schema = {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "Compound": {"type": "STRING"},
                    "Reasoning": {"type": "STRING"}
                },
                "required": ["Compound", "Reasoning"]
            }
        }
        
        try:
            with st.spinner("AI is thinking... Predicting compounds from functional groups..."):
                response = self.ai_generator_model.generate_content(
                    full_prompt,
                    generation_config={"response_mime_type": "application/json", "response_schema": response_schema}
                )
            
            json_string = response.text.strip()
            
            # Robust JSON parsing for AI response
            if not json_string.startswith("[") or not json_string.endswith("]"):
                st.warning("AI response not clean JSON array. Attempting to extract JSON...")
                json_start = json_string.find('[')
                json_end = json_string.rfind(']')
                if json_start != -1 and json_end != -1:
                    json_string = json_string[json_start : json_end + 1]
                else:
                    raise json.JSONDecodeError("Could not extract clean JSON array from AI response.", json_string, 0)

            predicted_compounds = json.loads(json_string)

            # Basic validation of the parsed structure
            if not isinstance(predicted_compounds, list):
                raise ValueError("AI response is not a JSON list as expected.")
            for item in predicted_compounds:
                if not isinstance(item, dict) or "Compound" not in item or "Reasoning" not in item:
                    raise ValueError("AI response list items are not in expected {Compound, Reasoning} format.")
            
            return predicted_compounds

        except json.JSONDecodeError as jde:
            st.error(f"Failed to parse AI's JSON response: {jde}. Response: {json_string[:500]}...")
            return []
        except ValueError as ve:
            st.error(f"AI response structural error: {ve}")
            return []
        except Exception as e:
            st.error(f"AI compound prediction failed: {e}. Please check model access or prompt quality.")
            return []


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
                color = plt.cm.get_cmap('viridis', len(spectra_data))(i) # Get a distinct color
                ax.plot(data['wavenumbers'], data['intensities'], label=data['label'], color=color, linewidth=1.5)
                # Plot detected peaks
                peak_indices = np.searchsorted(data['wavenumbers'], data['peaks'])
                peak_indices = peak_indices[(peak_indices >= 0) & (peak_indices < len(data['wavenumbers']))]
                ax.scatter(data['peaks'], data['intensities'][peak_indices],
                           color=color, s=50, zorder=5, edgecolor='k', alpha=0.8)
            ax.legend(loc='best', fontsize=10)
            ax.set_ylabel("Intensity (Arb. Units)", fontsize=12)

        elif plot_type == "stacked":
            nrows = len(spectra_data)
            fig, axes = plt.subplots(nrows=nrows, figsize=(10, 2 * nrows), sharex=True)
            if nrows == 1: # Handle single subplot case
                axes = [axes]

            fig.suptitle("Raman Spectra Stacked View", fontsize=16, y=1.02) # Adjusted title position
            
            # Calculate offsets for better visibility in stacked plot
            max_intensity_overall = max(np.max(d['intensities']) for d in spectra_data)
            offset_factor = max_intensity_overall * 1.2 # Adjust this multiplier for more/less spacing

            for i, data in enumerate(spectra_data):
                ax = axes[i]
                current_offset = i * offset_factor # Simple linear offset
                
                # Apply offset to intensities for stacked view
                ax.plot(data['wavenumbers'], data['intensities'] + current_offset, 
                        label=data['label'], color='#1f77b4', linewidth=1.5)
                
                # Plot detected peaks with offset
                peak_indices = np.searchsorted(data['wavenumbers'], data['peaks'])
                peak_indices = peak_indices[(peak_indices >= 0) & (peak_indices < len(data['wavenumbers']))]
                ax.scatter(data['peaks'], data['intensities'][peak_indices] + current_offset,
                           color='red', s=50, zorder=5, edgecolor='k', alpha=0.8)
                
                ax.set_ylabel(data['label'], fontsize=10) # Label each subplot with its spectrum name
                ax.set_yticks([]) # Hide y-axis ticks for cleaner look in stacked view
                ax.tick_params(axis='both', which='major', labelsize=10)
                ax.grid(True, linestyle='--', alpha=0.6)
                
            # Only set common xlabel for the last subplot
            axes[-1].set_xlabel("Raman Shift (cm⁻¹)", fontsize=12)

        ax.invert_xaxis() # Invert x-axis for Raman spectra

        plt.tight_layout()
        plt.subplots_adjust(top=0.95) # Adjust spacing for suptitle
        return fig

# Instantiate Analyzer Core once using st.cache_resource
@st.cache_resource
def get_analyzer_instance(json_db_paths: List[str], ml_model_path: str = None, 
                          expert_functional_group_db_paths: List[str] = None, 
                          ai_raw_raman_shifts_db_paths: List[str] = None) -> RamanAnalyzer:
    """
    Initializes and caches the RamanAnalyzer instance.
    Handles API key retrieval and Gemini configuration here, preventing early Streamlit errors.
    """
    # --- API Key retrieval and Gemini configuration moved HERE ---
    gemini_api_key = st.secrets.get("GEMINI_API_KEY") 
    
    if not gemini_api_key:
        st.error("GEMINI_API_KEY not found. Please set it in Streamlit secrets (`.streamlit/secrets.toml`) or as an environment variable.")
        st.stop() # This will stop the app if the key is not found

    try:
        genai.configure(api_key=gemini_api_key)
        st.toast("Google Gemini API configured successfully.")
    except Exception as e:
        st.error(f"Error configuring Google Gemini API: {e}. AI summary generation will not work.")
        st.stop() # Critical error, stop app.

    # Define the model name here, after API key configuration
    gemini_model_name_for_analyzer = "gemini-2.0-flash" 

    # Use a dummy RamanAnalyzer instance temporarily to access _load_json_data
    # This is a bit of a hack but ensures _load_json_data is available and respects st.cache_resource
    temp_analyzer = RamanAnalyzer()
    
    # Load functional group data for ExpertInterpreter
    expert_functional_group_data = temp_analyzer._load_json_data(expert_functional_group_db_paths, is_compound_db=False)
    
    # Load raw Raman shifts data specifically for AI prompt
    ai_raw_raman_shifts_data = temp_analyzer._load_json_data(ai_raw_raman_shifts_db_paths, is_compound_db=False)
    
    # Now, initialize the actual RamanAnalyzer, passing both functional group databases
    return RamanAnalyzer(json_db_paths, ml_model_path, 
                         gemini_model_name=gemini_model_name_for_analyzer, 
                         functional_group_database=expert_functional_group_data, # For ExpertInterpreter
                         ai_raw_raman_shifts_data=ai_raw_raman_shifts_data) # For AI prompt


# --- PubChem API Integration ---
@st.cache_data(show_spinner="Fetching data from PubChem...")
def fetch_pubchem_data(compound_name: str) -> Dict[str, Any]:
    """Fetches chemical information for a compound from PubChem PUG-REST API."""
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    
    # 1. Get CID from compound name
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

    # 2. Get properties (MolecularFormula, IUPACName, CanonicalSMILES)
    properties_url = f"{base_url}/compound/cid/{cid}/property/MolecularFormula,IUPACName,CanonicalSMILES/JSON"
    properties = {}
    try:
        response = requests.get(properties_url, timeout=10)
        response.raise_for_status()
        prop_data = response.json()
        props_list = prop_data.get("PropertyTable", {}).get("Properties")
        if props_list:
            properties = props_list[0]
            properties.pop('CID', None) # Remove CID from properties if it's there
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

    # Get current script directory for local file paths
    script_directory = os.path.dirname(os.path.abspath(__file__))
    
    # Database configuration - use GitHub URL for remote access
    GITHUB_DB_URL = "https://raw.githubusercontent.com/deviprasath-009/raman-analyser/main/data/up.json"
    LOCAL_DB_PATH = os.path.join(script_directory, "raman data 1 .json")
    DB_JSON_PATHS = [GITHUB_DB_URL, LOCAL_DB_PATH] 

    # Functional Group Database for ExpertInterpreter (original functional_groups.json)
    # Reverting to the previous expert functional group data source
    EXPERT_FG_DB_JSON_PATHS = ["https://raw.githubusercontent.com/deviprasath-009/raman-analyser/refs/heads/main/data/raw_raman_shiifts.json"] 

    # Raw Raman Shifts data specifically for AI prompt (raw_raman_shiifts.json)
    GITHUB_RAW_RAMAN_SHIFTS_URL = "https://raw.githubusercontent.com/deviprasath-009/raman-analyser/refs/heads/main/data/raw_raman_shiifts.json"
    RAW_RAMAN_SHIFTS_JSON_PATHS = [GITHUB_RAW_RAMAN_SHIFTS_URL]

    # Model path for your ML model (e.g., trained MLP classifier)
    ML_MODEL_PATH = os.path.join(script_directory, "raman_mlp_model.joblib")
    
    # Initialize the analyzer instance, passing both functional group paths
    analyzer = get_analyzer_instance(DB_JSON_PATHS, ML_MODEL_PATH, 
                                     expert_functional_group_db_paths=EXPERT_FG_DB_JSON_PATHS,
                                     ai_raw_raman_shifts_db_paths=RAW_RAMAN_SHIFTS_JSON_PATHS)

    st.sidebar.header("📂 Data & Sample Information")

    uploaded_files = st.sidebar.file_uploader(
        "Upload Raman Spectrum(s) (CSV)", 
        type=["csv"], 
        help="Upload one or more two-column CSV files (Wavenumber, Intensity) OR single CSV with multiple intensity columns.",
        accept_multiple_files=True # <--- Allow multiple files
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
        all_compound_suggestions_db = {} 
        all_functional_groups = [] 
        all_diagnostics = [] 
        all_ai_predicted_compounds = [] 

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

                    # Handle multi-column CSVs
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

                        # Aggregate Diagnostics
                        for diag in results['diagnostics']:
                            if diag not in all_diagnostics:
                                all_diagnostics.append(diag)
                        
                        # Aggregate Functional Groups
                        for fg_name, fg_peak in results['functional_groups']:
                            # Ensure unique functional group entries based on name and rounded peak
                            found = False
                            for existing_fg_name, existing_fg_peak in all_functional_groups:
                                if existing_fg_name == fg_name and abs(existing_fg_peak - fg_peak) < 1.0: # small tolerance for peak
                                    found = True
                                    break
                            if not found:
                                all_functional_groups.append((fg_name, fg_peak))
                        
                        # Aggregate Database Compound Suggestions
                        for suggestion in results['compound_suggestions']:
                            compound_name = suggestion['Compound']
                            if compound_name not in all_compound_suggestions_db:
                                all_compound_suggestions_db[compound_name] = {
                                    'Group': suggestion['Group'],
                                    'Total Matched Peaks': suggestion['Matched Peaks Count'],
                                    'Occurrences': 1,
                                    'Source Spectra': [intensity_label]
                                }
                            else:
                                all_compound_suggestions_db[compound_name]['Total Matched Peaks'] += suggestion['Matched Peaks Count']
                                all_compound_suggestions_db[compound_name]['Occurrences'] += 1
                                if intensity_label not in all_compound_suggestions_db[compound_name]['Source Spectra']:
                                    all_compound_suggestions_db[compound_name]['Source Spectra'].append(intensity_label)

                        # --- NEW: AI Compound Prediction for each spectrum ---
                        if analyzer.ai_generator_model:
                            status.write(f"AI predicting compounds for {intensity_label}...")
                            ai_predictions = analyzer.predict_compounds_with_ai(
                                results['peaks'].tolist(), # Pass peaks as list
                                results['functional_groups'],
                                results['diagnostics'],
                                meta
                            )
                            for pred in ai_predictions:
                                # Add source spectrum info to AI predictions as well
                                pred_with_source = pred.copy()
                                pred_with_source['Source Spectrum'] = intensity_label
                                all_ai_predicted_compounds.append(pred_with_source)
                        else:
                            st.warning("AI model not available for compound prediction.")

                except Exception as e:
                    st.error(f"An error occurred processing {uploaded_file.name}: {e}")
                    st.exception(e) # Show full traceback for debugging

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
                for name, peak in sorted(all_functional_groups, key=lambda x: x[1]): # Sort by peak for readability
                    st.success(f"- **{name}** at {peak:.1f} cm⁻¹") 
            else:
                st.info("No common functional groups detected across all spectra based on expert rules.")
            st.markdown("---")

            st.markdown("#### 🧪 Top Compound Suggestions from Database (Aggregated)")
            if not all_compound_suggestions_db:
                st.warning("No matching compounds found in the loaded database with the given tolerance and minimum peak matches for any spectrum.")
                st.info("Consider adjusting the 'tolerance' or 'min_matches' parameters in the MolecularIdentifier class if you expect matches.")
            else:
                # Convert aggregated database suggestions to a DataFrame for display
                agg_db_suggestions_list = []
                for compound, details in all_compound_suggestions_db.items():
                    agg_db_suggestions_list.append({
                        "Compound": compound,
                        "Group": details['Group'],
                        "Total Matched Peaks": details['Total Matched Peaks'],
                        "Occurrences Across Spectra": details['Occurrences'],
                        "Source Spectra": ", ".join(details['Source Spectra'])
                    })
                
                # Sort by occurrences and then total matched peaks for higher confidence first
                df_agg_db_match = pd.DataFrame(agg_db_suggestions_list).sort_values(
                    by=["Occurrences Across Spectra", "Total Matched Peaks"], 
                    ascending=[False, False]
                ).reset_index(drop=True)
                st.dataframe(df_agg_db_match, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.markdown("#### 🧠 AI-Predicted Compounds (Based on Functional Groups)")
            if not all_ai_predicted_compounds:
                st.info("No compounds predicted by AI for the processed spectra.")
            else:
                # Convert AI predicted compounds to DataFrame for display
                df_ai_predictions = pd.DataFrame(all_ai_predicted_compounds)
                st.dataframe(df_ai_predictions, use_container_width=True, hide_index=True)

            st.markdown("---")

            # AI Summary for the highest confidence compound from DB, or from AI if DB is empty
            if not all_compound_suggestions_db and not all_ai_predicted_compounds:
                st.info("No compound suggestions (database or AI) available for summary/PubChem lookup.")
            else:
                top_match_compound_summary = None
                top_match_group_summary = None
                summary_source_type = None

                # Determine the top compound for summary based on which source has data
                if not df_agg_db_match.empty: # Prefer DB top match for detailed summary
                    top_match_compound_summary = df_agg_db_match.iloc[0]["Compound"]
                    top_match_group_summary = df_agg_db_match.iloc[0]["Group"]
                    summary_source_type = "Database"
                elif not df_ai_predictions.empty: # Fallback to AI top match if no DB matches
                    top_match_compound_summary = df_ai_predictions.iloc[0]["Compound"]
                    top_match_group_summary = "AI Prediction" 
                    summary_source_type = "AI"
                
                if top_match_compound_summary:
                    st.markdown(f"#### 🧠 AI-Generated Summary for Top Compound ({summary_source_type})")
                    if analyzer.ai_generator_model:
                        with st.spinner(f"Generating AI summary for {top_match_compound_summary} using gemini-2.0-flash..."):
                            ai_summary = analyzer.generate_summary(top_match_compound_summary, top_match_group_summary)
                        st.markdown(ai_summary)
                    else:
                        st.warning("AI summary generation skipped because the model could not be loaded.")

                    # PubChem Details for the highest confidence compound
                    st.markdown(f"#### 🌐 Fetch Details from PubChem (Top Compound from {summary_source_type})")
                    if st.button(f"Get PubChem Details for {top_match_compound_summary}"):
                        pubchem_info = fetch_pubchem_data(top_match_compound_summary)
                        if "error" in pubchem_info:
                            st.error(pubchem_info["error"])
                        else:
                            with st.expander(f"**Details for {top_match_compound_summary} (CID: {pubchem_info.get('cid', 'N/A')})**", expanded=True):
                                st.markdown(f"**Molecular Formula:** {pubchem_info['properties'].get('MolecularFormula', 'N/A')}")
                                st.markdown(f"**IUPAC Name:** {pubchem_info['properties'].get('IUPACName', 'N/A')}")
                                st.markdown(f"**Canonical SMILES:** `{pubchem_info['properties'].get('CanonicalSMILES', 'N/A')}`")
                                st.markdown("---")
                                st.markdown("**Description:**")
                                st.write(pubchem_info.get('description', 'No detailed description available.'))
                                st.markdown(f"[View on PubChem](https://pubchem.ncbi.nlm.nih.gov/compound/{pubchem_info.get('cid')})")
                    st.markdown("---")


    else:
        st.info("Please upload one or more Raman spectrum CSV files to begin analysis.")
        st.markdown("""
        **Getting Started:**
        1.  **Upload your Raman spectrum data** as one or more CSV files. Each CSV can have:
            * Two columns (Wavenumber, Intensity) for a single spectrum.
            * Multiple columns (1st column Wavenumber, subsequent columns as individual Intensity spectra).
        2.  **Adjust the sample metadata** and measurement parameters in the sidebar. These apply to all uploaded spectra.
        3.  The analysis results, including functional groups, compound suggestions, and an AI summary, will appear here.
        """)
        st.markdown("---")
        st.subheader("App Features:")
        st.markdown("""
        -   **Multi-Spectrum Analysis:** Upload multiple CSVs or a single CSV with multiple intensity columns.
        -   **Flexible Visualization:** Choose between **Overlay** and **Stacked** plots for your spectra.
        -   **Automated Pre-processing:** Despiking and peak detection for each spectrum.
        -   **Expert Rules (Externalized):** Functional group identification rules are now loaded from an external JSON file for easier updates.
        -   **Database Matching:** Suggests compounds by comparing your spectrum's peaks to a custom database, with aggregated confidence.
        -   **AI-Powered Compound Prediction:** Utilizes Google Gemini to predict compounds based on detected functional groups and other spectral features.
        -   **AI-Powered Summary:** Generates concise descriptions of the highest confidence suggested compound using **Google Gemini API**.
        -   **PubChem Integration:** Fetches additional chemical details (formula, IUPAC name, SMILES, description) from the PubChem database for the highest confidence compound.
        """)

if __name__ == "__main__":
    main()
