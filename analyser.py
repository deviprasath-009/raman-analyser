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
import os
import requests
from typing import List, Dict, Any, Tuple, Optional

# Import for Google Generative AI
import google.generativeai as genai

# --- SET PAGE CONFIG FIRST ---
# This line MUST be the very first Streamlit command in your script.
st.set_page_config(page_title="AI Raman Analyzer", layout="wide", initial_sidebar_state="expanded")

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
                wavenumber_range_str = rule.get("wavenumber_range_cm-1")
                if wavenumber_range_str:
                    try:
                        # Split the range string by '–' (en dash) or '-' (hyphen)
                        if '–' in wavenumber_range_str:
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
            return matches

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
                 gemini_model_name: str = "gemini-pro", 
                 functional_group_database: List[Dict] = None, 
                 ai_raw_raman_shifts_data: List[Dict] = None):
        
        self._init_messages = [] # Collect initialization messages here

        self.model = Pipeline([('scaler', StandardScaler()), ('mlp', MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42))])
        if model_path and os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                self._init_messages.append({"type": "success", "text": f"ML model loaded successfully from '{os.path.basename(model_path)}'."})
            except Exception as e:
                self._init_messages.append({"type": "warning", "text": f"Could not load ML model from '{os.path.basename(model_path)}': {e}. A new (untrained) model will be used."})
        else:
            self._init_messages.append({"type": "info", "text": "No pre-trained ML model found or path invalid. A new (untrained) model will be used."})

        self.identifier = MolecularIdentifier()
        # _load_json_data now returns (data, success_msg, error_msg)
        self.database, db_success, db_error = self._load_json_data(json_paths, is_compound_db=True)
        if db_success:
            self._init_messages.append({"type": "success", "text": db_success})
        if db_error:
            self._init_messages.append({"type": "error", "text": db_error})

        self.functional_group_database = functional_group_database if functional_group_database is not None else []
        self.ai_raw_raman_shifts_data = ai_raw_raman_shifts_data if ai_raw_raman_shifts_data is not None else []
        
        # Initialize Google Generative AI model
        try:
            self.ai_generator_model = genai.GenerativeModel(gemini_model_name)
        except Exception as e:
            self.ai_generator_model = None
            self._init_messages.append({"type": "error", "text": f"Error initializing Google Gemini model: {e}. AI features will not work."})

    def _load_json_data(self, paths: List[str], is_compound_db: bool = False) -> Tuple[Any, Optional[str], Optional[str]]:
        """
        Loads JSON data from a list of paths (URL or local file).
        Returns (data, success_message, error_message).
        """
        data_collection: Any = {} if is_compound_db else []
        success_messages = []
        error_messages = []

        if not paths:
            error_messages.append(f"No database paths provided for {'compound' if is_compound_db else 'functional group'} data.")
            return data_collection, None, "; ".join(error_messages)

        for path in paths:
            if path is None:
                continue

            try:
                if path.startswith(('http://', 'https://')):
                    response = requests.get(path)
                    response.raise_for_status()
                    data = response.json()
                    success_messages.append(f"Loaded {'compound' if is_compound_db else 'functional group'} data from URL: {path}")
                else:
                    if not os.path.isabs(path):
                        script_dir = os.path.dirname(os.path.abspath(__file__))
                        full_path = os.path.join(script_dir, path)
                    else:
                        full_path = path

                    if os.path.exists(full_path):
                        with open(full_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        success_messages.append(f"Loaded {'compound' if is_compound_db else 'functional group'} data from file: {os.path.basename(path)}")
                    else:
                        error_messages.append(f"{'Compound' if is_compound_db else 'Functional group'} file not found at: {full_path}")
                        continue

                if is_compound_db:
                    if isinstance(data, dict):
                        for category, compounds in data.items():
                            if isinstance(compounds, list):
                                data_collection.setdefault(category, []).extend(compounds)
                            else:
                                error_messages.append(f"Category '{category}' in {path} is not a list, skipping.")
                    elif isinstance(data, list):
                        data_collection.setdefault("Uncategorized", []).extend(data)
                        error_messages.append(f"JSON file {os.path.basename(path)} is a list. Wrapped under 'Uncategorized'.")
                    else:
                        error_messages.append(f"Unsupported JSON structure in {path} for compound database. Skipped.")
                else:
                    if isinstance(data, list):
                        data_collection.extend(data)
                    else:
                        error_messages.append(f"Functional group data in {path} is not a list. Skipped.")

            except json.JSONDecodeError as e:
                error_messages.append(f"JSON parsing error in {path} for {'compound' if is_compound_db else 'functional group'} data: {e}")
            except requests.exceptions.RequestException as e:
                error_messages.append(f"Request error loading from {path} for {'compound' if is_compound_db else 'functional group'} data: {e}")
            except Exception as e:
                error_messages.append(f"Unexpected error loading {path} for {'compound' if is_compound_db else 'functional group'} data: {e}")

        return data_collection, "; ".join(success_messages) if success_messages else None, "; ".join(error_messages) if error_messages else None

    def get_init_messages(self) -> List[Dict[str, str]]:
        """Returns collected initialization messages."""
        return self._init_messages

    def analyze(self, wavenumbers: np.ndarray, intensities: np.ndarray, metadata: Dict) -> Dict:
        """Performs the full Raman analysis workflow."""
        intensities_despiked = despike_spectrum(intensities)
        peaks, peak_intensities = detect_peaks(wavenumbers, intensities_despiked)

        interpreter = ExpertInterpreter(peaks.tolist(), peak_intensities.tolist(), metadata, self.functional_group_database)
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

    def predict_compounds_with_ai(self, peaks: List[float], functional_groups: List[Tuple[str, float]], diagnostics: List[str], metadata: Dict) -> List[Dict]:
        """
        Predicts plausible compounds using the AI model based on detected peaks,
        functional groups, diagnostics, metadata, and the raw Raman shifts data.
        Requests a structured JSON response from the AI.
        """
        if not self.ai_generator_model:
            return []

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
        
        prompt_parts.append(f"Sample State: {metadata.get('sample_state')}, Crystalline: {metadata.get('crystalline')}, Excitation: {metadata.get('excitation')}.")
        
        if self.ai_raw_raman_shifts_data:
            prompt_parts.append("\nHere is additional raw Raman shift data (wavenumber range, vibrational mode, compound functionality) that can be considered for more nuanced predictions:")
            for entry in self.ai_raw_raman_shifts_data:
                range_str = entry.get("wavenumber_range_cm-1", "N/A")
                mode = entry.get("vibrational_mode", "N/A")
                functionality = entry.get("compound_functionality", "N/A")
                prompt_parts.append(f"- Range: {range_str}, Mode: {mode}, Functionality: {functionality}")
        
        prompt_parts.append("\nConsidering these features, identify the most plausible chemical compounds that could be present in this sample. For each compound, provide a concise explanation of why it is suggested, linking it to the provided peaks and functional groups. Focus on compounds strongly indicated by the data.")
        prompt_parts.append("Return your response as a JSON array of objects. Each object MUST have two keys: 'Compound' (string, the name of the chemical compound) and 'Reasoning' (string, a brief explanation). Do not include any introductory or concluding text outside the JSON array.")

        full_prompt = "\n".join(prompt_parts)

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
            
            if not json_string.startswith("[") or not json_string.endswith("]"):
                json_start = json_string.find('[')
                json_end = json_string.rfind(']')
                if json_start != -1 and json_end != -1:
                    json_string = json_string[json_start : json_end + 1]
                else:
                    raise json.JSONDecodeError("Could not extract clean JSON array from AI response.", json_string, 0)

            predicted_compounds = json.loads(json_string)

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

# Instantiate Analyzer Core once using st.cache_resource
@st.cache_resource
def get_analyzer_instance(json_db_paths: List[str], ml_model_path: str = None, 
                          expert_functional_group_db_paths: List[str] = None, 
                          ai_raw_raman_shifts_db_paths: List[str] = None) -> Tuple[Optional[RamanAnalyzer], List[Dict[str, str]]]:
    """
    Initializes and caches the RamanAnalyzer instance.
    Handles API key retrieval and Gemini configuration.
    Returns (analyzer_instance, init_messages).
    """
    init_messages = []
    analyzer_instance = None

    gemini_api_key = st.secrets.get("GEMINI_API_KEY") 
    
    if not gemini_api_key:
        init_messages.append({"type": "error", "text": "GEMINI_API_KEY not found. Please set it in Streamlit secrets (`.streamlit/secrets.toml`) or as an environment variable. AI features will be disabled."})
        return None, init_messages

    try:
        genai.configure(api_key=gemini_api_key)
        init_messages.append({"type": "success", "text": "Google Gemini API configured successfully."})
    except Exception as e:
        init_messages.append({"type": "error", "text": f"Error configuring Google Gemini API: {e}. AI features will be disabled."})
        return None, init_messages

    gemini_model_name_for_analyzer = "gemini-2.0-flash" 

    class DataLoader: # Renamed from DummyLoader for clarity
        def _load_json_data(self, paths: List[str], is_compound_db: bool) -> Tuple[Any, Optional[str], Optional[str]]:
            data_collection: Any = {} if is_compound_db else []
            success_messages = []
            error_messages = []

            if not paths:
                return data_collection, None, "No paths provided."

            for path in paths:
                try:
                    if path is None:
                        continue
                    if path.startswith(('http://', 'https://')):
                        response = requests.get(path)
                        response.raise_for_status()
                        data = response.json()
                        success_messages.append(f"Loaded {'compound' if is_compound_db else 'functional group'} data from URL: {os.path.basename(path)}")
                    else:
                        if not os.path.isabs(path):
                            script_dir = os.path.dirname(os.path.abspath(__file__))
                            full_path = os.path.join(script_dir, path)
                        else:
                            full_path = path

                        if os.path.exists(full_path):
                            with open(full_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                            success_messages.append(f"Loaded {'compound' if is_compound_db else 'functional group'} data from file: {os.path.basename(path)}")
                        else:
                            error_messages.append(f"File not found: {os.path.basename(path)}")
                            continue

                    if is_compound_db:
                        if isinstance(data, dict):
                            for category, compounds in data.items():
                                if isinstance(compounds, list):
                                    data_collection.setdefault(category, []).extend(compounds)
                                else:
                                    error_messages.append(f"Category '{category}' in {os.path.basename(path)} is not a list.")
                        elif isinstance(data, list):
                            data_collection.setdefault("Uncategorized", []).extend(data)
                            error_messages.append(f"JSON file {os.path.basename(path)} is a list. Wrapped under 'Uncategorized'.")
                        else:
                            error_messages.append(f"Unsupported JSON structure in {os.path.basename(path)}.")
                    else:
                        if isinstance(data, list):
                            data_collection.extend(data)
                        else:
                            error_messages.append(f"Functional group data in {os.path.basename(path)} is not a list. Skipped.")

                except json.JSONDecodeError as e:
                    error_messages.append(f"JSON parsing error in {os.path.basename(path)}: {e}")
                except requests.exceptions.RequestException as e:
                    error_messages.append(f"Request error loading from {os.path.basename(path)}: {e}")
                except Exception as e:
                    error_messages.append(f"Unexpected error loading {os.path.basename(path)}: {e}")
            
            return data_collection, "; ".join(success_messages) if success_messages else None, "; ".join(error_messages) if error_messages else None

    loader = DataLoader() # Create an instance of the data loader

    # Load functional group data for ExpertInterpreter
    expert_functional_group_data, fg_expert_success, fg_expert_error = loader._load_json_data(expert_functional_group_db_paths, is_compound_db=False)
    if fg_expert_success: init_messages.append({"type": "success", "text": fg_expert_success})
    if fg_expert_error: init_messages.append({"type": "error", "text": fg_expert_error})
    
    # Load raw Raman shifts data specifically for AI prompt
    ai_raw_raman_shifts_data, fg_ai_success, fg_ai_error = loader._load_json_data(ai_raw_raman_shifts_db_paths, is_compound_db=False)
    if fg_ai_success: init_messages.append({"type": "success", "text": fg_ai_success})
    if fg_ai_error: init_messages.append({"type": "error", "text": fg_ai_error})

    try:
        analyzer_instance = RamanAnalyzer(json_db_paths, ml_model_path, 
                                          gemini_model_name=gemini_model_name_for_analyzer, 
                                          functional_group_database=expert_functional_group_data, 
                                          ai_raw_raman_shifts_data=ai_raw_raman_shifts_data)
        # Append messages from RamanAnalyzer's __init__
        init_messages.extend(analyzer_instance.get_init_messages())
    except Exception as e:
        init_messages.append({"type": "error", "text": f"Failed to initialize RamanAnalyzer: {e}"})
        return None, init_messages

    return analyzer_instance, init_messages


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
    EXPERT_FG_DB_JSON_PATHS = ["https://raw.githubusercontent.com/deviprasath-009/raman-analyser/main/data/functional_groups.json"] 

    # Raw Raman Shifts data specifically for AI prompt (raw_raman_shiifts.json)
    GITHUB_RAW_RAMAN_SHIFTS_URL = "https://raw.githubusercontent.com/deviprasath-009/raman-analyser/refs/heads/main/data/raw_raman_shiifts.json"
    RAW_RAMAN_SHIFTS_JSON_PATHS = [GITHUB_RAW_RAMAN_SHIFTS_URL]

    # Model path for your ML model (e.g., trained MLP classifier)
    ML_MODEL_PATH = os.path.join(script_directory, "raman_mlp_model.joblib")
    
    # Initialize the analyzer instance and get initialization messages
    analyzer, init_messages = get_analyzer_instance(DB_JSON_PATHS, ML_MODEL_PATH, 
                                     expert_functional_group_db_paths=EXPERT_FG_DB_JSON_PATHS,
                                     ai_raw_raman_shifts_db_paths=RAW_RAMAN_SHIFTS_JSON_PATHS)

    # Display initialization messages at the bottom
    # These messages will now appear correctly without causing caching issues.
    for msg in init_messages:
        if msg["type"] == "success":
            st.toast(msg["text"], icon="✅")
        elif msg["type"] == "warning":
            st.warning(msg["text"])
        elif msg["type"] == "error":
            st.error(msg["text"])
        elif msg["type"] == "info":
            st.info(msg["text"])


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

    # Moving app description and getting started guide to the top
    st.info("Welcome to rudra's Raman Analyzer! This tool helps you analyze Raman spectra, identify compounds, and get AI-powered insights.")
    st.subheader("Getting Started:")
    st.markdown("""
    1.  **Upload your Raman spectrum data** as one or more CSV files using the uploader in the sidebar. Each CSV can have:
        * Two columns (Wavenumber, Intensity) for a single spectrum.
        * Multiple columns (1st column Wavenumber, subsequent columns as individual Intensity spectra).
    2.  **Adjust the sample metadata** and measurement parameters in the sidebar. These apply to all uploaded spectra.
    3.  The analysis results, including functional groups, compound suggestions, and AI insights, will appear below.
    """)
    st.markdown("---")
    st.subheader("App Features:")
    st.markdown("""
    -   **Multi-Spectrum Analysis:** Upload multiple CSVs or a single CSV with multiple intensity columns.
    -   **Flexible Visualization:** Choose between **Overlay** and **Stacked** plots for your spectra.
    -   **Automated Pre-processing:** Despiking and peak detection for each spectrum.
    -   **Expert Rules (Externalized):** Functional group identification rules are loaded from an external JSON file.
    -   **Database Matching:** Suggests compounds by comparing your spectrum's peaks to a custom database, with aggregated confidence.
    -   **AI-Powered Compound Prediction:** Utilizes Google Gemini to predict compounds based on detected functional groups and other spectral features, leveraging a separate raw Raman shifts dataset.
    -   **AI-Powered Summary:** Generates concise descriptions of the highest confidence suggested compound using **Google Gemini API**.
    -   **PubChem Integration:** Fetches additional chemical details (formula, IUPAC name, SMILES, description) from the PubChem database for the highest confidence compound.
    """)
    st.markdown("---") # Add a separator

    # Conditional display based on analyzer initialization success
    if analyzer is None:
        st.error("Raman Analyzer could not be initialized. Please check the error messages above for details.")
        st.stop() # Stop further execution if analyzer is not available

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
                            # This handles slight variations in peak detection for the same group
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
                                results['peaks'].tolist(),
                                results['functional_groups'],
                                results['diagnostics'],
                                meta
                            )
                            for pred in ai_predictions:
                                pred_with_source = pred.copy()
                                pred_with_source['Source Spectrum'] = intensity_label
                                all_ai_predicted_compounds.append(pred_with_source)
                        else:
                            st.warning("AI model not available for compound prediction.")

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

            st.markdown("#### 🧪 Top Compound Suggestions from Database (Aggregated)")
            if not all_compound_suggestions_db:
                st.warning("No matching compounds found in the loaded database with the given tolerance and minimum peak matches for any spectrum.")
                st.info("Consider adjusting the 'tolerance' or 'min_matches' parameters in the MolecularIdentifier class if you expect matches.")
            else:
                agg_db_suggestions_list = []
                for compound, details in all_compound_suggestions_db.items():
                    agg_db_suggestions_list.append({
                        "Compound": compound,
                        "Group": details['Group'],
                        "Total Matched Peaks": details['Total Matched Peaks'],
                        "Occurrences Across Spectra": details['Occurrences'],
                        "Source Spectra": ", ".join(details['Source Spectra'])
                    })
                
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
                df_ai_predictions = pd.DataFrame(all_ai_predicted_compounds)
                st.dataframe(df_ai_predictions, use_container_width=True, hide_index=True)

            st.markdown("---")

            if not all_compound_suggestions_db and not all_ai_predicted_compounds:
                st.info("No compound suggestions (database or AI) available for summary/PubChem lookup.")
            else:
                top_match_compound_summary = None
                top_match_group_summary = None
                summary_source_type = None

                if not df_agg_db_match.empty:
                    top_match_compound_summary = df_agg_db_match.iloc[0]["Compound"]
                    top_match_group_summary = df_agg_db_match.iloc[0]["Group"]
                    summary_source_type = "Database"
                elif not df_ai_predictions.empty:
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

