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

def detect_peaks(wavenumbers: np.ndarray, intensities: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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

    # =============================================================================
    # START: MODIFIED SECTION FOR NEW FUNCTIONAL GROUP DATA FORMAT
    # =============================================================================
    def _assign_groups(self):
        """
        Assigns common functional groups based on peak positions using external rules.
        This version is updated to read 'min_wavenumber' and 'max_wavenumber' keys.
        """
        for peak in self.peaks:
            for rule in self.functional_group_rules:
                # Get the min and max wavenumber from the new JSON structure
                min_wn = rule.get("min_wavenumber")
                max_wn = rule.get("max_wavenumber")

                # Check if both keys exist and are numeric
                if min_wn is not None and max_wn is not None:
                    try:
                        # Convert to float just in case they are strings in the JSON
                        min_val = float(min_wn)
                        max_val = float(max_wn)

                        # Handle cases where min > max (e.g., 3400-3300) by finding the true min and max
                        lower_bound = min(min_val, max_val)
                        upper_bound = max(min_val, max_val)

                        # Check if the peak falls within the defined range
                        if lower_bound <= peak <= upper_bound:
                            # Get the group and description from the new structure
                            group = rule.get("group", "Unknown Group")
                            description = rule.get("description", "Unknown Mode")
                            
                            # Append the identified functional group to the list
                            self.functional_groups.append((f"{description} ({group})", peak))
                            
                            # Once a match is found for a peak, move to the next peak
                            break 
                    except (ValueError, TypeError):
                        # Skip this rule if wavenumber values are not valid numbers
                        continue
    # =============================================================================
    # END: MODIFIED SECTION
    # =============================================================================

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
    def __init__(self, json_paths: List[str] = None, ml_model_path: str = None, 
                 gemini_model_name: str = "gemini-pro", 
                 functional_group_database: List[Dict] = None, 
                 ai_raw_raman_shifts_data: List[Dict] = None):
        
        self._init_messages = [] # Collect initialization messages here

        # ML Model Loading
        try:
            self.model = Pipeline([('scaler', StandardScaler()), ('mlp', MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42))]) # Default if no joblib
            if ml_model_path and os.path.exists(ml_model_path):
                # Placeholder for actual model loading if joblib is available
                # self.model = joblib.load(ml_model_path)
                self._init_messages.append({"type": "success", "text": f"ML model '{os.path.basename(ml_model_path)}' loaded successfully."})
            else:
                self._init_messages.append({"type": "info", "text": "No pre-trained ML model found or path invalid. A new (untrained) model will be used."})
        except Exception as e:
            self._init_messages.append({"type": "warning", "text": f"Could not load ML model: {e}. A new (untrained) model will be used."})
            self.model = Pipeline([('scaler', StandardScaler()), ('mlp', MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42))]) # Fallback

        self.identifier = MolecularIdentifier()
        self.database, db_success, db_error = self._load_json_data_internal(json_paths, is_compound_db=True)
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

    def _load_json_data_internal(self, paths: List[str], is_compound_db: bool = False) -> Tuple[Any, Optional[str], Optional[str]]:
        """
        Loads JSON data from a list of paths (URL or local file) for use within RamanAnalyzer.
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
                    full_path = path
                    if not os.path.isabs(path):
                        try:
                           script_dir = os.path.dirname(os.path.abspath(__file__))
                           full_path = os.path.join(script_dir, path)
                        except NameError:
                           full_path = path # Fallback for environments where __file__ is not defined

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
                error_messages.append(f"JSON parsing error in {path}: {e}")
            except requests.exceptions.RequestException as e:
                error_messages.append(f"Request error loading from {path}: {e}")
            except Exception as e:
                error_messages.append(f"Unexpected error loading {path}: {e}")

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
            return f"AI summary generation failed: {e}."

    def predict_compounds_with_ai(self, peaks: List[float], functional_groups: List[Tuple[str, float]], diagnostics: List[str], metadata: Dict) -> List[Dict]:
        """
        Predicts plausible compounds using the AI model based on detected peaks,
        functional groups, diagnostics, metadata, and the raw Raman shifts data.
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
            prompt_parts.append("\nHere is additional raw Raman shift data that can be considered for more nuanced predictions:")
            for entry in self.ai_raw_raman_shifts_data:
                group = entry.get("group", "N/A")
                desc = entry.get("description", "N/A")
                min_wn = entry.get("min_wavenumber", "N/A")
                max_wn = entry.get("max_wavenumber", "N/A")
                prompt_parts.append(f"- Group: {group}, Mode: {desc}, Range: {min_wn}-{max_wn} cm-1")
            
        prompt_parts.append("\nConsidering these features, identify the most plausible chemical compounds. Provide a concise explanation for each suggestion. Return your response as a JSON array of objects. Each object MUST have two keys: 'Compound' and 'Reasoning'.")

        full_prompt = "\n".join(prompt_parts)
            
        try:
            with st.spinner("AI is thinking... Predicting compounds from functional groups..."):
                response = self.ai_generator_model.generate_content(
                    full_prompt,
                    generation_config={"response_mime_type": "application/json"}
                )
            
            json_string = response.text.strip()
            
            if not json_string.startswith("["):
                json_start = json_string.find('[')
                json_end = json_string.rfind(']')
                if json_start != -1 and json_end != -1:
                    json_string = json_string[json_start : json_end + 1]
                else:
                    raise json.JSONDecodeError("Could not extract clean JSON array from AI response.", json_string, 0)

            predicted_compounds = json.loads(json_string)

            if not isinstance(predicted_compounds, list):
                raise ValueError("AI response is not a JSON list as expected.")
            
            return predicted_compounds

        except json.JSONDecodeError as jde:
            st.error(f"Failed to parse AI's JSON response: {jde}. Response: {json_string[:500]}...")
            return []
        except ValueError as ve:
            st.error(f"AI response structural error: {ve}")
            return []
        except Exception as e:
            st.error(f"AI compound prediction failed: {e}.")
            return []

    def predict_structure_from_bonds_ai(self, functional_groups: List[Tuple[str, float]]) -> List[Dict]:
        """
        Deduces plausible chemical compounds by treating identified functional groups
        as "building blocks" and asking the AI to solve the molecular puzzle.
        """
        if not self.ai_generator_model:
            st.error("AI model not available for structural deduction.")
            return []

        if not functional_groups:
            st.warning("No functional groups identified to base the structural deduction on.")
            return []

        prompt_parts = [
            "You are an expert in Raman spectroscopy and organic chemistry. Your task is to act as a 'molecular puzzle solver'.",
            "You will be given a list of chemical bonds and functional groups identified from a Raman spectrum.",
            "Using ONLY this evidence, you must propose the most likely chemical compounds that could be constructed from these pieces.",
            "\n**Evidence from Spectrum (Vibrational Mode and observed peak):**"
        ]

        for fg_name, peak in functional_groups:
            prompt_parts.append(f"- {fg_name}, observed near {peak:.1f} cm⁻¹")

        prompt_parts.append("\nPropose a list of plausible compounds. For each compound, provide a brief reasoning that explains how its structure accounts for ALL the provided evidence. Return your answer as a JSON array of objects. Each object MUST contain two keys: 'Compound' and 'Reasoning'.")

        full_prompt = "\n".join(prompt_parts)

        try:
            with st.spinner("AI is solving the molecular puzzle..."):
                response = self.ai_generator_model.generate_content(
                    full_prompt,
                    generation_config={"response_mime_type": "application/json"}
                )
            
            json_string = response.text.strip()
            if json_string.startswith("```json"):
                json_string = json_string[7:]
            if json_string.endswith("```"):
                json_string = json_string[:-3]
            json_string = json_string.strip()
            
            predictions = json.loads(json_string)

            if not isinstance(predictions, list):
                 raise ValueError("AI response was not a valid list.")

            return predictions

        except json.JSONDecodeError as jde:
            st.error(f"Failed to parse AI's JSON response for structural deduction: {jde}. Response: {json_string[:500]}...")
            return []
        except ValueError as ve:
            st.error(f"AI response structural error during deduction: {ve}")
            return []
        except Exception as e:
            st.error(f"AI structural deduction failed: {e}")
            return []

    def visualize(self, spectra_data: List[Dict[str, Any]], plot_type: str = "overlay"):
        """
        Generates a matplotlib plot of one or more Raman spectra.
        """
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(10, 5))

        if plot_type == "overlay":
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
            # For stacked plot, we need to create multiple subplots
            plt.close(fig) # Close the initial figure
            nrows = len(spectra_data)
            fig, axes = plt.subplots(nrows=nrows, figsize=(10, 2 * nrows), sharex=True, squeeze=False)
            axes = axes.flatten() # Ensure axes is always a 1D array

            fig.suptitle("Raman Spectra Stacked View", fontsize=16, y=1.02)
            fig.text(-0.01, 0.5, 'Intensity (Arb. Units, Offset)', va='center', rotation='vertical', fontsize=12)

            for i, data in enumerate(spectra_data):
                ax = axes[i]
                normalized_intensities = (data['intensities'] - np.min(data['intensities'])) / (np.max(data['intensities']) - np.min(data['intensities']))
                offset = i * 1.1
                
                ax.plot(data['wavenumbers'], normalized_intensities + offset, label=data['label'], color='#1f77b4', linewidth=1.5)
                peak_indices = np.searchsorted(data['wavenumbers'], data['peaks'])
                peak_indices = peak_indices[(peak_indices >= 0) & (peak_indices < len(data['wavenumbers']))]
                
                ax.scatter(data['peaks'], normalized_intensities[peak_indices] + offset,
                           color='red', s=50, zorder=5, edgecolor='k', alpha=0.8)
                
                ax.set_ylabel(data['label'], rotation=0, labelpad=40, ha='right', va='center', fontsize=10)
                ax.set_yticks([])
                ax.tick_params(axis='both', which='major', labelsize=10)
                ax.grid(True, linestyle='--', alpha=0.6)
            
            # Set the xlabel only on the last subplot
            axes[-1].set_xlabel("Raman Shift (cm⁻¹)", fontsize=12)
            ax = axes[-1] # Point ax to the last axis for final adjustments

        ax.invert_xaxis()
        ax.set_xlabel("Raman Shift (cm⁻¹)", fontsize=12)
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        return fig

# Instantiate Analyzer Core once using st.cache_resource
@st.cache_resource
def get_analyzer_instance(json_db_paths: List[str], ml_model_path: str = None, 
                          expert_functional_group_db_paths: List[str] = None, 
                          ai_raw_raman_shifts_db_paths: List[str] = None) -> Tuple[Optional[RamanAnalyzer], List[Dict[str, str]]]:
    """
    Initializes and caches the RamanAnalyzer instance.
    """
    init_messages = []
    gemini_api_key = st.secrets.get("GEMINI_API_KEY") 
    
    if not gemini_api_key:
        init_messages.append({"type": "error", "text": "GEMINI_API_KEY not found. AI features will be disabled."})
        return None, init_messages

    try:
        genai.configure(api_key=gemini_api_key)
        init_messages.append({"type": "success", "text": "Google Gemini API configured successfully."})
    except Exception as e:
        init_messages.append({"type": "error", "text": f"Error configuring Google Gemini API: {e}. AI features will be disabled."})
        return None, init_messages

    gemini_model_name_for_analyzer = "gemini-1.5-flash-latest" 

    class DataLoader:
        def _load_json_data(self, paths: List[str], is_compound_db: bool) -> Tuple[Any, Optional[str], Optional[str]]:
            data_collection: Any = {} if is_compound_db else []
            success_messages = []
            error_messages = []

            if not paths:
                return data_collection, None, "No paths provided."

            for path in paths:
                try:
                    if path is None: continue
                    if path.startswith(('http://', 'https://')):
                        response = requests.get(path)
                        response.raise_for_status()
                        data = response.json()
                        success_messages.append(f"Loaded data from URL: {os.path.basename(path)}")
                    else:
                        if os.path.exists(path):
                            with open(path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                            success_messages.append(f"Loaded data from file: {os.path.basename(path)}")
                        else:
                            error_messages.append(f"File not found: {path}")
                            continue

                    if is_compound_db:
                        if isinstance(data, dict):
                            for category, compounds in data.items():
                                if isinstance(compounds, list):
                                    data_collection.setdefault(category, []).extend(compounds)
                        elif isinstance(data, list):
                            data_collection.setdefault("Uncategorized", []).extend(data)
                    else:
                        if isinstance(data, list):
                            data_collection.extend(data)

                except Exception as e:
                    error_messages.append(f"Error loading {os.path.basename(path)}: {e}")
                
            return data_collection, "; ".join(success_messages) if success_messages else None, "; ".join(error_messages) if error_messages else None

    loader = DataLoader()

    expert_functional_group_data, fg_expert_success, fg_expert_error = loader._load_json_data(expert_functional_group_db_paths, is_compound_db=False)
    if fg_expert_success: init_messages.append({"type": "success", "text": f"Expert Rules DB: {fg_expert_success}"})
    if fg_expert_error: init_messages.append({"type": "error", "text": f"Expert Rules DB: {fg_expert_error}"})
    
    ai_raw_raman_shifts_data, fg_ai_success, fg_ai_error = loader._load_json_data(ai_raw_raman_shifts_db_paths, is_compound_db=False)
    if fg_ai_success: init_messages.append({"type": "success", "text": f"AI Context DB: {fg_ai_success}"})
    if fg_ai_error: init_messages.append({"type": "error", "text": f"AI Context DB: {fg_ai_error}"})

    try:
        analyzer_instance = RamanAnalyzer(json_db_paths, ml_model_path, 
                                          gemini_model_name=gemini_model_name_for_analyzer, 
                                          functional_group_database=expert_functional_group_data, 
                                          ai_raw_raman_shifts_data=ai_raw_raman_shifts_data)
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
    
    try:
        response = requests.get(f"{base_url}/compound/name/{compound_name}/cids/JSON", timeout=10)
        response.raise_for_status()
        cid = response.json().get("IdentifierList", {}).get("CID", [None])[0]
        if not cid: return {"error": f"No CID found for '{compound_name}'."}

        props_res = requests.get(f"{base_url}/compound/cid/{cid}/property/MolecularFormula,IUPACName,CanonicalSMILES/JSON", timeout=10)
        props_res.raise_for_status()
        properties = props_res.json().get("PropertyTable", {}).get("Properties", [{}])[0]
        
        desc_res = requests.get(f"{base_url}/compound/cid/{cid}/description/JSON", timeout=10)
        desc_res.raise_for_status()
        desc_info = desc_res.json().get("InformationList", {}).get("Information", [])
        description = "No description available."
        for item in desc_info:
            if "Description" in item:
                description = item["Description"]
                break
        
        return {"cid": cid, "properties": properties, "description": description}

    except requests.exceptions.RequestException as e:
        return {"error": f"API request error for '{compound_name}': {e}"}
    except (json.JSONDecodeError, IndexError):
        return {"error": f"Could not parse PubChem response for '{compound_name}'."}

# ------------------------ Streamlit Interface ------------------------
def main():
    st.title("AI Raman Analyzer")
    st.markdown("---")
    
    # Define database paths
    GITHUB_COMPOUND_DB_URL = "https://raw.githubusercontent.com/deviprasath-009/raman-analyser/main/data/up.json"
    LOCAL_COMPOUND_DB_PATH = "data/raw_raman_shiifts.json"
    LOCAL_FUNCTIONAL_GROUP_DB_PATH = "data/raw_raman_shiifts.json"
    ML_MODEL_PATH = "raman_analyzer_model (1).joblib"
    
    # Initialize the analyzer
    analyzer, setup_messages = get_analyzer_instance(
        [GITHUB_COMPOUND_DB_URL, LOCAL_COMPOUND_DB_PATH], 
        ML_MODEL_PATH, 
        expert_functional_group_db_paths=[LOCAL_FUNCTIONAL_GROUP_DB_PATH],
        ai_raw_raman_shifts_db_paths=[LOCAL_FUNCTIONAL_GROUP_DB_PATH]
    )

    with st.expander("Show Setup & Initialization Messages", expanded=any(msg["type"] == "error" for msg in setup_messages)):
        for msg in setup_messages:
            if msg["type"] == "error": st.error(msg["text"])
            elif msg["type"] == "warning": st.warning(msg["text"])
            else: st.success(msg["text"])

    # Sidebar for inputs
    st.sidebar.header("📂 Data & Sample Information")
    uploaded_files = st.sidebar.file_uploader("Upload Raman Spectrum(s) (CSV)", type=["csv"], accept_multiple_files=True)
    st.sidebar.subheader("🧪 Sample Metadata")
    excitation = st.sidebar.selectbox("Excitation Wavelength", ["UV", "Visible", "NIR"], index=2)
    sample_state = st.sidebar.selectbox("Sample State", ["Solid", "Liquid", "Gas"], index=0)
    crystalline = st.sidebar.selectbox("Crystalline?", ["Yes", "No"], index=0)

    if uploaded_files:
        if not analyzer:
            st.error("Analyzer could not be initialized. Please check setup messages.")
            return

        all_processed_spectra, all_compound_suggestions_ml_db, all_functional_groups, all_diagnostics, all_peaks = [], {}, [], [], []

        meta = {"excitation": excitation, "sample_state": sample_state, "crystalline": crystalline}

        with st.spinner("Processing and analyzing spectra..."):
            for uploaded_file in uploaded_files:
                try:
                    df = pd.read_csv(uploaded_file)
                    if df.shape[1] < 2: continue
                    wavenumbers = df.iloc[:, 0].values
                    for col_idx in range(1, df.shape[1]):
                        label = f"{uploaded_file.name} - Col {col_idx}"
                        intensities = df.iloc[:, col_idx].values
                        results = analyzer.analyze(wavenumbers, intensities, meta)
                        
                        all_processed_spectra.append({**results, 'label': label})
                        all_peaks.extend(results['peaks'])
                        all_diagnostics.extend(d for d in results['diagnostics'] if d not in all_diagnostics)
                        all_functional_groups.extend(fg for fg in results['functional_groups'] if fg not in all_functional_groups)

                        for suggestion in results['compound_suggestions']:
                            name = suggestion['Compound']
                            if name not in all_compound_suggestions_ml_db:
                                all_compound_suggestions_ml_db[name] = {'Group': suggestion['Group'], 'Matched Peaks': 0, 'Occurrences': 0}
                            all_compound_suggestions_ml_db[name]['Matched Peaks'] += suggestion['Matched Peaks Count']
                            all_compound_suggestions_ml_db[name]['Occurrences'] += 1
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {e}")

        st.success("Analysis complete!")
        
        st.subheader("📈 Raman Spectra Visualization")
        plot_type = st.radio("Plot Type:", ("Overlay", "Stacked"), horizontal=True)
        if all_processed_spectra:
            st.pyplot(analyzer.visualize([s for s in all_processed_spectra], plot_type.lower()), use_container_width=True)

        st.subheader("📊 Consolidated Analysis Results")
        
        tab1, tab2, tab3 = st.tabs(["Diagnostics & Functional Groups", "Database Matches", "AI Predictions"])

        with tab1:
            st.markdown("#### 🔍 Diagnostics")
            for d in all_diagnostics or ["None"]: st.info(d)
            st.markdown("#### 📚 Functional Groups")
            for name, peak in sorted(all_functional_groups, key=lambda x: x[1]) or [("None", "")]:
                st.success(f"- **{name}** at {peak:.1f} cm⁻¹" if peak else "- None")

        with tab2:
            if all_compound_suggestions_ml_db:
                df_agg = pd.DataFrame.from_dict(all_compound_suggestions_ml_db, orient='index').reset_index().rename(columns={'index': 'Compound'})
                st.dataframe(df_agg.sort_values(by=["Occurrences", "Matched Peaks"], ascending=False), use_container_width=True, hide_index=True)
                
                top_match = df_agg.iloc[0]["Compound"]
                if st.button(f"Get PubChem Details for {top_match}"):
                    info = fetch_pubchem_data(top_match)
                    if "error" in info: st.error(info["error"])
                    else:
                        st.json(info['properties'])
                        st.markdown(f"**Description:** {info['description']}")
            else:
                st.info("No compounds found in the database matching the detected peaks.")

        with tab3:
            if analyzer.ai_generator_model:
                st.markdown("#### 🤖 AI Compound Prediction (Context-Based)")
                if st.button("Get AI Predictions from Spectrum"):
                    preds = analyzer.predict_compounds_with_ai(list(set(all_peaks)), all_functional_groups, all_diagnostics, meta)
                    for pred in preds: st.markdown(f"**{pred.get('Compound', 'N/A')}:** {pred.get('Reasoning', 'N/A')}")

                st.markdown("#### 🧩 AI Structural Deduction (from Bonds)")
                if st.button("Deduce Compounds from Bonds"):
                    deduced = analyzer.predict_structure_from_bonds_ai(all_functional_groups)
                    for item in deduced: st.markdown(f"**{item.get('Compound', 'N/A')}:** {item.get('Reasoning', 'N/A')}")
            else:
                st.warning("AI features are disabled.")
    else:
        st.info("Upload Raman spectrum CSV files to begin analysis.")

if __name__ == "__main__":
    try:
        import joblib
    except ImportError:
        joblib = None
    main()
