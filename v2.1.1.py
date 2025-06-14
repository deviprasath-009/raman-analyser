# Combined AI Raman Analyzer with Expert Rules + ML + Database Matching + GPT Description + PubChem API
# Enhanced with Realistic Peak Deconvolution

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, medfilt
from scipy.optimize import curve_fit
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

def gaussian(x: np.ndarray, A: float, mu: float, sigma: float) -> np.ndarray:
    """Gaussian function for peak fitting."""
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

def deconvolve_peaks(wavenumbers: np.ndarray, intensities: np.ndarray, peak_indices: List[int]) -> List[Dict]:
    """Deconvolves overlapping peaks using Gaussian curve fitting."""
    deconvolved_peaks = []
    
    for peak_idx in peak_indices:
        # Extract region around peak (±20 cm⁻¹)
        window = 20
        mask = (wavenumbers > wavenumbers[peak_idx] - window) & \
               (wavenumbers < wavenumbers[peak_idx] + window)
        
        x_data = wavenumbers[mask]
        y_data = intensities[mask]
        
        try:
            # Initial parameters (amplitude, center, width)
            p0 = [intensities[peak_idx], wavenumbers[peak_idx], 5]
            
            # Set physical bounds (amplitude>0, width>0)
            bounds = ([0, wavenumbers[peak_idx]-10, 0.1], 
                      [np.inf, wavenumbers[peak_idx]+10, 50])
            
            # Fit Gaussian
            params, _ = curve_fit(gaussian, x_data, y_data, p0=p0, bounds=bounds)
            A, mu, sigma = params
            
            # Calculate goodness of fit (R²)
            y_pred = gaussian(x_data, A, mu, sigma)
            ss_res = np.sum((y_data - y_pred)**2)
            ss_tot = np.sum((y_data - np.mean(y_data))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            deconvolved_peaks.append({
                'position': mu,
                'amplitude': A,
                'fwhm': 2.355 * sigma,  # Convert to FWHM
                'r_squared': r_squared,
                'is_deconvolved': True
            })
        except RuntimeError:
            # Fit failed - keep original peak
            deconvolved_peaks.append({
                'position': wavenumbers[peak_idx],
                'amplitude': intensities[peak_idx],
                'fwhm': np.nan,
                'r_squared': 0,
                'is_deconvolved': False
            })
    
    return deconvolved_peaks

def detect_peaks(wavenumbers: np.ndarray, intensities: np.ndarray) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
    """Detects peaks in the spectrum and deconvolves overlapping peaks."""
    prominence = np.std(intensities) * 0.5
    peaks, _ = find_peaks(intensities, prominence=prominence, distance=10)
    peak_info = []
    
    # Identify close peaks for deconvolution (within 15 cm⁻¹)
    close_peak_groups = []
    sorted_peaks = np.sort(peaks)
    
    if len(sorted_peaks) > 1:
        current_group = [sorted_peaks[0]]
        for i in range(1, len(sorted_peaks)):
            if abs(wavenumbers[sorted_peaks[i]] - wavenumbers[sorted_peaks[i-1]]) < 15:
                current_group.append(sorted_peaks[i])
            else:
                if len(current_group) > 1:
                    close_peak_groups.append(current_group)
                current_group = [sorted_peaks[i]]
        
        if len(current_group) > 1:
            close_peak_groups.append(current_group)
    
    # Deconvolve close peaks
    deconvolved_peaks = []
    for group in close_peak_groups:
        deconv_results = deconvolve_peaks(wavenumbers, intensities, group)
        deconvolved_peaks.extend(deconv_results)
    
    # Process non-overlapping peaks
    non_overlapping_peaks = [p for p in peaks if not any(p in group for group in close_peak_groups)]
    for peak_idx in non_overlapping_peaks:
        peak_info.append({
            'position': wavenumbers[peak_idx],
            'amplitude': intensities[peak_idx],
            'fwhm': np.nan,  # Will be calculated later if needed
            'r_squared': 0,
            'is_deconvolved': False
        })
    
    # Combine all peaks
    peak_info.extend(deconvolved_peaks)
    
    # Sort peaks by position
    peak_info.sort(key=lambda x: x['position'])
    
    # Extract positions and amplitudes for compatibility
    peak_positions = np.array([p['position'] for p in peak_info])
    peak_amplitudes = np.array([p['amplitude'] for p in peak_info])
    
    return peak_info, peak_positions, peak_amplitudes

# ------------------------ Expert Interpreter ------------------------
class ExpertInterpreter:
    """Interprets Raman spectrum features based on predefined expert rules."""
    def __init__(self, peak_info: List[Dict], metadata: Dict, functional_group_rules: List[Dict]):
        self.peak_info = peak_info
        self.metadata = metadata
        self.functional_groups = []
        self.diagnostics = []
        self.functional_group_rules = functional_group_rules

    def interpret(self):
        """Runs the expert interpretation logic."""
        self._check_conditions()
        self._assign_groups()
        self._handle_complexities()
        return {
            "diagnostics": self.diagnostics,
            "functional_groups": self.functional_groups,
            "peak_info": self.peak_info
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
        for peak in self.peak_info:
            position = peak['position']
            for rule in self.functional_group_rules:
                wavenumber_range_str = rule.get("wavenumber_range_cm-1")
                if wavenumber_range_str:
                    try:
                        # Split the range string by '–' (en dash) or '-' (hyphen)
                        if '–' in wavenumber_range_str:
                            start_str, end_str = wavenumber_range_str.split('–')
                        elif '-' in wavenumber_range_str:
                            start_str, end_str = wavenumber_range_str.split('-')
                        else:  # Handle single peak values
                            start_str = end_str = wavenumber_range_str

                        range_start = float(start_str.strip())
                        range_end = float(end_str.strip())

                        if range_start <= position <= range_end:
                            self.functional_groups.append({
                                "group": f"{rule['vibrational_mode']} ({rule['compound_functionality']})",
                                "position": position,
                                "amplitude": peak['amplitude'],
                                "fwhm": peak['fwhm']
                            })
                            break  # Move to next observed peak
                    except ValueError:
                        # Handle cases where range parsing fails
                        pass

    def _handle_complexities(self):
        """Identifies more complex spectral features using deconvolution results."""
        # Peaks below 500 cm⁻¹
        if any(p['position'] < 500 for p in self.peak_info):
            self.diagnostics.append("⚠️ Peaks below 500 cm⁻¹ often suggest minerals or inorganic compounds.")
        
        # Broad peaks (FWHM > 25 cm⁻¹)
        broad_peaks = [p for p in self.peak_info if not np.isnan(p['fwhm']) and p['fwhm'] > 25]
        if broad_peaks:
            self.diagnostics.append("🔍 Broad peaks detected (FWHM >25 cm⁻¹) - may indicate amorphous materials or overlapping signals.")
        
        # Low-quality fits
        poor_fits = [p for p in self.peak_info if p['is_deconvolved'] and p['r_squared'] < 0.6]
        if poor_fits:
            self.diagnostics.append("⚠️ Low-quality peak fits detected - verify manual integration or adjust parameters.")

# ------------------------ Molecular Identifier ------------------------
class MolecularIdentifier:
    """Identifies potential compounds by matching detected peaks against a database."""
    def __init__(self, tolerance: float = 30, min_matches: int = 1):
        self.tolerance = tolerance
        self.min_matches = min_matches

    def identify(self, peak_positions: List[float], database: Dict) -> List[Dict]:
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
                        for obs_peak in peak_positions:
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
        
        self._init_messages = []  # Collect initialization messages here

        try:
            self.model = Pipeline([('scaler', StandardScaler()), ('mlp', MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42))])
            if ml_model_path and os.path.exists(ml_model_path):
                self._init_messages.append({"type": "info", "text": f"ML model loaded from '{os.path.basename(ml_model_path)}'."})
            else:
                 self._init_messages.append({"type": "info", "text": "Using default ML model."})
        except Exception as e:
            self._init_messages.append({"type": "warning", "text": f"Could not load ML model: {e}. Using fallback model."})
            self.model = Pipeline([('scaler', StandardScaler()), ('mlp', MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42))])

        self.identifier = MolecularIdentifier()
        self.database, db_success, db_error = self._load_json_data_internal(json_paths, is_compound_db=True)
        if db_success: self._init_messages.append({"type": "success", "text": db_success})
        if db_error: self._init_messages.append({"type": "error", "text": db_error})

        self.functional_group_database = functional_group_database if functional_group_database is not None else []
        self.ai_raw_raman_shifts_data = ai_raw_raman_shifts_data if ai_raw_raman_shifts_data is not None else []
            
        # Initialize Google Generative AI model
        try:
            self.ai_generator_model = genai.GenerativeModel(gemini_model_name)
        except Exception as e:
            self.ai_generator_model = None
            self._init_messages.append({"type": "error", "text": f"Error initializing Google Gemini model: {e}. AI features will not work."})

    def _load_json_data_internal(self, paths: List[str], is_compound_db: bool = False) -> Tuple[Any, Optional[str], Optional[str]]:
        """Loads JSON data from a list of paths."""
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
        peak_info, peak_positions, peak_amplitudes = detect_peaks(wavenumbers, intensities_despiked)

        interpreter = ExpertInterpreter(peak_info, metadata, self.functional_group_database)
        interpretation = interpreter.interpret()

        suggestions = self.identifier.identify(peak_positions.tolist(), self.database)
            
        return {
            "peak_info": interpretation["peak_info"],
            "peaks": peak_positions,
            "peak_intensities": peak_amplitudes,
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

    def predict_compounds_with_ai(self, peak_info: List[Dict], functional_groups: List[Dict], diagnostics: List[str], metadata: Dict) -> List[Dict]:
        """Predicts plausible compounds using the AI model based on detected peaks."""
        if not self.ai_generator_model:
            return []

        prompt_parts = [
            "Based on the following Raman spectral analysis:",
            f"Detected Peak Positions (cm⁻¹): {sorted([p['position'] for p in peak_info])}",
            "Identified Functional Groups (with peak positions and FWHM):"
        ]
        
        if functional_groups:
            for fg in functional_groups:
                prompt_parts.append(f"- {fg['group']} at {fg['position']:.1f} cm⁻¹ (FWHM: {fg.get('fwhm', 'N/A'):.1f} cm⁻¹)")
        else:
            prompt_parts.append("- None explicitly identified by expert rules.")
        
        if diagnostics:
            prompt_parts.append("Diagnostics/Observations:")
            for diag in diagnostics:
                prompt_parts.append(f"- {diag}")
            
        prompt_parts.append(f"Sample State: {metadata.get('sample_state')}, Crystalline: {metadata.get('crystalline')}, Excitation: {metadata.get('excitation')}.")
            
        if self.ai_raw_raman_shifts_data:
            prompt_parts.append("\nAdditional Raman shift data:")
            for entry in self.ai_raw_raman_shifts_data[:5]:  # Show first 5 entries to avoid excessive length
                range_str = entry.get("wavenumber_range_cm-1", "N/A")
                mode = entry.get("vibrational_mode", "N/A")
                functionality = entry.get("compound_functionality", "N/A")
                prompt_parts.append(f"- Range: {range_str}, Mode: {mode}, Functionality: {functionality}")
            
        prompt_parts.append("\nConsidering these features, identify the most plausible chemical compounds. For each compound, provide a concise explanation linking it to the provided peaks and functional groups.")
        prompt_parts.append("Return your response as a JSON array of objects. Each object MUST have two keys: 'Compound' (string) and 'Reasoning' (string).")

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
        """Generates a matplotlib plot of one or more Raman spectra."""
        plt.style.use('seaborn-v0_8-darkgrid')

        if plot_type == "overlay":
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.set_title("Raman Spectra Overlay", fontsize=16, pad=15)
            for i, data in enumerate(spectra_data):
                color = plt.cm.get_cmap('viridis', len(spectra_data))(i)
                ax.plot(data['wavenumbers'], data['intensities'], label=data['label'], color=color, linewidth=1.5)
                
                # Plot peaks using stored amplitude
                peak_positions = [p['position'] for p in data['peak_info']]
                peak_amplitudes = [p['amplitude'] for p in data['peak_info']]
                ax.scatter(peak_positions, peak_amplitudes,
                           color=color, s=50, zorder=5, edgecolor='k', alpha=0.8)
                
                # Annotate deconvolved peaks
                for peak in data['peak_info']:
                    if peak['is_deconvolved']:
                        ax.text(peak['position'], peak['amplitude']*1.05, f"{peak['position']:.1f}", 
                                fontsize=8, ha='center', color=color)
            
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
                
                # Plot peaks using stored amplitude
                peak_positions = [p['position'] for p in data['peak_info']]
                peak_amplitudes = [p['amplitude'] for p in data['peak_info']]
                ax.scatter(peak_positions, [amp + current_offset for amp in peak_amplitudes],
                           color='red', s=50, zorder=5, edgecolor='k', alpha=0.8)
                
                # Annotate deconvolved peaks
                for peak in data['peak_info']:
                    if peak['is_deconvolved']:
                        ax.text(peak['position'], peak['amplitude'] + current_offset + 0.05*offset_factor, 
                                f"{peak['position']:.1f}", fontsize=8, ha='center')
                
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
    """Initializes and caches the RamanAnalyzer instance."""
    init_messages = []
    analyzer_instance = None

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

    gemini_model_name_for_analyzer = "gemini-1.5-flash"  # Updated to more powerful model

    class DataLoader:
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

    loader = DataLoader()

    # Load functional group data for ExpertInterpreter
    expert_functional_group_data, fg_expert_success, fg_expert_error = loader._load_json_data(expert_functional_group_db_paths, is_compound_db=False)
    if fg_expert_success: init_messages.append({"type": "success", "text": fg_expert_success})
    if fg_expert_error: init_messages.append({"type": "error", "text": fg_expert_error})
    
    # Load raw Raman shifts data for AI prompt
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
    """Fetches chemical information for a compound from PubChem."""
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

    # 2. Get properties
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
    st.title("AI Raman Analyzer with Advanced Deconvolution")
    st.markdown("---")

    # Get current script directory for local file paths
    script_directory = os.path.dirname(os.path.abspath(__file__))
    
    # Define database paths
    GITHUB_COMPOUND_DB_URL = "https://raw.githubusercontent.com/deviprasath-009/raman-analyser/main/data/up.json"
    LOCAL_COMPOUND_DB_PATH = os.path.join(script_directory, "raman data 1 .json")
    LOCAL_FUNCTIONAL_GROUP_DB_PATH = os.path.join(script_directory, "data/raw_raman_shiifts.json")
    LOCAL_AI_RAMAN_SHIFTS_DB_PATH = os.path.join(script_directory, "data/raw_raman_shiifts.json")
    ML_MODEL_PATH = os.path.join(script_directory, "raman_analyzer_model (1).joblib")
    
    # List of paths for databases
    COMPOUND_DB_PATHS = [GITHUB_COMPOUND_DB_URL, LOCAL_COMPOUND_DB_PATH]
    EXPERT_FG_DB_PATHS = [LOCAL_FUNCTIONAL_GROUP_DB_PATH]
    AI_RAMAN_SHIFTS_DB_PATHS = [LOCAL_AI_RAMAN_SHIFTS_DB_PATH]

    # Initialize the analyzer
    analyzer, setup_messages = get_analyzer_instance(
        COMPOUND_DB_PATHS, 
        ML_MODEL_PATH, 
        expert_functional_group_db_paths=EXPERT_FG_DB_PATHS,
        ai_raw_raman_shifts_db_paths=AI_RAMAN_SHIFTS_DB_PATHS
    )

    # Display setup messages
    has_critical_messages = any(
        msg["type"] == "error" or msg["type"] == "warning"
        for msg in setup_messages
    )
    
    with st.expander("Show Setup & Initialization Messages", expanded=has_critical_messages):
        if setup_messages:
            for msg in setup_messages:
                if msg["type"] == "error":
                    st.error(msg["text"])
                elif msg["type"] == "warning":
                    st.warning(msg["text"])
                elif msg["type"] == "success":
                    st.success(msg["text"])
                elif msg["type"] == "info":
                    st.info(msg["text"])
        else:
            st.info("No specific setup messages to display.")

    st.sidebar.header("📂 Data & Sample Information")
    uploaded_files = st.sidebar.file_uploader(
        "Upload Raman Spectrum(s) (CSV)", 
        type=["csv"], 
        help="Upload one or more two-column CSV files (Wavenumber, Intensity)",
        accept_multiple_files=True
    )

    st.sidebar.subheader("🧪 Sample Metadata")
    excitation = st.sidebar.selectbox(
        "Excitation Wavelength", 
        ["UV", "Visible", "NIR"], 
        index=2
    )
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        sample_state = st.selectbox("Sample State", ["Solid", "Liquid", "Gas"], index=0)
    with col2:
        crystalline = st.selectbox("Crystalline?", ["Yes", "No"], index=0)
    
    col3, col4 = st.sidebar.columns(2)
    with col3:
        polarized = st.selectbox("Polarized?", ["Yes", "No"], index=1)
    with col4:
        sample_origin = st.text_input("Sample Origin", value="biological")

    st.sidebar.subheader("⚡ Measurement Parameters")
    laser_power = st.sidebar.slider("Laser Power (mW)", 1, 500, 50)
    integration_time = st.sidebar.slider("Integration Time (ms)", 10, 10000, 1000)

    if uploaded_files:
        if analyzer is None:
            st.error("Analyzer could not be initialized due to critical errors.")
            return

        all_processed_spectra = []
        all_compound_suggestions_ml_db = {}
        all_functional_groups = []
        all_diagnostics = []
        all_peak_info = []
        ai_prediction_inputs = []

        with st.status("Processing spectrum(s)...", expanded=True) as status:
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
                st.write(f"Processing file {file_idx + 1}: **{uploaded_file.name}**")
                try:
                    df = pd.read_csv(uploaded_file)
                    
                    if df.shape[1] < 2:
                        st.warning(f"Skipping {uploaded_file.name}: Requires at least two columns (Wavenumber, Intensity).")
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
                            'peak_info': results['peak_info'],
                            'label': intensity_label
                        })
                        
                        # Collect all peak information
                        all_peak_info.extend(results['peak_info'])
                        for diag in results['diagnostics']:
                            if diag not in all_diagnostics:
                                all_diagnostics.append(diag)
                        for fg in results['functional_groups']:
                            if fg not in all_functional_groups:
                                all_functional_groups.append(fg)
                        
                        for suggestion in results['compound_suggestions']:
                            compound_name = suggestion['Compound']
                            if compound_name not in all_compound_suggestions_ml_db:
                                all_compound_suggestions_ml_db[compound_name] = {
                                    'Group': suggestion['Group'],
                                    'Total Matched Peaks': suggestion['Matched Peaks Count'],
                                    'Occurrences': 1,
                                    'Source Spectra': [intensity_label]
                                }
                            else:
                                all_compound_suggestions_ml_db[compound_name]['Total Matched Peaks'] += suggestion['Matched Peaks Count']
                                all_compound_suggestions_ml_db[compound_name]['Occurrences'] += 1
                                if intensity_label not in all_compound_suggestions_ml_db[compound_name]['Source Spectra']:
                                    all_compound_suggestions_ml_db[compound_name]['Source Spectra'].append(intensity_label)
                        
                        ai_prediction_inputs.append({
                            "peak_info": results['peak_info'],
                            "functional_groups": results['functional_groups'],
                            "diagnostics": results['diagnostics'],
                            "metadata": meta,
                            "label": intensity_label
                        })

                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {e}")

            status.update(label="Analysis complete!", state="complete", expanded=False)
            st.success("Analysis completed for all spectra!")

            st.subheader("📈 Raman Spectra Visualization")
            plot_type = st.radio("Select Plot Type:", ("Overlay View", "Stacked View"), horizontal=True)
            
            if all_processed_spectra:
                st.pyplot(analyzer.visualize(all_processed_spectra, plot_type.split()[0].lower()), use_container_width=True)
            else:
                st.warning("No valid spectra for visualization.")
            st.markdown("---")

            st.subheader("📊 Consolidated Analysis Results")

            st.markdown("#### 🔍 Diagnostics from All Spectra")
            if all_diagnostics:
                for d in all_diagnostics:
                    st.info(f"- {d}")
            else:
                st.info("No diagnostics identified.")
            st.markdown("---")

            st.markdown("#### 📚 Functional Groups Identified")
            if all_functional_groups:
                # Sort by position
                sorted_functional_groups = sorted(all_functional_groups, key=lambda x: x['position'])
                for fg in sorted_functional_groups:
                    fwhm_info = f" | FWHM: {fg['fwhm']:.1f} cm⁻¹" if not np.isnan(fg['fwhm']) else ""
                    st.success(f"- **{fg['group']}** at {fg['position']:.1f} cm⁻¹{fwhm_info}")
            else:
                st.info("No functional groups detected.")
            st.markdown("---")

            st.markdown("#### 🧪 Top Compound Suggestions")
            if not all_compound_suggestions_ml_db:
                st.warning("No matching compounds found in database.")
            else:
                agg_suggestions_list = []
                for compound, details in all_compound_suggestions_ml_db.items():
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
                    top_match_compound_ml_db = df_agg_match.iloc[0]["Compound"]
                    top_match_group_ml_db = df_agg_match.iloc[0]["Group"]
                    
                    st.markdown("#### 🧠 AI-Generated Summary")
                    if analyzer.ai_generator_model:
                        with st.spinner(f"Generating AI summary for {top_match_compound_ml_db}..."):
                            ai_summary = analyzer.generate_summary(top_match_compound_ml_db, top_match_group_ml_db)
                        st.markdown(ai_summary)
                    else:
                        st.warning("AI model not available for summaries.")

                    st.markdown("#### 🌐 PubChem Details")
                    if st.button(f"Get PubChem Details for {top_match_compound_ml_db}"):
                        pubchem_info = fetch_pubchem_data(top_match_compound_ml_db)
                        if "error" in pubchem_info:
                            st.error(pubchem_info["error"])
                        else:
                            st.subheader(f"PubChem Details for {top_match_compound_ml_db} (CID: {pubchem_info.get('cid', 'N/A')})")
                            st.json(pubchem_info['properties'])
                            st.markdown(f"**Description:** {pubchem_info['description']}")
            
            st.markdown("---")
            st.markdown("#### 🤖 AI-Predicted Compounds")
            if analyzer.ai_generator_model:
                if ai_prediction_inputs:
                    unique_peaks_overall = list({p['position'] for input_data in ai_prediction_inputs for p in input_data['peak_info']})
                    unique_functional_groups_overall = list({fg['group'] for fg in all_functional_groups})
                    unique_diagnostics_overall = list(set(all_diagnostics))
                    
                    if st.button("Get AI Compound Predictions"):
                        ai_predictions = analyzer.predict_compounds_with_ai(
                            all_peak_info, 
                            all_functional_groups, 
                            all_diagnostics, 
                            meta
                        )
                        if ai_predictions:
                            st.write("AI suggests the following compounds:")
                            for i, pred in enumerate(ai_predictions):
                                st.markdown(f"**{i+1}. {pred.get('Compound', 'N/A')}**")
                                st.write(f"   Reasoning: {pred.get('Reasoning', 'No reasoning provided.')}")
                        else:
                            st.info("AI could not generate predictions.")
                else:
                    st.info("Upload spectra to enable AI predictions.")
            else:
                st.warning("AI features not available.")

    else:
        st.info("Please upload Raman spectrum CSV files to begin analysis.")
        st.markdown("""
        **Instructions:**
        1. Upload CSV file(s) with Wavenumbers in first column and Intensities in subsequent columns
        2. Adjust sample metadata in sidebar
        3. The application will process, analyze, and visualize your spectra
        """)

if __name__ == "__main__":
    try:
        import joblib
    except ImportError:
        st.error("The 'joblib' library is required. Please install it (`pip install joblib`).")
    
    main()
