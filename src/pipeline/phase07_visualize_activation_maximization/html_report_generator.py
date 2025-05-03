import logging
import os
from pathlib import Path
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

# ============================================
# HTML Report Generation with Tabs
# ============================================

def generate_html_report(
    visualization_results: List[Dict[str, Any]],
    report_path: Path,
    report_config: Dict,
    examples_found: bool
) -> None:
    """
    Generates an HTML report summarizing the activation maximization results,
    using a tabbed layout for examples inspired by Phase 06.

    Args:
        visualization_results: List of dicts (sorted by importance) with visualization info.
        report_path: Path to save the HTML report.
        report_config: Configuration dictionary for reporting.
        examples_found: Boolean indicating if real examples were found and saved.
    """
    logger.info(f"Generating HTML report with tabbed layout at {report_path}...")

    # --- CSS Styles (incorporating tab styles) ---
    css_styles = """
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, "Open Sans", "Helvetica Neue", sans-serif; margin: 20px; background-color: #f8f9fa; color: #343a40; }
        h1 { color: #212529; border-bottom: 2px solid #dee2e6; padding-bottom: 10px; margin-bottom: 25px; }
        .feature-container { 
            background-color: #ffffff; border: 1px solid #ced4da; 
            margin-bottom: 30px; padding: 25px; border-radius: 8px; 
            box-shadow: 0 4px 8px rgba(0,0,0,0.05); 
        }
        .feature-header { font-size: 1.4em; font-weight: 600; margin-bottom: 12px; color: #0056b3; }
        .details { margin-bottom: 25px; font-size: 1em; color: #495057; }
        .details span { font-weight: 600; color: #212529; }
        
        /* Main flex container for layout */
        .visualization-section { 
            display: flex; 
            gap: 25px; /* Space between synthetic and examples block */
            align-items: flex-start; 
        }
        
        /* Block for the single synthetic image */
        .synthetic-block { 
            text-align: center; 
            flex: 0 0 320px; /* Fixed width for synthetic image block */
            padding-right: 25px;
            border-right: 1px solid #eee;
        }
        .synthetic-block h3 {
             font-size: 1.1em; margin-bottom: 10px; color: #343a40; font-weight: 500;
        }
        .synthetic-block img {
            max-width: 100%; /* Fill the container width */
            height: auto; 
            border: 1px solid #adb5bd; 
            border-radius: 4px;
            margin-bottom: 5px; 
        }

        /* Block containing the two example columns */
        .example-columns-block {
            display: flex;
            flex-grow: 1; /* Take remaining space */
            gap: 20px; /* Space between the two columns */
            min-width: 0; 
        }

        /* Styling for each example column (High/Low) */
        .example-column {
            flex: 1; /* Each column takes half the space */
            min-width: 0; /* Allow shrinking */
        }
        .example-column h3 {
            font-size: 1.1em; margin-bottom: 10px; color: #343a40; font-weight: 500;
            text-align: center;
        }

        /* Tab Styles adapted from Phase 06 */
        .tabs {
            display: flex;
            flex-wrap: wrap; /* Allow tabs to wrap */
            margin-bottom: 0; /* Remove bottom margin as border is on content */
            padding-left: 0;
            list-style: none;
        }
        .tab-btn {
            padding: 8px 12px;
            cursor: pointer;
            background-color: #e9ecef;
            border: 1px solid #dee2e6;
            border-bottom: none; /* Bottom border on content */
            border-radius: 4px 4px 0 0;
            margin-right: 3px;
            margin-bottom: -1px; /* Overlap bottom border */
            font-size: 0.9em;
            color: #495057;
            transition: background-color 0.2s ease;
        }
        .tab-btn:hover {
             background-color: #ced4da;
        }
        .tab-btn.active {
            background-color: #ffffff;
            font-weight: 600;
            color: #0056b3;
            border-color: #dee2e6 #dee2e6 #ffffff; /* Make bottom border disappear */
        }
        .tab-content {
            display: none;
            padding: 15px;
            border: 1px solid #dee2e6;
            border-radius: 0 0 4px 4px; /* Rounded bottom corners */
            background-color: #ffffff;
            text-align: center; /* Center image in tab */
        }
        .tab-content.active {
            display: block;
        }
        .tab-content img {
            max-width: 100%; /* Image fills tab content width */
            height: auto;
            border: 1px solid #adb5bd;
            border-radius: 4px;
        }
        .activation-value { font-size: 0.85em; color: #6c757d; display: block; margin-top: 5px; } 
        .no-examples { font-style: italic; color: #6c757d; font-size: 0.9em; padding: 15px; }
    </style>
    """

    # --- HTML Start ---
    html_start = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Activation Maximization Report</title>
    {css_styles}
</head>
<body>
    <h1>Activation Maximization Report</h1>
"""
    # TODO: Add description from config if available

    html_content = ""
    report_dir = report_path.parent

    # --- Loop through Features ---
    for unit_index, unit_info in enumerate(visualization_results):
        # Make synthetic viz path relative
        try:
            viz_rel_path = Path(unit_info.get('visualization_path', '')).relative_to(report_dir)
        except (ValueError, TypeError):
             viz_rel_path = None

        # Unique ID prefix for this feature's tabs
        feature_id_prefix = f"feature-{unit_index}"

        html_content += f"""
    <div class="feature-container">
        <div class="feature-header">Feature {unit_info.get('overall_rank', 'N/A')}: Layer <span>{unit_info['layer_name']}</span>, Unit <span>{unit_info['unit_index']}</span></div>
        <div class="details">
            Importance Score (Mean Activation on Positive Class): <span>{unit_info.get('importance_score', 'N/A'):.4f}</span>
        </div>
        
        <div class="visualization-section">
            
            <div class="synthetic-block">
                <h3>Synthetic Visualization</h3>
                {f'<img src="{viz_rel_path}" alt="Synthetic Visualization">' if viz_rel_path else '<p class="no-examples">Not Available</p>'}
            </div>

            <div class="example-columns-block">
        """
        
        # --- High Activation Examples Column ---
        html_content += """
                <div class="example-column">
                    <h3>High Activation Examples</h3>
        """
        if examples_found:
            high_examples = unit_info.get('high_activation_examples', [])
            if high_examples:
                # Tabs
                html_content += f'<div class="tabs" id="{feature_id_prefix}-high-tabs">'
                for i in range(len(high_examples)):
                    active_class = "active" if i == 0 else ""
                    html_content += f'<div class="tab-btn {active_class}" onclick="switchTab(this, \'{feature_id_prefix}-high-tab-{i}\')">Ex {i+1}</div>'
                html_content += '</div>'
                # Tab Content
                for i, ex in enumerate(high_examples):
                    active_class = "active" if i == 0 else ""
                    try:
                        ex_rel_path = Path(ex['path']).relative_to(report_dir)
                        html_content += f'<div class="tab-content {active_class}" id="{feature_id_prefix}-high-tab-{i}">'
                        html_content += f'<img src="{ex_rel_path}" title="Activation: {ex["activation"]:.3f}">'
                        html_content += f'<span class="activation-value">Activation: {ex["activation"]:.3f}</span>'
                        html_content += '</div>'
                    except (ValueError, TypeError): continue 
            else:
                 html_content += '<p class="no-examples">None found/saved.</p>'
        else:
             html_content += '<p class="no-examples">Example finding disabled.</p>'
        html_content += """
                </div> <!-- end example-column (high) -->
        """

        # --- Low Activation Examples Column ---
        html_content += """
                <div class="example-column">
                    <h3>Low Activation Examples</h3>
        """
        if examples_found:
            low_examples = unit_info.get('low_activation_examples', [])
            if low_examples:
                # Tabs
                html_content += f'<div class="tabs" id="{feature_id_prefix}-low-tabs">'
                for i in range(len(low_examples)):
                    active_class = "active" if i == 0 else ""
                    html_content += f'<div class="tab-btn {active_class}" onclick="switchTab(this, \'{feature_id_prefix}-low-tab-{i}\')">Ex {i+1}</div>'
                html_content += '</div>'
                # Tab Content
                for i, ex in enumerate(low_examples):
                    active_class = "active" if i == 0 else ""
                    try:
                        ex_rel_path = Path(ex['path']).relative_to(report_dir)
                        html_content += f'<div class="tab-content {active_class}" id="{feature_id_prefix}-low-tab-{i}">'
                        html_content += f'<img src="{ex_rel_path}" title="Activation: {ex["activation"]:.3f}">'
                        html_content += f'<span class="activation-value">Activation: {ex["activation"]:.3f}</span>'
                        html_content += '</div>'
                    except (ValueError, TypeError): continue
            else:
                 html_content += '<p class="no-examples">None found/saved.</p>'
        else:
             html_content += '<p class="no-examples">Example finding disabled.</p>'
        html_content += """
                </div> <!-- end example-column (low) -->
        """
            
        html_content += """
            </div> <!-- end example-columns-block -->
        </div> <!-- end visualization-section -->
    </div> <!-- end feature-container -->
        """

    # --- JavaScript for Tabs ---
    js_script = """
        <script>
            function switchTab(tabBtn, tabId) {
                // Find the parent example-column which contains this set of tabs
                const parentColumn = tabBtn.closest('.example-column');
                if (!parentColumn) return; 

                // Remove active class from all tab buttons and contents within this column
                parentColumn.querySelectorAll('.tab-btn').forEach(btn => {
                    btn.classList.remove('active');
                });
                parentColumn.querySelectorAll('.tab-content').forEach(content => {
                    content.classList.remove('active');
                });
                
                // Add active class to clicked button and corresponding content
                tabBtn.classList.add('active');
                const contentToShow = parentColumn.querySelector('#' + tabId);
                if (contentToShow) {
                    contentToShow.classList.add('active');
                }
            }
        </script>
    """

    html_end = f"""
    {js_script}
</body>
</html>
"""

    # --- Write HTML File ---
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True) # Ensure dir exists
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_start + html_content + html_end)
        logger.info("HTML report generation complete.")
    except Exception as e:
        logger.error(f"Failed to write HTML report to {report_path}: {e}")
