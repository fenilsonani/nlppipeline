"""Export NLP processing results to various formats."""

import json
import csv
import os
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import pandas as pd
from pathlib import Path
import html
from io import StringIO
import zipfile


class ResultExporter:
    """Export NLP results to CSV, JSON, HTML, and other formats."""
    
    def __init__(self, output_dir: str = "exports"):
        """
        Initialize result exporter.
        
        Args:
            output_dir: Directory to save exported files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def export_to_json(self, data: Union[Dict[str, Any], List[Dict[str, Any]]], 
                      filename: Optional[str] = None,
                      pretty: bool = True) -> str:
        """
        Export data to JSON format.
        
        Args:
            data: Data to export
            filename: Output filename (auto-generated if None)
            pretty: Whether to format JSON with indentation
            
        Returns:
            Path to exported file
        """
        if filename is None:
            filename = f"nlp_results_{self.timestamp}.json"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            if pretty:
                json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                json.dump(data, f, ensure_ascii=False)
        
        return str(filepath)
    
    def export_to_csv(self, results: List[Dict[str, Any]], 
                     filename: Optional[str] = None,
                     include_entities: bool = True) -> str:
        """
        Export results to CSV format.
        
        Args:
            results: List of processing results
            filename: Output filename (auto-generated if None)
            include_entities: Whether to include entity columns
            
        Returns:
            Path to exported file
        """
        if filename is None:
            filename = f"nlp_results_{self.timestamp}.csv"
        
        filepath = self.output_dir / filename
        
        # Prepare data for CSV
        csv_data = []
        for result in results:
            row = {
                "document_id": result.get("document_id", ""),
                "text": result.get("text", "")[:500],  # Limit text length
                "text_length": len(result.get("text", "")),
                "word_count": len(result.get("text", "").split())
            }
            
            # Add sentiment data
            if "sentiment" in result:
                row["sentiment_label"] = result["sentiment"].get("label", "")
                row["sentiment_score"] = result["sentiment"].get("score", 0.0)
            
            # Add classification data
            if "classification" in result:
                row["classification_label"] = result["classification"].get("label", "")
                row["classification_score"] = result["classification"].get("score", 0.0)
            
            # Add entity data
            if include_entities and "entities" in result:
                entity_texts = []
                entity_types = []
                for entity in result["entities"]:
                    entity_texts.append(entity.get("text", ""))
                    entity_types.append(entity.get("label", ""))
                
                row["entity_count"] = len(result["entities"])
                row["entities"] = "; ".join(entity_texts)
                row["entity_types"] = "; ".join(set(entity_types))
            
            # Add processing metadata
            row["processed_at"] = result.get("processed_at", "")
            row["processing_time"] = result.get("processing_time", 0.0)
            row["error"] = result.get("error", "")
            
            csv_data.append(row)
        
        # Write to CSV
        if csv_data:
            df = pd.DataFrame(csv_data)
            df.to_csv(filepath, index=False, encoding='utf-8')
        else:
            # Write empty CSV with headers
            with open(filepath, 'w', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["document_id", "text", "sentiment_label", "sentiment_score"])
        
        return str(filepath)
    
    def export_to_html(self, results: List[Dict[str, Any]], 
                      report_data: Optional[Dict[str, Any]] = None,
                      filename: Optional[str] = None,
                      include_visualizations: bool = True) -> str:
        """
        Export results to HTML report.
        
        Args:
            results: List of processing results
            report_data: Additional report data to include
            filename: Output filename (auto-generated if None)
            include_visualizations: Whether to include chart placeholders
            
        Returns:
            Path to exported file
        """
        if filename is None:
            filename = f"nlp_report_{self.timestamp}.html"
        
        filepath = self.output_dir / filename
        
        html_content = self._generate_html_report(results, report_data, include_visualizations)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(filepath)
    
    def export_entities_to_csv(self, results: List[Dict[str, Any]], 
                             filename: Optional[str] = None) -> str:
        """
        Export entities to separate CSV file.
        
        Args:
            results: List of processing results
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to exported file
        """
        if filename is None:
            filename = f"entities_{self.timestamp}.csv"
        
        filepath = self.output_dir / filename
        
        # Extract all entities
        entity_data = []
        for result in results:
            if "entities" in result:
                for entity in result["entities"]:
                    entity_data.append({
                        "document_id": result.get("document_id", ""),
                        "entity_text": entity.get("text", ""),
                        "entity_type": entity.get("label", ""),
                        "confidence": entity.get("score", 0.0),
                        "start_position": entity.get("start", -1),
                        "end_position": entity.get("end", -1)
                    })
        
        # Write to CSV
        if entity_data:
            df = pd.DataFrame(entity_data)
            df.to_csv(filepath, index=False, encoding='utf-8')
        else:
            # Write empty CSV with headers
            with open(filepath, 'w', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["document_id", "entity_text", "entity_type", 
                               "confidence", "start_position", "end_position"])
        
        return str(filepath)
    
    def export_summary_to_markdown(self, summary_data: Dict[str, Any], 
                                 filename: Optional[str] = None) -> str:
        """
        Export summary data to Markdown format.
        
        Args:
            summary_data: Summary statistics and insights
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to exported file
        """
        if filename is None:
            filename = f"summary_{self.timestamp}.md"
        
        filepath = self.output_dir / filename
        
        md_content = self._generate_markdown_summary(summary_data)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        return str(filepath)
    
    def export_batch(self, results: List[Dict[str, Any]], 
                    report_data: Optional[Dict[str, Any]] = None,
                    formats: List[str] = ["json", "csv", "html"],
                    archive: bool = True) -> Union[str, Dict[str, str]]:
        """
        Export results in multiple formats, optionally creating an archive.
        
        Args:
            results: List of processing results
            report_data: Additional report data
            formats: List of export formats
            archive: Whether to create a zip archive
            
        Returns:
            Path to archive or dictionary of exported files
        """
        exported_files = {}
        
        for format_type in formats:
            if format_type == "json":
                exported_files["json"] = self.export_to_json(results)
            elif format_type == "csv":
                exported_files["csv"] = self.export_to_csv(results)
                exported_files["entities_csv"] = self.export_entities_to_csv(results)
            elif format_type == "html":
                exported_files["html"] = self.export_to_html(results, report_data)
            elif format_type == "markdown" and report_data:
                exported_files["markdown"] = self.export_summary_to_markdown(report_data)
        
        if archive:
            archive_name = f"nlp_export_{self.timestamp}.zip"
            archive_path = self.output_dir / archive_name
            
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_type, file_path in exported_files.items():
                    zipf.write(file_path, os.path.basename(file_path))
            
            # Clean up individual files
            for file_path in exported_files.values():
                os.remove(file_path)
            
            return str(archive_path)
        
        return exported_files
    
    def export_to_excel(self, results: List[Dict[str, Any]], 
                       report_data: Optional[Dict[str, Any]] = None,
                       filename: Optional[str] = None) -> str:
        """
        Export results to Excel file with multiple sheets.
        
        Args:
            results: List of processing results
            report_data: Additional report data
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to exported file
        """
        if filename is None:
            filename = f"nlp_analysis_{self.timestamp}.xlsx"
        
        filepath = self.output_dir / filename
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Main results sheet
            main_data = []
            for result in results:
                row = {
                    "Document ID": result.get("document_id", ""),
                    "Text Preview": result.get("text", "")[:100] + "...",
                    "Sentiment": result.get("sentiment", {}).get("label", ""),
                    "Sentiment Score": result.get("sentiment", {}).get("score", 0.0),
                    "Entity Count": len(result.get("entities", [])),
                    "Word Count": len(result.get("text", "").split())
                }
                main_data.append(row)
            
            if main_data:
                df_main = pd.DataFrame(main_data)
                df_main.to_excel(writer, sheet_name='Results', index=False)
            
            # Entities sheet
            entity_data = []
            for result in results:
                if "entities" in result:
                    for entity in result["entities"]:
                        entity_data.append({
                            "Document ID": result.get("document_id", ""),
                            "Entity": entity.get("text", ""),
                            "Type": entity.get("label", ""),
                            "Confidence": entity.get("score", 0.0)
                        })
            
            if entity_data:
                df_entities = pd.DataFrame(entity_data)
                df_entities.to_excel(writer, sheet_name='Entities', index=False)
            
            # Summary sheet
            if report_data:
                summary_data = []
                if "metadata" in report_data:
                    summary_data.append(["Metric", "Value"])
                    summary_data.append(["Total Documents", report_data["metadata"].get("total_documents", 0)])
                    summary_data.append(["Report Generated", report_data["metadata"].get("report_generated", "")])
                
                if summary_data:
                    df_summary = pd.DataFrame(summary_data[1:], columns=summary_data[0])
                    df_summary.to_excel(writer, sheet_name='Summary', index=False)
        
        return str(filepath)
    
    def _generate_html_report(self, results: List[Dict[str, Any]], 
                            report_data: Optional[Dict[str, Any]],
                            include_visualizations: bool) -> str:
        """Generate HTML report content."""
        html_parts = []
        
        # HTML header
        html_parts.append("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NLP Analysis Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        h1, h2, h3 { color: #333; }
        .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }
        .summary-card { background-color: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #007bff; }
        .summary-card h3 { margin-top: 0; color: #007bff; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #007bff; color: white; }
        tr:hover { background-color: #f5f5f5; }
        .sentiment-positive { color: #28a745; font-weight: bold; }
        .sentiment-negative { color: #dc3545; font-weight: bold; }
        .sentiment-neutral { color: #6c757d; font-weight: bold; }
        .entity-tag { display: inline-block; padding: 2px 8px; margin: 2px; background-color: #e9ecef; border-radius: 3px; font-size: 12px; }
        .chart-container { margin: 20px 0; padding: 20px; background-color: #f8f9fa; border-radius: 5px; text-align: center; }
    </style>
</head>
<body>
    <div class="container">
        <h1>NLP Analysis Report</h1>
        <p>Generated on: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "</p>")
        
        # Summary section
        if report_data:
            html_parts.append("""
        <h2>Summary</h2>
        <div class="summary-grid">""")
            
            if "metadata" in report_data:
                meta = report_data["metadata"]
                html_parts.append(f"""
            <div class="summary-card">
                <h3>Documents</h3>
                <p>Total: {meta.get('total_documents', 0)}</p>
                <p>Success Rate: {meta.get('processing_summary', {}).get('success_rate', 0):.1%}</p>
            </div>""")
            
            if "sentiment_analysis" in report_data:
                sent = report_data["sentiment_analysis"]
                html_parts.append(f"""
            <div class="summary-card">
                <h3>Sentiment</h3>
                <p>Dominant: {sent.get('dominant_sentiment', 'N/A')}</p>
                <p>Total Analyzed: {sum(sent.get('distribution', {}).values())}</p>
            </div>""")
            
            if "entity_analysis" in report_data:
                ent = report_data["entity_analysis"]
                html_parts.append(f"""
            <div class="summary-card">
                <h3>Entities</h3>
                <p>Total: {ent.get('total_entities', 0)}</p>
                <p>Unique: {ent.get('unique_entities', 0)}</p>
            </div>""")
            
            html_parts.append("</div>")
        
        # Visualization placeholders
        if include_visualizations:
            html_parts.append("""
        <h2>Visualizations</h2>
        <div class="chart-container">
            <p><em>Sentiment Distribution Chart Placeholder</em></p>
        </div>
        <div class="chart-container">
            <p><em>Entity Types Chart Placeholder</em></p>
        </div>""")
        
        # Results table
        html_parts.append("""
        <h2>Detailed Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Document ID</th>
                    <th>Text Preview</th>
                    <th>Sentiment</th>
                    <th>Confidence</th>
                    <th>Entities</th>
                </tr>
            </thead>
            <tbody>""")
        
        for result in results[:100]:  # Limit to first 100 results
            doc_id = html.escape(str(result.get("document_id", "")))
            text_preview = html.escape(result.get("text", "")[:100] + "...")
            
            sentiment_label = result.get("sentiment", {}).get("label", "N/A")
            sentiment_score = result.get("sentiment", {}).get("score", 0.0)
            sentiment_class = f"sentiment-{sentiment_label.lower()}"
            
            entities = result.get("entities", [])
            entity_html = ""
            for entity in entities[:5]:  # Limit to first 5 entities
                entity_text = html.escape(entity.get("text", ""))
                entity_type = html.escape(entity.get("label", ""))
                entity_html += f'<span class="entity-tag">{entity_text} ({entity_type})</span>'
            
            if len(entities) > 5:
                entity_html += f'<span class="entity-tag">+{len(entities) - 5} more</span>'
            
            html_parts.append(f"""
                <tr>
                    <td>{doc_id}</td>
                    <td>{text_preview}</td>
                    <td class="{sentiment_class}">{sentiment_label}</td>
                    <td>{sentiment_score:.3f}</td>
                    <td>{entity_html}</td>
                </tr>""")
        
        html_parts.append("""
            </tbody>
        </table>
    </div>
</body>
</html>""")
        
        return "\n".join(html_parts)
    
    def _generate_markdown_summary(self, summary_data: Dict[str, Any]) -> str:
        """Generate Markdown summary content."""
        md_parts = []
        
        md_parts.append(f"# NLP Analysis Summary\n")
        md_parts.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        if "total_documents" in summary_data:
            md_parts.append(f"\n## Overview\n")
            md_parts.append(f"- **Total Documents**: {summary_data['total_documents']}\n")
            md_parts.append(f"- **Success Rate**: {summary_data.get('success_rate', 0):.1%}\n")
        
        if "sentiment" in summary_data:
            md_parts.append(f"\n## Sentiment Analysis\n")
            sent = summary_data["sentiment"]
            
            if "distribution" in sent:
                md_parts.append(f"\n### Distribution\n")
                for sentiment, count in sent["distribution"].items():
                    md_parts.append(f"- **{sentiment}**: {count}\n")
            
            md_parts.append(f"\n### Metrics\n")
            md_parts.append(f"- **Average Confidence**: {sent.get('average_confidence', 0):.3f}\n")
            md_parts.append(f"- **Confidence Std Dev**: {sent.get('confidence_std_dev', 0):.3f}\n")
        
        if "entities" in summary_data:
            md_parts.append(f"\n## Entity Analysis\n")
            ent = summary_data["entities"]
            md_parts.append(f"- **Total Entities**: {ent.get('total', 0)}\n")
            md_parts.append(f"- **Unique Entities**: {ent.get('unique', 0)}\n")
            md_parts.append(f"- **Average per Document**: {ent.get('average_per_document', 0):.2f}\n")
            
            if "type_distribution" in ent:
                md_parts.append(f"\n### Entity Types\n")
                for entity_type, count in ent["type_distribution"].items():
                    md_parts.append(f"- **{entity_type}**: {count}\n")
        
        if "text" in summary_data:
            md_parts.append(f"\n## Text Statistics\n")
            text = summary_data["text"]
            md_parts.append(f"- **Average Length**: {text.get('average_length', 0):.0f} characters\n")
            md_parts.append(f"- **Average Words**: {text.get('average_words', 0):.0f} words\n")
        
        return "".join(md_parts)