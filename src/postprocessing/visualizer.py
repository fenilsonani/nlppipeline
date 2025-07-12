"""Create interactive visualizations for NLP analysis results."""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from collections import Counter
from datetime import datetime
import numpy as np


class Visualizer:
    """Create various visualizations for sentiment and entity analysis."""
    
    def __init__(self, theme: str = "plotly_white"):
        """
        Initialize visualizer with theme.
        
        Args:
            theme: Plotly theme to use for all visualizations
        """
        self.theme = theme
        self.color_schemes = {
            "sentiment": {
                "positive": "#2ecc71",
                "negative": "#e74c3c",
                "neutral": "#95a5a6",
                "mixed": "#f39c12"
            },
            "entities": px.colors.qualitative.Set3
        }
    
    def plot_sentiment_distribution(self, results: List[Dict[str, Any]], 
                                  title: str = "Sentiment Distribution") -> go.Figure:
        """
        Create pie chart of sentiment distribution.
        
        Args:
            results: List of document processing results
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        sentiments = []
        for result in results:
            if "sentiment" in result and "label" in result["sentiment"]:
                sentiments.append(result["sentiment"]["label"])
        
        sentiment_counts = Counter(sentiments)
        
        fig = go.Figure(data=[go.Pie(
            labels=list(sentiment_counts.keys()),
            values=list(sentiment_counts.values()),
            hole=.3,
            marker=dict(colors=[self.color_schemes["sentiment"].get(s, "#bdc3c7") 
                              for s in sentiment_counts.keys()]),
            textinfo='label+percent',
            textposition='auto'
        )])
        
        fig.update_layout(
            title=title,
            template=self.theme,
            showlegend=True,
            height=500,
            annotations=[dict(text=f'Total: {len(sentiments)}', x=0.5, y=0.5, 
                            font_size=20, showarrow=False)]
        )
        
        return fig
    
    def plot_sentiment_confidence(self, results: List[Dict[str, Any]], 
                                title: str = "Sentiment Confidence Distribution") -> go.Figure:
        """
        Create box plot of sentiment confidence scores by sentiment type.
        
        Args:
            results: List of document processing results
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        data = []
        for result in results:
            if "sentiment" in result:
                data.append({
                    "sentiment": result["sentiment"].get("label", "unknown"),
                    "confidence": result["sentiment"].get("score", 0.0)
                })
        
        if not data:
            return self._create_empty_figure("No sentiment data available")
        
        df = pd.DataFrame(data)
        
        fig = go.Figure()
        for sentiment in df['sentiment'].unique():
            sentiment_data = df[df['sentiment'] == sentiment]['confidence']
            fig.add_trace(go.Box(
                y=sentiment_data,
                name=sentiment,
                marker_color=self.color_schemes["sentiment"].get(sentiment, "#bdc3c7"),
                boxmean='sd'
            ))
        
        fig.update_layout(
            title=title,
            yaxis_title="Confidence Score",
            xaxis_title="Sentiment",
            template=self.theme,
            showlegend=False,
            height=500
        )
        
        return fig
    
    def plot_entity_frequency(self, results: List[Dict[str, Any]], 
                            entity_type: Optional[str] = None,
                            top_n: int = 20,
                            title: Optional[str] = None) -> go.Figure:
        """
        Create bar chart of most frequent entities.
        
        Args:
            results: List of document processing results
            entity_type: Filter by specific entity type (None for all)
            top_n: Number of top entities to show
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        entities = []
        for result in results:
            if "entities" in result:
                for entity in result["entities"]:
                    if entity_type is None or entity.get("label") == entity_type:
                        entities.append(entity.get("text", ""))
        
        if not entities:
            return self._create_empty_figure("No entity data available")
        
        entity_counts = Counter(entities).most_common(top_n)
        
        fig = go.Figure(data=[go.Bar(
            x=[count for _, count in entity_counts],
            y=[entity for entity, _ in entity_counts],
            orientation='h',
            marker_color='#3498db',
            text=[count for _, count in entity_counts],
            textposition='auto'
        )])
        
        default_title = f"Top {top_n} Entities" + (f" ({entity_type})" if entity_type else "")
        fig.update_layout(
            title=title or default_title,
            xaxis_title="Frequency",
            yaxis_title="Entity",
            template=self.theme,
            height=max(400, len(entity_counts) * 25),
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
    
    def plot_entity_types_distribution(self, results: List[Dict[str, Any]], 
                                     title: str = "Entity Types Distribution") -> go.Figure:
        """
        Create sunburst chart of entity types and their frequencies.
        
        Args:
            results: List of document processing results
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        entity_data = []
        for result in results:
            if "entities" in result:
                for entity in result["entities"]:
                    entity_data.append({
                        "type": entity.get("label", "UNKNOWN"),
                        "text": entity.get("text", ""),
                        "count": 1
                    })
        
        if not entity_data:
            return self._create_empty_figure("No entity data available")
        
        # Aggregate data
        df = pd.DataFrame(entity_data)
        type_counts = df.groupby('type')['count'].sum().reset_index()
        
        # Create sunburst data
        fig = px.sunburst(
            type_counts,
            path=['type'],
            values='count',
            title=title,
            color='count',
            color_continuous_scale='Blues'
        )
        
        fig.update_layout(
            template=self.theme,
            height=600
        )
        
        return fig
    
    def plot_sentiment_timeline(self, results: List[Dict[str, Any]], 
                              date_field: str = "processed_at",
                              title: str = "Sentiment Over Time") -> go.Figure:
        """
        Create timeline of sentiment changes.
        
        Args:
            results: List of document processing results with timestamps
            date_field: Field name containing timestamp
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        timeline_data = []
        for result in results:
            if "sentiment" in result and date_field in result:
                try:
                    timestamp = pd.to_datetime(result[date_field])
                    timeline_data.append({
                        "date": timestamp,
                        "sentiment": result["sentiment"].get("label", "unknown"),
                        "confidence": result["sentiment"].get("score", 0.0)
                    })
                except:
                    continue
        
        if not timeline_data:
            return self._create_empty_figure("No timeline data available")
        
        df = pd.DataFrame(timeline_data).sort_values('date')
        
        # Create stacked area chart
        fig = go.Figure()
        
        for sentiment in df['sentiment'].unique():
            sentiment_df = df[df['sentiment'] == sentiment]
            sentiment_counts = sentiment_df.groupby('date').size().reset_index(name='count')
            
            fig.add_trace(go.Scatter(
                x=sentiment_counts['date'],
                y=sentiment_counts['count'],
                mode='lines',
                name=sentiment,
                stackgroup='one',
                fillcolor=self.color_schemes["sentiment"].get(sentiment, "#bdc3c7")
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Document Count",
            template=self.theme,
            hovermode='x unified',
            height=500
        )
        
        return fig
    
    def plot_entity_co_occurrence(self, results: List[Dict[str, Any]], 
                                min_count: int = 5,
                                title: str = "Entity Co-occurrence Network") -> go.Figure:
        """
        Create network graph of entity co-occurrences.
        
        Args:
            results: List of document processing results
            min_count: Minimum co-occurrence count to include
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        # Build co-occurrence matrix
        co_occurrences = Counter()
        
        for result in results:
            if "entities" in result:
                entities = list(set(e.get("text", "") for e in result["entities"]))
                for i in range(len(entities)):
                    for j in range(i + 1, len(entities)):
                        pair = tuple(sorted([entities[i], entities[j]]))
                        co_occurrences[pair] += 1
        
        # Filter by minimum count
        filtered_pairs = [(e1, e2, count) for (e1, e2), count in co_occurrences.items() 
                         if count >= min_count]
        
        if not filtered_pairs:
            return self._create_empty_figure("No significant co-occurrences found")
        
        # Create network graph
        edge_trace = []
        node_trace = []
        
        # Get unique nodes
        nodes = list(set([e1 for e1, _, _ in filtered_pairs] + 
                        [e2 for _, e2, _ in filtered_pairs]))
        node_positions = self._generate_node_positions(len(nodes))
        
        # Create edges
        for e1, e2, count in filtered_pairs:
            x0, y0 = node_positions[nodes.index(e1)]
            x1, y1 = node_positions[nodes.index(e2)]
            
            edge_trace.append(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=min(count/2, 10), color='#888'),
                hoverinfo='none'
            ))
        
        # Create nodes
        node_trace = go.Scatter(
            x=[pos[0] for pos in node_positions],
            y=[pos[1] for pos in node_positions],
            mode='markers+text',
            marker=dict(size=20, color='#3498db'),
            text=nodes,
            textposition="top center",
            hoverinfo='text',
            hovertext=nodes
        )
        
        fig = go.Figure(data=edge_trace + [node_trace])
        
        fig.update_layout(
            title=title,
            template=self.theme,
            showlegend=False,
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600
        )
        
        return fig
    
    def create_dashboard(self, results: List[Dict[str, Any]], 
                       title: str = "NLP Analysis Dashboard") -> go.Figure:
        """
        Create comprehensive dashboard with multiple visualizations.
        
        Args:
            results: List of document processing results
            title: Dashboard title
            
        Returns:
            Plotly figure object with subplots
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Sentiment Distribution", "Entity Types", 
                          "Top Entities", "Confidence Scores"),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "box"}]],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # Sentiment distribution
        sentiments = [r["sentiment"]["label"] for r in results 
                     if "sentiment" in r and "label" in r["sentiment"]]
        sentiment_counts = Counter(sentiments)
        
        fig.add_trace(go.Pie(
            labels=list(sentiment_counts.keys()),
            values=list(sentiment_counts.values()),
            hole=.3,
            marker=dict(colors=[self.color_schemes["sentiment"].get(s, "#bdc3c7") 
                              for s in sentiment_counts.keys()]),
            showlegend=False
        ), row=1, col=1)
        
        # Entity types
        entity_types = []
        for result in results:
            if "entities" in result:
                entity_types.extend([e.get("label", "UNKNOWN") for e in result["entities"]])
        
        type_counts = Counter(entity_types).most_common(10)
        if type_counts:
            fig.add_trace(go.Bar(
                x=[t for t, _ in type_counts],
                y=[c for _, c in type_counts],
                marker_color='#e74c3c',
                showlegend=False
            ), row=1, col=2)
        
        # Top entities
        entities = []
        for result in results:
            if "entities" in result:
                entities.extend([e.get("text", "") for e in result["entities"]])
        
        entity_counts = Counter(entities).most_common(10)
        if entity_counts:
            fig.add_trace(go.Bar(
                y=[e for e, _ in entity_counts],
                x=[c for _, c in entity_counts],
                orientation='h',
                marker_color='#3498db',
                showlegend=False
            ), row=2, col=1)
        
        # Confidence scores
        confidences = {"positive": [], "negative": [], "neutral": []}
        for result in results:
            if "sentiment" in result:
                label = result["sentiment"].get("label", "neutral")
                score = result["sentiment"].get("score", 0.0)
                if label in confidences:
                    confidences[label].append(score)
        
        for sentiment, scores in confidences.items():
            if scores:
                fig.add_trace(go.Box(
                    y=scores,
                    name=sentiment,
                    marker_color=self.color_schemes["sentiment"].get(sentiment, "#bdc3c7"),
                    showlegend=False
                ), row=2, col=2)
        
        fig.update_layout(
            title_text=title,
            template=self.theme,
            height=800
        )
        
        return fig
    
    def _create_empty_figure(self, message: str) -> go.Figure:
        """Create empty figure with message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20, color="gray")
        )
        fig.update_layout(
            template=self.theme,
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False)
        )
        return fig
    
    def _generate_node_positions(self, n_nodes: int) -> List[Tuple[float, float]]:
        """Generate circular positions for network nodes."""
        positions = []
        angle_step = 2 * np.pi / n_nodes
        
        for i in range(n_nodes):
            angle = i * angle_step
            x = np.cos(angle)
            y = np.sin(angle)
            positions.append((x, y))
        
        return positions