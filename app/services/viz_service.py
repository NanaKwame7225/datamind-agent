from __future__ import annotations
import logging, json
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class VizService:

    def plotly_line(self, df, x, y, title="Time Series"):
        import plotly.express as px
        return json.loads(px.line(df, x=x, y=y, title=title, template="plotly_dark").to_json())

    def plotly_bar(self, df, x, y, title="Bar Chart"):
        import plotly.express as px
        return json.loads(px.bar(df, x=x, y=y, title=title, template="plotly_dark").to_json())

    def plotly_histogram(self, df, column, title="Distribution", bins=30):
        import plotly.express as px
        return json.loads(px.histogram(df, x=column, title=title, nbins=bins, template="plotly_dark").to_json())

    def plotly_heatmap(self, df, title="Correlation Matrix"):
        import plotly.graph_objects as go
        corr = df.select_dtypes(include="number").corr().round(2)
        fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns.tolist(),
                                         y=corr.columns.tolist(), colorscale="RdBu", zmid=0))
        fig.update_layout(title=title, template="plotly_dark")
        return json.loads(fig.to_json())

    def plotly_forecast(self, historical, forecast, lower, upper, title="Forecast"):
        import plotly.graph_objects as go
        n = len(historical)
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=historical, mode="lines", name="Historical", line=dict(color="#00d4aa")))
        fig.add_trace(go.Scatter(x=list(range(n, n+len(forecast))), y=forecast, mode="lines",
                                  name="Forecast", line=dict(color="#0ea5e9", dash="dash")))
        x_band = list(range(n, n+len(forecast)))
        fig.add_trace(go.Scatter(x=x_band+x_band[::-1], y=upper+lower[::-1],
                                  fill="toself", fillcolor="rgba(14,165,233,0.15)",
                                  line=dict(color="rgba(255,255,255,0)"), name="95% CI"))
        fig.update_layout(title=title, template="plotly_dark")
        return json.loads(fig.to_json())

    def auto_chart(self, df):
        charts = []
        numeric = df.select_dtypes(include="number").columns.tolist()
        cat = df.select_dtypes(include=["object","category"]).columns.tolist()
        date = df.select_dtypes(include=["datetime"]).columns.tolist()
        if len(numeric) >= 3:
            charts.append({"type":"heatmap","title":"Correlation Matrix","figure":self.plotly_heatmap(df)})
        if numeric:
            charts.append({"type":"histogram","title":f"Distribution: {numeric[0]}",
                           "figure":self.plotly_histogram(df, numeric[0])})
        if date and numeric:
            charts.append({"type":"line","title":f"{numeric[0]} over time",
                           "figure":self.plotly_line(df, date[0], numeric[0])})
        if cat and numeric:
            gb = df.groupby(cat[0])[numeric[0]].sum().reset_index().nlargest(15, numeric[0])
            charts.append({"type":"bar","title":f"{numeric[0]} by {cat[0]}",
                           "figure":self.plotly_bar(gb, cat[0], numeric[0])})
        return charts

viz_service = VizService()
