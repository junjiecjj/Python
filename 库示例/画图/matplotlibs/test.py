


import plotly.graph_objects as go
from plotly.validators.scatter.marker import SymbolValidator

raw_symbols = SymbolValidator().values
namestems = []
namevariants = []
symbols = []
for i in range(0,len(raw_symbols),3):
    name = raw_symbols[i+2]
    symbols.append(raw_symbols[i])
    namestems.append(name.replace("-open", "").replace("-dot", ""))
    namevariants.append(name[len(namestems[-1]):])

fig = go.Figure(go.Scatter(mode="markers", x=namevariants, y=namestems, marker_symbol=symbols,
                           marker_line_color="midnightblue", marker_color="lightskyblue",
                           marker_line_width=2, marker_size=15,
                           hovertemplate="name: %{y}%{x}<br>number: %{marker.symbol}<extra></extra>"))
fig.update_layout(title="Mouse over symbols for name & number!",
                  xaxis_range=[-1,4], yaxis_range=[len(set(namestems)),-1],
                  margin=dict(b=0,r=0), xaxis_side="top", height=1400, width=400)
fig.show()
