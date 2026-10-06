"""Interactive dashboard of the thesis results (Dash).

Choose an index and one of the five nested models: the three most significant coefficients are shown as cards, all
effects per standard deviation with their 95% intervals (HC3 errors) as a bar chart, and the 84-month rolling coefficients with a 95%
band (HAC errors) as lines.

Run:  python dashboard.py   ->  http://127.0.0.1:8050
"""
from pathlib import Path

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
from dash import Dash, Input, Output, dcc, html

HERE = Path(__file__).resolve().parent
df = pd.read_csv(HERE / "data" / "data_monthly.csv", parse_dates=["date"]).set_index("date")

BLUE, BG, CARD, NEG, POS, NS = "#002e6d", "#f6f6ff", "#ffffff", "#d9534f", "#2ca02c", "#bdbdbd"
LABELS = {
    "policy_rate_ECB_lag2": "ECB policy rate (lag 2)", "policy_rate_fed": "Fed policy rate", "d_oat10y": "OAT 10y",
    "d_us10y": "Treasury 10y", "dlog_vix": "VIX", "dlog_usd_eur": "USD/EUR (dollars per euro)",
    "dlog_prix_petrole": "Oil price", "d_inflation_FR": "French inflation", "dlog_inflation_US": "US inflation",
    "d_taux_chômage_FR": "French unemployment", "unemployment_rate_US ": "US unemployment rate", "post2008": "Post-2008 period",
    "postCOVID": "Post-COVID period", "QE_EU": "QE, euro area", "QE_US": "QE, United States"}
MODEL_LABELS = {1: "Model 1: policy rate only", 2: "Model 2: with long rate, VIX and regimes", 3: "Model 3: with domestic macro variables",
                4: "Model 4: with foreign variables", 5: "Model 5: model 4 on excess returns"}

FR = ["d_inflation_FR", "d_oat10y", "d_taux_chômage_FR", "dlog_usd_eur", "dlog_vix", "dlog_prix_petrole", "post2008", "postCOVID"]
US = ["dlog_inflation_US", "d_us10y", "unemployment_rate_US ", "dlog_usd_eur", "dlog_vix", "dlog_prix_petrole", "post2008", "postCOVID"]
FULL = ["policy_rate_ECB_lag2", "d_inflation_FR", "d_oat10y", "d_taux_chômage_FR", "dlog_usd_eur", "dlog_vix", "dlog_prix_petrole",
        "post2008", "postCOVID", "policy_rate_fed", "dlog_inflation_US", "d_us10y", "unemployment_rate_US ", "QE_EU", "QE_US"]
SPECS = {  # index: (return, excess return, own policy rate, own long rate, regressors of models 1 to 3)
    "CAC 40": ("r_cac40", "excess_cac40", "policy_rate_ECB_lag2", "d_oat10y", FR, "QE_EU"),
    "S&P 500": ("r_sp500", "excess_sp500", "policy_rate_fed", "d_us10y", US, "QE_US"),
    "Nasdaq 100": ("r_nasdaq100", "excess_nasdaq", "policy_rate_fed", "d_us10y", US, "QE_US")}
WINDOW = 84


def regressors(index, k):
    _, _, rate, long_rate, own, qe = SPECS[index]
    return {1: [rate], 2: [rate, long_rate, "post2008", "postCOVID", "dlog_vix"], 3: [rate] + own + [qe], 4: FULL, 5: FULL}[k]


MODELS = {(i, k): sm.OLS(df[SPECS[i][1] if k == 5 else SPECS[i][0]], sm.add_constant(df[regressors(i, k)])).fit(cov_type="HC3")
          for i in SPECS for k in range(1, 6)}


def rolling(index):
    y, _, rate, long_rate, _, _ = SPECS[index]
    cols = [rate, long_rate, "dlog_vix"]
    rows = []
    for i in range(WINDOW, len(df) + 1):
        sub = df.iloc[i - WINDOW:i]
        r = sm.OLS(sub[y], sm.add_constant(sub[cols])).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
        rows.append({"date": sub.index[-1], **{c: r.params[c] for c in cols}, **{c + "_se": r.bse[c] for c in cols}})
    return pd.DataFrame(rows).set_index("date"), cols


ROLL = {i: rolling(i) for i in SPECS}


def card(var, model):
    coef, p = model.params[var] * df[var].std(), model.pvalues[var]
    colour = (POS if coef > 0 else NEG) if p < 0.05 else NS
    return dbc.Card([html.H5(LABELS[var], style={"textAlign": "center", "color": BLUE}),
                     html.H2(f"{100 * coef:+.2f}", style={"textAlign": "center", "color": colour, "fontSize": "36px"}),
                     html.P("Significant at 5%" if p < 0.05 else "Not significant at 5%", style={"textAlign": "center", "fontWeight": "bold"})],
                    style={"padding": "16px", "borderRadius": "12px", "backgroundColor": CARD, "boxShadow": "0 2px 6px rgba(0,0,0,0.1)"})


def layout(fig, title, ytitle):
    fig.update_layout(title=dict(text=f"<b>{title}</b>", x=0.5, font=dict(size=17)), template="simple_white", height=440,
                      plot_bgcolor=BG, paper_bgcolor=BG, font=dict(color="#000000"), yaxis_title=ytitle, legend=dict(orientation="h", y=-0.25))
    return fig


def regression_figure(model, title):
    sd = df[model.params.drop("const").index].std()               # effects per standard deviation, comparable across variables
    b = model.params.drop("const") * sd * 100
    err = model.bse.drop("const") * sd * 100 * 1.96
    p = model.pvalues.drop("const")
    colours = [(POS if b[v] > 0 else NEG) if p[v] < 0.05 else NS for v in b.index]
    fig = go.Figure(go.Bar(x=[LABELS[v] for v in b.index], y=b.values, error_y=dict(type="data", array=err.values), marker_color=colours))
    fig.update_xaxes(tickangle=-35)
    fig.update_layout(margin=dict(b=140, l=110))
    return layout(fig, title, "Points (95% interval)")


def rolling_figure(index, selected):
    r, cols = ROLL[index]
    palette = dict(zip(cols, [BLUE, "#e6a817", NEG]))
    fig = go.Figure()
    for c in selected:
        b, se = 100 * r[c], 100 * r[c + "_se"]
        fig.add_trace(go.Scatter(x=list(r.index) + list(r.index[::-1]), y=list(b + 1.96 * se) + list((b - 1.96 * se)[::-1]), fill="toself",
                                 fillcolor=palette[c], opacity=0.15, line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=r.index, y=b, mode="lines", name=LABELS[c], line=dict(color=palette[c], width=2)))
    fig.add_hline(y=0, line_dash="dot", line_color="grey")
    fig.update_yaxes(range=[-30, 30])
    return layout(fig, f"84-month rolling windows, {index}", "Points (95% band, cut at 30)")


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "ECB, Fed and equity returns"
label = {"fontWeight": "bold", "color": BLUE}
app.layout = dbc.Container([
    dbc.Card(html.H1("ECB and Fed policy and equity returns: CAC 40, S&P 500, Nasdaq 100 (2000-2024)",
                     style={"textAlign": "center", "color": BLUE, "padding": "15px", "fontSize": "28px"}),
             style={"backgroundColor": CARD, "border": f"3px solid {BLUE}", "borderRadius": "12px", "marginTop": "20px"}),
    html.Br(),
    dbc.Card(dbc.Row([
        dbc.Col([html.Label("Index", style=label), dcc.Dropdown(id="index", options=list(SPECS), value="CAC 40", clearable=False)], width=4),
        dbc.Col([html.Label("Regression model", style=label),
                 dcc.Dropdown(id="model", options=[{"label": v, "value": k} for k, v in MODEL_LABELS.items()], value=5, clearable=False)], width=4),
        dbc.Col([html.Label("Rolling-window coefficients", style=label), dcc.Dropdown(id="vars", multi=True)], width=4)]),
        style={"padding": "16px", "borderRadius": "12px", "backgroundColor": CARD, "boxShadow": "0 2px 6px rgba(0,0,0,0.1)"}),
    html.Br(),
    dbc.Row(id="cards"),
    html.Br(),
    dbc.Row([dbc.Col(dcc.Graph(id="regression"), width=6), dbc.Col(dcc.Graph(id="rolling"), width=6)]),
    html.P("Cards and bars: monthly return in points for a rise of one standard deviation of the variable, HC3 errors. Green and red are significant at 5%, "
           "grey is not. Rolling coefficients are only informative when the policy rate moves inside the window.",
           style={"color": "#555", "fontSize": "13px", "padding": "8px 4px 20px"}),
], fluid=True, style={"backgroundColor": BG})


@app.callback(Output("vars", "options"), Output("vars", "value"), Input("index", "value"))
def rolling_options(index):
    cols = ROLL[index][1]
    return [{"label": LABELS[c], "value": c} for c in cols], cols[:1]


@app.callback(Output("cards", "children"), Output("regression", "figure"), Output("rolling", "figure"),
              Input("index", "value"), Input("model", "value"), Input("vars", "value"))
def update(index, k, selected):
    model = MODELS[(index, k)]
    top = model.pvalues.drop("const").sort_values().index[:3]
    cards = [dbc.Col(card(v, model), width=12 // len(top)) for v in top]
    return cards, regression_figure(model, f"{index}, {MODEL_LABELS[k]}"), rolling_figure(index, selected or ROLL[index][1][:1])


if __name__ == "__main__":
    app.run(debug=False, port=8050)
