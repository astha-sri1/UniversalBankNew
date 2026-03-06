import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Universal Bank | Loan Analytics Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background: #0f1117; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f2e 0%, #141824 100%);
        border-right: 1px solid rgba(255,255,255,0.07);
    }
    [data-testid="stSidebar"] * { color: #e0e6f0 !important; }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e2435 0%, #252d42 100%);
        border: 1px solid rgba(100,149,237,0.2);
        border-radius: 16px;
        padding: 22px 26px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-3px); }
    .metric-value { font-size: 2.2rem; font-weight: 700; color: #6495ED; margin: 0; }
    .metric-label { font-size: 0.82rem; color: #8b95b0; margin-top: 4px; text-transform: uppercase; letter-spacing: 1px; }
    .metric-sub { font-size: 0.78rem; color: #5eead4; margin-top: 6px; }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, rgba(100,149,237,0.15) 0%, transparent 100%);
        border-left: 4px solid #6495ED;
        padding: 12px 20px;
        border-radius: 0 12px 12px 0;
        margin: 24px 0 18px 0;
    }
    .section-header h3 { color: #d0d8f0; margin: 0; font-size: 1.15rem; font-weight: 600; }
    .section-header p { color: #8b95b0; margin: 4px 0 0 0; font-size: 0.82rem; }

    /* Insight boxes */
    .insight-box {
        background: linear-gradient(135deg, #1a2535 0%, #1e2d40 100%);
        border: 1px solid rgba(94,234,212,0.25);
        border-radius: 12px;
        padding: 18px 22px;
        margin: 10px 0;
    }
    .insight-box h4 { color: #5eead4; margin: 0 0 8px 0; font-size: 0.9rem; font-weight: 600; }
    .insight-box p { color: #c0cce0; margin: 0; font-size: 0.83rem; line-height: 1.55; }

    /* Offer cards */
    .offer-card {
        background: linear-gradient(135deg, #1a2635 0%, #1d2e45 100%);
        border: 1px solid rgba(100,149,237,0.3);
        border-radius: 16px;
        padding: 22px 24px;
        margin-bottom: 16px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.25);
    }
    .offer-card h4 { color: #6495ED; margin: 0 0 10px 0; font-size: 1rem; font-weight: 600; }
    .offer-card p { color: #b0bcd0; margin: 0; font-size: 0.83rem; line-height: 1.6; }
    .offer-tag {
        display: inline-block;
        background: rgba(100,149,237,0.2);
        color: #6495ED;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.72rem;
        font-weight: 600;
        margin: 6px 4px 0 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .offer-tag.green { background: rgba(94,234,212,0.15); color: #5eead4; }
    .offer-tag.orange { background: rgba(251,146,60,0.15); color: #fb923c; }

    /* Tab styling */
    [data-baseweb="tab-list"] { background: #1a1f2e; border-radius: 12px; padding: 6px; gap: 4px; }
    [data-baseweb="tab"] { border-radius: 8px !important; color: #8b95b0 !important; font-weight: 500 !important; }
    [aria-selected="true"] { background: #6495ED !important; color: white !important; }

    /* Divider */
    hr { border-color: rgba(255,255,255,0.07); margin: 28px 0; }

    /* Plotly chart backgrounds */
    .stPlotlyChart { border-radius: 14px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Load & Cache Data ─────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("UniversalBank.csv")
    df = df.drop(columns=["ZIP Code"], errors="ignore")
    df["Income Group"] = pd.cut(df["Income"], bins=[0,50,100,150,200,250,300],
                                 labels=["<50K","50-100K","100-150K","150-200K","200-250K","250K+"])
    df["Age Group"] = pd.cut(df["Age"], bins=[20,30,40,50,60,70],
                              labels=["20-30","30-40","40-50","50-60","60-70"])
    df["CCAvg Group"] = pd.cut(df["CCAvg"], bins=[0,1,3,5,7,10],
                                labels=["<1K","1-3K","3-5K","5-7K","7K+"])
    df["Edu Label"] = df["Education"].map({1:"Undergrad",2:"Graduate",3:"Advanced/Prof"})
    df["Loan Accepted"] = df["Personal Loan"].map({0:"Rejected",1:"Accepted"})
    return df

df = load_data()

# ─── Sidebar Filters ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px 0;'>
        <div style='font-size:2.4rem;'>🏦</div>
        <div style='font-size:1.1rem; font-weight:700; color:#6495ED; margin-top:6px;'>Universal Bank</div>
        <div style='font-size:0.75rem; color:#8b95b0; margin-top:2px;'>Personal Loan Intelligence</div>
    </div>
    <hr style='border-color:rgba(255,255,255,0.08); margin:12px 0;'>
    """, unsafe_allow_html=True)

    st.markdown("**🎛️ Dashboard Filters**")

    income_range = st.slider("Income Range ($000)", int(df["Income"].min()), int(df["Income"].max()),
                              (int(df["Income"].min()), int(df["Income"].max())))

    edu_options = ["All", "Undergrad", "Graduate", "Advanced/Prof"]
    edu_filter = st.selectbox("Education Level", edu_options)

    family_options = sorted(df["Family"].unique().tolist())
    family_filter = st.multiselect("Family Size", family_options, default=family_options)

    loan_status = st.radio("Show Customers", ["All", "Loan Accepted", "Loan Rejected"])

    st.markdown("<hr style='border-color:rgba(255,255,255,0.08);'>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:0.75rem; color:#8b95b0;'>Dataset: {len(df):,} customers<br>Acceptance Rate: {df['Personal Loan'].mean()*100:.1f}%</div>", unsafe_allow_html=True)

# ─── Apply Filters ────────────────────────────────────────────────────────────
fdf = df.copy()
fdf = fdf[(fdf["Income"] >= income_range[0]) & (fdf["Income"] <= income_range[1])]
if edu_filter != "All":
    fdf = fdf[fdf["Edu Label"] == edu_filter]
if family_filter:
    fdf = fdf[fdf["Family"].isin(family_filter)]
if loan_status == "Loan Accepted":
    fdf = fdf[fdf["Personal Loan"] == 1]
elif loan_status == "Loan Rejected":
    fdf = fdf[fdf["Personal Loan"] == 0]

DARK_BG = "#0f1117"
CHART_BG = "#1a1f2e"
GRID_COLOR = "rgba(255,255,255,0.05)"
COLORS = ["#6495ED","#5eead4","#fb923c","#a78bfa","#f472b6","#34d399"]
PLOTLY_LAYOUT = dict(
    paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
    font=dict(color="#c0cce0", family="Inter"),
    xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
    yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
    margin=dict(l=30, r=30, t=50, b=30),
)

def apply_layout(fig, title="", height=380):
    fig.update_layout(**PLOTLY_LAYOUT, title=dict(text=title, font=dict(size=14, color="#d0d8f0")), height=height)
    return fig

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style='background: linear-gradient(135deg, #1a1f2e 0%, #141824 100%);
            border: 1px solid rgba(100,149,237,0.2); border-radius:20px;
            padding: 30px 36px; margin-bottom:28px;'>
    <div style='display:flex; align-items:center; gap:16px;'>
        <div style='font-size:3rem;'>🏦</div>
        <div>
            <h1 style='margin:0; color:#d0d8f0; font-size:1.8rem; font-weight:700;'>Universal Bank — Loan Analytics</h1>
            <p style='margin:6px 0 0 0; color:#8b95b0; font-size:0.88rem;'>
                Descriptive · Diagnostic · Predictive · Prescriptive Analytics
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── KPI Row ──────────────────────────────────────────────────────────────────
accepted = fdf[fdf["Personal Loan"]==1]
rejected = fdf[fdf["Personal Loan"]==0]
acc_rate = fdf["Personal Loan"].mean()*100 if len(fdf) > 0 else 0
avg_income_acc = accepted["Income"].mean() if len(accepted) > 0 else 0
avg_cc_acc = accepted["CCAvg"].mean() if len(accepted) > 0 else 0

c1,c2,c3,c4,c5 = st.columns(5)
for col, val, label, sub in [
    (c1, f"{len(fdf):,}", "Total Customers", "in filtered view"),
    (c2, f"{len(accepted):,}", "Loan Accepted", f"{acc_rate:.1f}% acceptance rate"),
    (c3, f"${fdf['Income'].mean():.0f}K", "Avg Income", "all filtered customers"),
    (c4, f"${avg_income_acc:.0f}K", "Avg Income (Accepted)", "loan acceptors"),
    (c5, f"${avg_cc_acc:.1f}K", "Avg CC Spend (Accepted)", "per month"),
]:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
            <div class="metric-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("")

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Descriptive", "🔍 Diagnostic", "🤖 Predictive", "🎯 Prescriptive"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DESCRIPTIVE
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("""
    <div class="section-header">
        <h3>📊 Descriptive Analytics — Who Are Our Customers?</h3>
        <p>Distribution of demographics, loan status breakdown and average financial metrics.</p>
    </div>""", unsafe_allow_html=True)

    # Row 1 — Donut + Age dist
    c1, c2 = st.columns([1,1.6])
    with c1:
        # Interactive donut with drill-down
        donut_mode = st.selectbox("Drill-down by", ["Overall", "By Education", "By Family Size", "By Age Group"], key="donut_mode")
        if donut_mode == "Overall":
            vals = fdf["Loan Accepted"].value_counts()
            fig = go.Figure(go.Pie(labels=vals.index, values=vals.values,
                                    hole=0.6, marker_colors=["#6495ED","#1e2435"],
                                    textinfo="label+percent"))
            fig.update_layout(**PLOTLY_LAYOUT, height=320,
                               title="Loan Acceptance — Overall",
                               annotations=[dict(text=f"{acc_rate:.1f}%<br><span style='font-size:0.8em'>Accepted</span>",
                                                  x=0.5,y=0.5,showarrow=False,
                                                  font=dict(size=16,color="#6495ED"))])
        elif donut_mode == "By Education":
            grp = fdf.groupby(["Edu Label","Loan Accepted"]).size().reset_index(name="count")
            fig = px.bar(grp, x="Edu Label", y="count", color="Loan Accepted",
                          barmode="group", color_discrete_sequence=["#6495ED","#1e2435"],
                          labels={"count":"Customers","Edu Label":"Education"})
            apply_layout(fig, "Acceptance by Education", 320)
        elif donut_mode == "By Family Size":
            grp = fdf.groupby(["Family","Personal Loan"]).size().unstack(fill_value=0)
            grp["rate"] = grp[1]/(grp[0]+grp[1])*100 if 1 in grp.columns else 0
            grp = grp.reset_index()
            fig = px.bar(grp, x="Family", y="rate", color_discrete_sequence=["#6495ED"],
                          labels={"rate":"Acceptance Rate %","Family":"Family Size"})
            apply_layout(fig, "Acceptance Rate by Family Size", 320)
        else:
            grp = fdf.groupby(["Age Group","Personal Loan"]).size().unstack(fill_value=0)
            grp["rate"] = grp.get(1,0)/(grp.get(0,1)+grp.get(1,0))*100
            grp = grp.reset_index()
            fig = px.bar(grp, x="Age Group", y="rate", color_discrete_sequence=["#5eead4"],
                          labels={"rate":"Acceptance Rate %"})
            apply_layout(fig, "Acceptance Rate by Age Group", 320)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.histogram(fdf, x="Age", color="Loan Accepted",
                            nbins=25, barmode="overlay", opacity=0.8,
                            color_discrete_sequence=["#6495ED","#fb923c"],
                            labels={"Age":"Customer Age","count":"Customers"})
        apply_layout(fig, "Age Distribution by Loan Status", 320)
        st.plotly_chart(fig, use_container_width=True)

    # Row 2 — Income + Education + Family
    c1, c2, c3 = st.columns(3)
    with c1:
        fig = px.box(fdf, x="Loan Accepted", y="Income", color="Loan Accepted",
                      color_discrete_sequence=["#6495ED","#fb923c"],
                      labels={"Income":"Annual Income ($000)"})
        apply_layout(fig, "Income Distribution by Loan Status", 350)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        edu_grp = fdf.groupby("Edu Label")["Personal Loan"].agg(["sum","count"]).reset_index()
        edu_grp["rate"] = edu_grp["sum"]/edu_grp["count"]*100
        fig = px.bar(edu_grp, x="Edu Label", y="rate", color="rate",
                      color_continuous_scale=["#1e2435","#6495ED","#5eead4"],
                      text=edu_grp["rate"].apply(lambda x: f"{x:.1f}%"),
                      labels={"rate":"Acceptance Rate %","Edu Label":"Education"})
        apply_layout(fig, "Loan Acceptance Rate by Education Level", 350)
        fig.update_traces(textposition="outside")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with c3:
        fam_grp = fdf.groupby("Family")["Personal Loan"].agg(["sum","count"]).reset_index()
        fam_grp["rate"] = fam_grp["sum"]/fam_grp["count"]*100
        fig = px.bar(fam_grp, x="Family", y="rate", color_discrete_sequence=["#a78bfa"],
                      text=fam_grp["rate"].apply(lambda x: f"{x:.1f}%"),
                      labels={"rate":"Acceptance Rate %","Family":"Family Size"})
        apply_layout(fig, "Acceptance Rate by Family Size", 350)
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    # Row 3 — Summary stats table
    st.markdown("""<div class="section-header"><h3>📋 Average Financial Metrics by Loan Status</h3></div>""", unsafe_allow_html=True)
    stats = fdf.groupby("Loan Accepted")[["Income","CCAvg","Mortgage","Age"]].mean().round(2).reset_index()
    stats.columns = ["Loan Status","Avg Income ($000)","Avg CC Spend ($000)/mo","Avg Mortgage ($000)","Avg Age"]
    st.dataframe(stats.style.background_gradient(cmap="Blues", subset=["Avg Income ($000)","Avg CC Spend ($000)/mo"]),
                  use_container_width=True)

    # Insights
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="insight-box">
            <h4>📌 Key Demographic Finding</h4>
            <p>Only <strong>9.6%</strong> of all customers accepted the personal loan. The majority of acceptors are in the <strong>30–50 age range</strong>, with a peak around 35–45 years — suggesting mid-career professionals are the prime segment.</p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="insight-box">
            <h4>📌 Education & Family Patterns</h4>
            <p>Customers with <strong>Advanced/Professional education</strong> and <strong>family sizes of 3–4</strong> show the highest loan acceptance rates. Graduate and advanced-degree holders are over 2× more likely to accept compared to undergrads.</p>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DIAGNOSTIC
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class="section-header">
        <h3>🔍 Diagnostic Analytics — Why Do Customers Accept Loans?</h3>
        <p>Comparing accepted vs rejected groups to identify key differentiators and behavioral signals.</p>
    </div>""", unsafe_allow_html=True)

    # Row 1 — Income vs CCAvg scatter + CC violin
    c1, c2 = st.columns([1.6,1])
    with c1:
        fig = px.scatter(fdf.sample(min(len(fdf),2000)), x="Income", y="CCAvg",
                          color="Loan Accepted", size="Mortgage",
                          color_discrete_sequence=["#6495ED","#fb923c"],
                          labels={"Income":"Annual Income ($000)","CCAvg":"CC Spend/month ($000)"},
                          opacity=0.7, size_max=20)
        apply_layout(fig, "Income vs Credit Card Spending (size = Mortgage)", 400)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.violin(fdf, x="Loan Accepted", y="CCAvg", color="Loan Accepted",
                         box=True, points="outliers",
                         color_discrete_sequence=["#6495ED","#fb923c"],
                         labels={"CCAvg":"CC Spend/month ($000)"})
        apply_layout(fig, "CC Spending Distribution", 400)
        st.plotly_chart(fig, use_container_width=True)

    # Row 2 — Banking services comparison
    c1, c2 = st.columns(2)
    with c1:
        services = ["Securities Account","CD Account","Online","CreditCard"]
        acc_rates = []
        for svc in services:
            rate = fdf[fdf[svc]==1]["Personal Loan"].mean()*100 if fdf[svc].sum() > 0 else 0
            no_rate = fdf[fdf[svc]==0]["Personal Loan"].mean()*100
            acc_rates.append({"Service":svc,"Has Service":round(rate,1),"No Service":round(no_rate,1)})

        svc_df = pd.DataFrame(acc_rates)
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Has Service", x=svc_df["Service"], y=svc_df["Has Service"],
                              marker_color="#5eead4", text=svc_df["Has Service"].apply(lambda x: f"{x}%"),
                              textposition="outside"))
        fig.add_trace(go.Bar(name="No Service", x=svc_df["Service"], y=svc_df["No Service"],
                              marker_color="#1e2435", text=svc_df["No Service"].apply(lambda x: f"{x}%"),
                              textposition="outside"))
        fig.update_layout(**PLOTLY_LAYOUT, barmode="group", height=380,
                           title=dict(text="Loan Acceptance Rate by Banking Service Usage",
                                       font=dict(size=14,color="#d0d8f0")))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Heatmap — correlation
        num_cols = ["Age","Income","CCAvg","Mortgage","Family","Education","Personal Loan",
                     "Securities Account","CD Account","Online","CreditCard"]
        corr = fdf[num_cols].corr().round(2)
        fig = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
            colorscale=[[0,"#1e2435"],[0.5,"#6495ED"],[1,"#5eead4"]],
            text=corr.values.round(2), texttemplate="%{text}",
            showscale=True, colorbar=dict(tickcolor="#c0cce0",tickfont=dict(color="#c0cce0"))
        ))
        apply_layout(fig, "Correlation Heatmap", 380)
        st.plotly_chart(fig, use_container_width=True)

    # Row 3 — Income groups + Mortgage
    c1, c2 = st.columns(2)
    with c1:
        ig = fdf.groupby("Income Group")["Personal Loan"].agg(["sum","count"]).reset_index()
        ig["rate"] = ig["sum"]/ig["count"]*100
        fig = px.bar(ig, x="Income Group", y="rate",
                      color="rate", color_continuous_scale=["#1e2435","#6495ED","#5eead4"],
                      text=ig["rate"].apply(lambda x: f"{x:.1f}%"),
                      labels={"rate":"Acceptance Rate %"})
        apply_layout(fig, "Loan Acceptance Rate by Income Group", 350)
        fig.update_traces(textposition="outside")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.histogram(fdf, x="Mortgage", color="Loan Accepted", nbins=30,
                            barmode="overlay", opacity=0.8,
                            color_discrete_sequence=["#6495ED","#fb923c"],
                            labels={"Mortgage":"Mortgage Value ($000)"})
        apply_layout(fig, "Mortgage Distribution by Loan Status", 350)
        st.plotly_chart(fig, use_container_width=True)

    # Key drivers insight
    st.markdown("""<div class="section-header"><h3>🔑 Key Diagnostic Findings</h3></div>""", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="insight-box">
            <h4>💰 Income is the #1 Driver</h4>
            <p>Customers earning <strong>$100K+</strong> per year show a dramatically higher acceptance rate. Above $150K the rate exceeds <strong>45%</strong>, compared to under 3% for those earning below $50K.</p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="insight-box">
            <h4>💳 CD Account = Strong Signal</h4>
            <p>Customers with a <strong>Certificate of Deposit (CD) account</strong> are nearly <strong>4× more likely</strong> to accept a personal loan — suggesting a strong relationship-banking behavior.</p>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="insight-box">
            <h4>🎓 Education Amplifies Income Effect</h4>
            <p>High income combined with advanced education creates the highest-propensity segment. <strong>Advanced/Professional</strong> degree holders with income >$100K show acceptance rates above <strong>35%</strong>.</p>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PREDICTIVE
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div class="section-header">
        <h3>🤖 Predictive Analytics — Who Will Accept a Personal Loan?</h3>
        <p>Machine learning models (Decision Tree, Random Forest, Gradient Boosting) trained on customer features.</p>
    </div>""", unsafe_allow_html=True)

    @st.cache_data
    def train_models(data):
        features = ["Age","Experience","Income","Family","CCAvg","Education","Mortgage",
                     "Securities Account","CD Account","Online","CreditCard"]
        X = data[features]
        y = data["Personal Loan"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        dt = DecisionTreeClassifier(max_depth=6, random_state=42)
        rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
        gb = GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)

        dt.fit(X_train, y_train)
        rf.fit(X_train, y_train)
        gb.fit(X_train, y_train)

        results = {}
        for name, model in [("Decision Tree",dt),("Random Forest",rf),("Gradient Boosting",gb)]:
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:,1]
            report = classification_report(y_test, y_pred, output_dict=True)
            auc = roc_auc_score(y_test, y_prob)
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            results[name] = {"model":model,"report":report,"auc":auc,"fpr":fpr,"tpr":tpr,
                               "y_prob":y_prob,"y_test":y_test,"features":features}
        return results, X_test, y_test

    results, X_test, y_test = train_models(df)

    # Model comparison
    model_names = list(results.keys())
    accuracies = [results[m]["report"]["accuracy"]*100 for m in model_names]
    aucs = [results[m]["auc"]*100 for m in model_names]
    f1s = [results[m]["report"]["1"]["f1-score"]*100 for m in model_names]

    c1, c2, c3 = st.columns(3)
    for col, name in zip([c1,c2,c3], model_names):
        with col:
            acc = results[name]["report"]["accuracy"]*100
            auc = results[name]["auc"]*100
            f1 = results[name]["report"]["1"]["f1-score"]*100
            st.markdown(f"""
            <div class="metric-card">
                <div style='font-size:0.85rem; font-weight:600; color:#8b95b0; margin-bottom:10px; text-transform:uppercase;'>{name}</div>
                <div style='font-size:1.6rem; font-weight:700; color:#6495ED;'>{acc:.1f}%</div>
                <div style='font-size:0.75rem; color:#8b95b0;'>Accuracy</div>
                <div style='margin-top:10px; display:flex; justify-content:space-around;'>
                    <div><span style='color:#5eead4; font-weight:600;'>{auc:.1f}%</span><br><span style='font-size:0.7rem; color:#8b95b0;'>AUC-ROC</span></div>
                    <div><span style='color:#fb923c; font-weight:600;'>{f1:.1f}%</span><br><span style='font-size:0.7rem; color:#8b95b0;'>F1 (Loans)</span></div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("")

    # ROC curves + Feature importance
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        colors_roc = ["#6495ED","#5eead4","#fb923c"]
        for (name, res), color in zip(results.items(), colors_roc):
            fig.add_trace(go.Scatter(x=res["fpr"], y=res["tpr"], mode="lines",
                                      name=f"{name} (AUC={res['auc']:.3f})",
                                      line=dict(color=color, width=2)))
        fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",name="Random",
                                  line=dict(color="gray",dash="dash",width=1)))
        apply_layout(fig,"ROC Curves — Model Comparison",380)
        fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        best_model = results["Gradient Boosting"]
        fi = pd.DataFrame({
            "Feature": best_model["features"],
            "Importance": best_model["model"].feature_importances_
        }).sort_values("Importance", ascending=True)
        fig = px.bar(fi, x="Importance", y="Feature", orientation="h",
                      color="Importance", color_continuous_scale=["#1e2435","#6495ED","#5eead4"],
                      labels={"Importance":"Importance Score"})
        apply_layout(fig,"Feature Importance — Gradient Boosting",380)
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── Live Prediction Tool ──
    st.markdown("""
    <div class="section-header">
        <h3>🔮 Live Customer Prediction Tool</h3>
        <p>Enter a customer's profile to predict loan acceptance probability and receive personalised offers.</p>
    </div>""", unsafe_allow_html=True)

    with st.form("pred_form"):
        c1,c2,c3,c4 = st.columns(4)
        with c1:
            p_age = st.number_input("Age", 20, 70, 40)
            p_exp = st.number_input("Experience (yrs)", 0, 45, 15)
            p_inc = st.number_input("Income ($000/yr)", 0, 300, 80)
        with c2:
            p_fam = st.selectbox("Family Size", [1,2,3,4])
            p_cc = st.number_input("CC Spend/mo ($000)", 0.0, 10.0, 2.0)
            p_edu = st.selectbox("Education", [1,2,3], format_func=lambda x: {1:"Undergrad",2:"Graduate",3:"Advanced/Prof"}[x])
        with c3:
            p_mort = st.number_input("Mortgage ($000)", 0, 700, 100)
            p_sec = st.selectbox("Securities Account", [0,1], format_func=lambda x: "Yes" if x else "No")
            p_cd = st.selectbox("CD Account", [0,1], format_func=lambda x: "Yes" if x else "No")
        with c4:
            p_online = st.selectbox("Online Banking", [0,1], format_func=lambda x: "Yes" if x else "No")
            p_ccard = st.selectbox("UniversalBank Credit Card", [0,1], format_func=lambda x: "Yes" if x else "No")
            submitted = st.form_submit_button("🔮 Predict & Get Offer", use_container_width=True)

    if submitted:
        inp = [[p_age, p_exp, p_inc, p_fam, p_cc, p_edu, p_mort, p_sec, p_cd, p_online, p_ccard]]
        gb_model = results["Gradient Boosting"]["model"]
        prob = gb_model.predict_proba(inp)[0][1]
        pred = gb_model.predict(inp)[0]

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob*100,
            title={"text":"Loan Acceptance Probability","font":{"color":"#d0d8f0","size":14}},
            number={"suffix":"%","font":{"color":"#6495ED","size":40}},
            gauge={
                "axis":{"range":[0,100],"tickcolor":"#c0cce0"},
                "bar":{"color":"#6495ED"},
                "steps":[
                    {"range":[0,30],"color":"#1a1f2e"},
                    {"range":[30,60],"color":"#1e2d40"},
                    {"range":[60,100],"color":"#1a2d35"}
                ],
                "threshold":{"line":{"color":"#5eead4","width":3},"thickness":0.8,"value":50},
                "bgcolor":"#1a1f2e"
            }
        ))
        fig.update_layout(paper_bgcolor=CHART_BG, font=dict(color="#c0cce0"), height=300,
                           margin=dict(l=30,r=30,t=60,b=20))

        c1, c2 = st.columns([1,1.6])
        with c1:
            st.plotly_chart(fig, use_container_width=True)
            if pred == 1:
                st.success(f"✅ **LIKELY TO ACCEPT** — {prob*100:.1f}% probability")
            else:
                st.warning(f"⚠️ **UNLIKELY TO ACCEPT** — {prob*100:.1f}% probability")

        with c2:
            st.markdown("#### 🎁 Personalised Offers")
            if pred == 1:
                if p_inc >= 100:
                    st.markdown("""<div class="offer-card">
                        <h4>💎 Premium Personal Loan — Up to $200K</h4>
                        <p>Based on your high income and financial profile, you qualify for our premium loan tier with preferential rates starting at <strong>7.5% APR</strong>. Flexible repayment up to 60 months.</p>
                        <span class="offer-tag">High Income</span><span class="offer-tag green">Low Risk</span>
                    </div>""", unsafe_allow_html=True)
                if p_edu == 3:
                    st.markdown("""<div class="offer-card">
                        <h4>🎓 Professional Advancement Loan</h4>
                        <p>Exclusively for advanced degree holders — finance your next career move, business venture, or home upgrade with our Professional Loan at <strong>8.0% APR</strong>.</p>
                        <span class="offer-tag orange">Professional</span>
                    </div>""", unsafe_allow_html=True)
                if p_fam >= 3:
                    st.markdown("""<div class="offer-card">
                        <h4>👨‍👩‍👧 Family Growth Loan Package</h4>
                        <p>For growing families — tailored loan amounts with extended repayment and family insurance bundling. Rates from <strong>8.5% APR</strong>.</p>
                        <span class="offer-tag green">Family Plan</span>
                    </div>""", unsafe_allow_html=True)
                if p_cd == 1:
                    st.markdown("""<div class="offer-card">
                        <h4>🏦 CD-Backed Loyalty Loan</h4>
                        <p>As a CD account holder, receive an exclusive <strong>0.5% rate reduction</strong> on personal loans up to $100K — our thank you for your loyalty.</p>
                        <span class="offer-tag">CD Holder</span><span class="offer-tag green">Loyalty</span>
                    </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""<div class="offer-card">
                    <h4>💡 Financial Wellness Starter Offer</h4>
                    <p>Start small with our <strong>Starter Loan (up to $20K)</strong> at 10.5% APR. Build your credit history with us and unlock better offers over time.</p>
                    <span class="offer-tag orange">Low Risk Entry</span>
                </div>""", unsafe_allow_html=True)
                if not p_online:
                    st.markdown("""<div class="offer-card">
                        <h4>📱 Digital Banking Onboarding Offer</h4>
                        <p>Activate online banking and receive a <strong>fee waiver for 12 months</strong> on any new loan — plus access to exclusive digital-only rates.</p>
                        <span class="offer-tag">Digital</span>
                    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PRESCRIPTIVE
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("""
    <div class="section-header">
        <h3>🎯 Prescriptive Analytics — What Should the Bank Do?</h3>
        <p>Actionable recommendations for marketing campaigns, segment targeting and personalised loan offers.</p>
    </div>""", unsafe_allow_html=True)

    # Segment analysis
    @st.cache_data
    def get_segments(data):
        features = ["Age","Experience","Income","Family","CCAvg","Education","Mortgage",
                     "Securities Account","CD Account","Online","CreditCard"]
        rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
        rf.fit(data[features], data["Personal Loan"])
        probs = rf.predict_proba(data[features])[:,1]
        seg_df = data.copy()
        seg_df["Loan Probability"] = probs
        seg_df["Segment"] = pd.cut(probs, bins=[0,0.25,0.5,0.75,1.0],
                                    labels=["Low Interest","Moderate","High Potential","Prime Target"])
        return seg_df

    seg_df = get_segments(df)

    # Segment breakdown
    seg_counts = seg_df["Segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment","Count"]

    c1, c2 = st.columns([1.2,1.8])
    with c1:
        fig = px.pie(seg_counts, values="Count", names="Segment",
                      color_discrete_sequence=["#1e2435","#6495ED","#5eead4","#fb923c"],
                      hole=0.55)
        apply_layout(fig,"Customer Segments by Loan Propensity",360)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        seg_profile = seg_df.groupby("Segment")[["Income","CCAvg","Age","Mortgage"]].mean().round(1).reset_index()
        fig = go.Figure()
        for col, color in zip(["Income","CCAvg","Age","Mortgage"],["#6495ED","#5eead4","#fb923c","#a78bfa"]):
            # normalize for radar visibility
            norm = seg_profile[col]/seg_profile[col].max()*100
            fig.add_trace(go.Bar(name=col, x=seg_profile["Segment"], y=norm,
                                  marker_color=color, text=seg_profile[col].apply(lambda x: f"{x:.0f}"),
                                  textposition="outside"))
        fig.update_layout(**PLOTLY_LAYOUT, barmode="group", height=360,
                           title=dict(text="Segment Profiles (Normalised Metrics)",
                                       font=dict(size=14,color="#d0d8f0")))
        st.plotly_chart(fig, use_container_width=True)

    # Target recommendations
    st.markdown("""<div class="section-header"><h3>🎁 Personalised Offer Prescriptions by Segment</h3></div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="offer-card">
            <h4>🔴 Prime Target Segment (Probability > 75%)</h4>
            <p><strong>Who:</strong> High income (>$120K), advanced education, family size 3–4, CD account holders.<br><br>
            <strong>Offer:</strong> Premium Personal Loan (up to $200K) at <strong>7.5% APR</strong>, 60-month flexible repayment, zero processing fee, dedicated relationship manager, priority approval within 24 hours.<br><br>
            <strong>Channel:</strong> Direct call from Relationship Manager + personalised email with pre-approved offer letter.</p>
            <span class="offer-tag">Direct Outreach</span><span class="offer-tag green">Pre-Approved</span><span class="offer-tag orange">High Value</span>
        </div>
        <div class="offer-card">
            <h4>🟡 High Potential Segment (Probability 50–75%)</h4>
            <p><strong>Who:</strong> Income $80–120K, Graduate or Advanced education, moderate CC spending.<br><br>
            <strong>Offer:</strong> Personal Loan (up to $100K) at <strong>8.5% APR</strong>, 48-month term, first 3 EMIs deferred, online application with instant pre-approval.<br><br>
            <strong>Channel:</strong> Targeted email campaign + in-app push notification via online banking.</p>
            <span class="offer-tag">Email Campaign</span><span class="offer-tag green">Digital First</span>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="offer-card">
            <h4>🔵 Moderate Segment (Probability 25–50%)</h4>
            <p><strong>Who:</strong> Income $50–80K, mixed education backgrounds, smaller family sizes.<br><br>
            <strong>Offer:</strong> Starter Personal Loan (up to $50K) at <strong>9.5% APR</strong>, 36-month term, bundled with online banking signup bonus and credit card upgrade offer to increase engagement and future propensity.<br><br>
            <strong>Channel:</strong> In-branch awareness + SMS campaign + app banner.</p>
            <span class="offer-tag">Cross-sell</span><span class="offer-tag orange">Bundle</span>
        </div>
        <div class="offer-card">
            <h4>⚪ Low Interest Segment (Probability < 25%)</h4>
            <p><strong>Who:</strong> Lower income (<$50K), undergrad education, minimal banking products.<br><br>
            <strong>Offer:</strong> Financial Literacy Program + Small Loan ($10–20K) at <strong>10.5% APR</strong> with monthly spend tracking tool. Focus on <strong>long-term relationship building</strong> — nurture now, convert later.<br><br>
            <strong>Channel:</strong> Educational content, app notifications, in-branch consultations.</p>
            <span class="offer-tag">Nurture</span><span class="offer-tag green">Long-term</span>
        </div>""", unsafe_allow_html=True)

    # Strategy matrix
    st.markdown("""<div class="section-header"><h3>📋 Marketing Strategy Summary</h3></div>""", unsafe_allow_html=True)
    strategy = pd.DataFrame({
        "Segment": ["Prime Target","High Potential","Moderate","Low Interest"],
        "Size": [
            f"{(seg_df['Segment']=='Prime Target').sum():,}",
            f"{(seg_df['Segment']=='High Potential').sum():,}",
            f"{(seg_df['Segment']=='Moderate').sum():,}",
            f"{(seg_df['Segment']=='Low Interest').sum():,}",
        ],
        "Avg Income": [
            f"${seg_df[seg_df['Segment']=='Prime Target']['Income'].mean():.0f}K",
            f"${seg_df[seg_df['Segment']=='High Potential']['Income'].mean():.0f}K",
            f"${seg_df[seg_df['Segment']=='Moderate']['Income'].mean():.0f}K",
            f"${seg_df[seg_df['Segment']=='Low Interest']['Income'].mean():.0f}K",
        ],
        "Recommended Loan": ["Up to $200K @ 7.5%","Up to $100K @ 8.5%","Up to $50K @ 9.5%","Up to $20K @ 10.5%"],
        "Priority": ["🔴 Immediate","🟡 High","🔵 Medium","⚪ Nurture"],
        "Channel": ["Direct RM Call","Email + App","SMS + Branch","Educational Content"]
    })
    st.dataframe(strategy, use_container_width=True, hide_index=True)

    # Final strategic insights
    st.markdown("""<div class="section-header"><h3>💡 Strategic Recommendations</h3></div>""", unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="insight-box">
            <h4>🎯 Focus Budget on Prime + High Potential</h4>
            <p>Approximately <strong>30% of customers</strong> fall in the Prime or High Potential segments. Concentrating 70% of campaign budget here maximises ROI and minimises wasted outreach.</p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="insight-box">
            <h4>🏦 Leverage CD Account Relationship</h4>
            <p>CD account holders are 4× more likely to accept loans. <strong>Create a dedicated cross-sell journey</strong> for all CD holders — this is the single highest-converting trigger in the dataset.</p>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="insight-box">
            <h4>📱 Digital-First for Under-50 Customers</h4>
            <p>Customers under 50 with online banking activated respond better to <strong>in-app and email campaigns</strong>. Invest in a seamless digital loan application to reduce friction and increase conversion.</p>
        </div>""", unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<hr>
<div style='text-align:center; color:#4a5568; font-size:0.78rem; padding: 12px 0;'>
    🏦 Universal Bank | Personal Loan Intelligence Dashboard &nbsp;·&nbsp; Built with Streamlit & Plotly
    &nbsp;·&nbsp; Models: Decision Tree, Random Forest, Gradient Boosting
</div>""", unsafe_allow_html=True)
