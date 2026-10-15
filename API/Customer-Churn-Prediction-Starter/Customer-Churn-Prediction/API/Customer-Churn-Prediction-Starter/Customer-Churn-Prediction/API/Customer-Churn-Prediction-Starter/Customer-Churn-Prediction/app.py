from pathlib import Path
from typing import Any
import os
import requests
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title='ChurnGuard AI | From Data to Decisions', page_icon='✦', layout='wide', initial_sidebar_state='collapsed')
API_BASE_URL = os.getenv('API_BASE_URL', 'http://127.0.0.1:8000').rstrip('/')
PREDICT_URL = f'{API_BASE_URL}/predict'
HEALTH_URL = f'{API_BASE_URL}/health'
ROOT = Path(__file__).resolve().parent
DATA_CANDIDATES = [ROOT / 'Customer Churn.csv', ROOT.parent / 'Customer Churn.csv', ROOT.parent.parent / 'Customer Churn.csv', ROOT.parent.parent.parent / 'Customer Churn.csv']

st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root {--bg:#0b1024;--card:#151d39;--purple:#ad91ff;--mint:#61e4d2;}
.stApp {background:radial-gradient(ellipse at 85% 0%,#302359 0%,transparent 34%),linear-gradient(155deg,#090e20,#111a32 70%,#0b1024);color:#f4f5ff;font-family:'DM Sans',sans-serif;}
[data-testid='stSidebar'] {background:linear-gradient(180deg,#141936,#0d1329);border-right:1px solid #353653;}
h1,h2,h3 {font-family:'Space Grotesk',sans-serif!important;color:#f5f4ff!important;letter-spacing:-.025em;}
p, label, [data-testid='stMarkdownContainer'] {color:#e0e3f6;}
.block-container {max-width:1350px;padding-top:2.4rem;padding-bottom:3rem;}
.hero {padding:3rem 3.2rem;border-radius:26px;background:radial-gradient(circle at 85% 30%,#5d3da5 0%,transparent 35%),linear-gradient(110deg,#20224d,#121a38);border:1px solid #554c84;box-shadow:0 18px 60px #03061665;animation:rise .6s ease-out;}
.hero h1 {font-size:clamp(2.2rem,4vw,4.3rem);line-height:1.1;margin:.45rem 0;}
.eyebrow {color:#71e6d4!important;letter-spacing:.2em;font-size:.76rem;font-weight:700;}
.accent {color:#c6afff;}
.card {background:linear-gradient(145deg,#1c2547,#141b36);border:1px solid #3a4165;border-radius:20px;padding:1.35rem 1.5rem;margin:.55rem 0;min-height:130px;box-shadow:0 12px 30px #04081636;animation:rise .7s ease-out;}
.card h3 {margin:.1rem 0 .55rem;font-size:1.1rem;}.card p {color:#bfc8e3;margin:0;}
.kpi {font-size:2rem;font-weight:700;color:#bfa9ff;font-family:'Space Grotesk';}
.tag {display:inline-block;background:#302a59;color:#d7c9ff;border:1px solid #67559b;padding:.3rem .7rem;border-radius:30px;font-size:.78rem;}
[data-testid='stMetric'] {background:#1b2444;border:1px solid #3a4265;border-radius:18px;padding:1rem;}
.stButton>button[kind='primary'],[data-testid='stFormSubmitButton'] button {background:linear-gradient(90deg,#8f6bea,#ad87fc)!important;color:white!important;border:none!important;border-radius:12px!important;font-weight:700!important;}
.stButton>button {border-radius:12px;}
[data-testid='stAlert'] {border-radius:14px;}
@keyframes rise {from{opacity:0;transform:translateY(18px)}to{opacity:1;transform:translateY(0)}}

/* Strong contrast for all KPI numbers, metrics and chart labels */
.kpi {color:#ffffff!important;font-size:2.25rem!important;font-weight:800!important;text-shadow:0 2px 12px rgba(0,0,0,.25);}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] *,
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] *,
[data-testid="stMetricDelta"], [data-testid="stMetricDelta"] * {color:#ffffff!important;opacity:1!important;}
[data-testid="stMetricValue"] {font-size:2rem!important;font-weight:800!important;}
[data-testid="stMetricLabel"] {color:#dbe5ff!important;font-weight:600!important;}
.card p {color:#e3e9ff!important;}
.top-brand {font-family:'Space Grotesk',sans-serif;font-size:1.5rem;font-weight:800;color:#ffffff;padding:.25rem 0 1rem;letter-spacing:.025em;}
.top-brand span {font-family:'DM Sans',sans-serif;font-size:.72rem;color:#b5ffed;font-weight:700;letter-spacing:.12em;}
/* Horizontal top navigation: pill-style controls */
[data-testid="stRadio"] > div[role="radiogroup"] {display:flex;flex-wrap:wrap;gap:.65rem;padding:.5rem 0 1rem;}
[data-testid="stRadio"] > div[role="radiogroup"] > label {background:#1b2445;border:1px solid #5c608e;border-radius:13px;padding:.75rem 1rem;min-width:130px;justify-content:center;transition:background .2s,transform .2s;}
[data-testid="stRadio"] > div[role="radiogroup"] > label:hover {background:#353365;transform:translateY(-2px);}
[data-testid="stRadio"] > div[role="radiogroup"] > label p {color:#ffffff!important;font-weight:700!important;}
[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) {background:#5d439f;border-color:#c3aaff;box-shadow:0 0 0 1px #c3aaff55;}
[data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child {display:none;}
</style>''', unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_data():
    for path in DATA_CANDIDATES:
        if path.is_file():
            return pd.read_csv(path), str(path)
    return None, None

def card(title, body, value=None):
    number = f'<div class="kpi">{value}</div>' if value is not None else ''
    st.markdown(f'<div class="card"><h3>{title}</h3>{number}<p>{body}</p></div>', unsafe_allow_html=True)

def section(title, caption=None):
    st.title(title)
    if caption: st.caption(caption)

def chart_style(fig):
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font_color='#e7e6fb',margin=dict(l=12,r=12,t=42,b=15),legend_title_text='',colorway=['#a98aff','#61e4d2','#f7a7c4','#ffc97b'])
    fig.update_xaxes(gridcolor='#303858',zerolinecolor='#303858')
    fig.update_yaxes(gridcolor='#303858',zerolinecolor='#303858')
    return fig

if 'page' not in st.session_state:
    st.session_state.page = 'Home'

st.markdown('<div class="top-brand">✦ CHURNGUARD AI <span> / FROM DATA TO DECISIONS</span></div>', unsafe_allow_html=True)
pages = ['Home', 'Analytics', 'Prediction', 'Model Performance', 'About Project']
icons = {'Home':'⌂  Home', 'Analytics':'▥  Analytics', 'Prediction':'✧  Prediction',
         'Model Performance':'◈  Model Performance', 'About Project':'ⓘ  About Project'}
page = st.radio('Navigate', pages, index=pages.index(st.session_state.page),
                format_func=lambda name: icons[name], horizontal=True, label_visibility='collapsed')
st.session_state.page = page

try:
    health = requests.get(HEALTH_URL, timeout=2)
    if health.ok and health.json().get('model_loaded'):
        st.caption('🟢 API connected · Model ready')
    else:
        st.caption('🟠 API reachable · Model not ready')
except requests.RequestException:
    st.caption('🔴 API offline — start FastAPI to make predictions')

if page == 'Home':
    st.markdown('''<div class="hero"><div class="eyebrow">MACHINE LEARNING CAPSTONE · INTELLIGENT ANALYTICS</div><h1>From Data<br><span class="accent">to Decisions.</span></h1><p style="font-size:1.15rem;max-width:680px;color:#d7d6ef">Discover customer churn patterns, explore the evidence, and estimate churn risk with a trained machine-learning pipeline.</p><span class="tag">✦ CHURNGUARD AI</span></div>''',unsafe_allow_html=True)
    st.write('')
    a,b,c=st.columns(3)
    with a: card('01 · Explore','Understand the dataset and customer behavior through interactive visualizations.')
    with b: card('02 · Predict','Enter a customer profile and request a live prediction from FastAPI.')
    with c: card('03 · Evaluate','Review held-out model metrics and the trade-offs behind them.')
    if st.button('Start a prediction →',type='primary'):
        st.session_state.page='Prediction';st.rerun()
    st.subheader('How it works')
    st.markdown('**Customer data** → **Validation** → **Feature engineering** → **Saved ML pipeline** → **Churn estimate**')

elif page == 'Analytics':
    section('Customer Analytics','Interactive exploratory analysis from the original Customer Churn.csv dataset. Associations do not prove causation.')
    df,path=load_data()
    if df is None:
        st.warning('Dataset not found. Keep Customer Churn.csv in the final pro folder (or an ancestor of app.py).')
    elif 'Churn' not in df.columns:
        st.error('The dataset does not contain a Churn column.')
    else:
        total=len(df);churn=int((df['Churn']==1).sum());retained=int((df['Churn']==0).sum())
        a,b,c=st.columns(3)
        a.metric('Original records',f'{total:,}');b.metric('Churned',f'{churn:,}');c.metric('Retained',f'{retained:,}')
        x,y=st.columns(2)
        with x:
            counts=df['Churn'].map({0:'Retained',1:'Churned'}).value_counts().reset_index();counts.columns=['Status','Customers']
            fig=px.pie(counts,names='Status',values='Customers',hole=.67,color='Status',color_discrete_map={'Retained':'#61e4d2','Churned':'#aa8cff'},title='Customer churn distribution')
            st.plotly_chart(chart_style(fig),use_container_width=True)
        with y:
            if 'Complains' in df:
                rates=df.groupby('Complains')['Churn'].mean().mul(100).reset_index();rates['Complains']=rates['Complains'].map({0:'No complaint',1:'Complaint'})
                fig=px.bar(rates,x='Complains',y='Churn',title='Churn rate by complaint status',labels={'Churn':'Churn rate (%)'},color='Complains',color_discrete_sequence=['#61e4d2','#aa8cff'])
                st.plotly_chart(chart_style(fig),use_container_width=True)
        left,right=st.columns(2)
        with left:
            if 'Tariff Plan' in df:
                g=df.groupby('Tariff Plan')['Churn'].mean().mul(100).reset_index()
                fig=px.bar(g,x='Tariff Plan',y='Churn',title='Churn rate by tariff plan',labels={'Churn':'Churn rate (%)'})
                st.plotly_chart(chart_style(fig),use_container_width=True)
        with right:
            if 'Age Group' in df:
                g=df.groupby('Age Group')['Churn'].mean().mul(100).reset_index()
                fig=px.bar(g,x='Age Group',y='Churn',title='Churn rate by age group',labels={'Churn':'Churn rate (%)'})
                st.plotly_chart(chart_style(fig),use_container_width=True)
        numeric=[col for col in ['Subscription  Length','Seconds of Use','Frequency of use','Customer Value'] if col in df]
        if numeric:
            feature=st.selectbox('Compare a numeric feature by churn status',numeric)
            plot_df=df.copy();plot_df['Churn status']=plot_df['Churn'].map({0:'Retained',1:'Churned'})
            fig=px.box(plot_df,x='Churn status',y=feature,color='Churn status',title=f'{feature} by churn status',color_discrete_sequence=['#61e4d2','#aa8cff'])
            st.plotly_chart(chart_style(fig),use_container_width=True)
        st.caption('These charts describe the original dataset; training data was cleaned separately in the notebook.')

elif page == 'Prediction':
    section('Live Churn Prediction','Enter a customer profile. The API validates inputs and computes Avg_Usage_Duration before running the saved model.')
    with st.form('customer_form'):
        a,b,c=st.columns(3)
        with a:
            st.markdown('### Usage')
            call_failure=st.number_input('Call Failure',0,36,8)
            seconds_of_use=st.number_input('Seconds of Use',0,17090,4370)
            frequency_of_use=st.number_input('Frequency of use',0,255,71)
            frequency_of_sms=st.number_input('Frequency of SMS',0,522,5)
            distinct_called_numbers=st.number_input('Distinct Called Numbers',0,97,17)
        with b:
            st.markdown('### Subscription')
            complains=st.selectbox('Complains',[0,1],format_func=lambda x:'No (0)' if x==0 else 'Yes (1)')
            subscription_length=st.number_input('Subscription Length',3,47,38)
            charge_amount=st.selectbox('Charge Amount',list(range(11)))
            tariff_plan=st.selectbox('Tariff Plan',[1,2],format_func=lambda x:f'Plan {x}')
            age_group=st.selectbox('Age Group',[1,2,3,4,5])
        with c:
            st.markdown('### Customer profile')
            age=st.number_input('Age',15,55,30)
            customer_value=st.number_input('Customer Value',0.0,2165.28,197.64,step=1.0)
            st.info('Inputs are limited to observed dataset ranges, not universal business limits.')
        submitted=st.form_submit_button('✦ Predict customer churn',type='primary',use_container_width=True)
    if submitted:
        payload:dict[str,Any]={
            'Call  Failure':int(call_failure),'Complains':int(complains),
            'Subscription  Length':int(subscription_length),'Charge  Amount':int(charge_amount),
            'Seconds of Use':int(seconds_of_use),'Frequency of use':int(frequency_of_use),
            'Frequency of SMS':int(frequency_of_sms),'Distinct Called Numbers':int(distinct_called_numbers),
            'Age Group':int(age_group),'Tariff Plan':int(tariff_plan),
            'Age':int(age),'Customer Value':float(customer_value)}
        try:
            with st.spinner('Requesting prediction from FastAPI...'):
                response=requests.post(PREDICT_URL,json=payload,timeout=10)
            if response.status_code==422:
                st.error('Backend validation failed. Please check the entered values.')
            elif not response.ok:
                try: detail=response.json().get('detail','Backend returned an error.')
                except ValueError: detail='Backend returned an unreadable response.'
                st.error(f'Prediction unavailable: {detail}')
            else:
                result=response.json();st.session_state.last_result=result
        except requests.Timeout:
            st.error('Request timed out. Please try again.')
        except requests.ConnectionError:
            st.error('Cannot connect to FastAPI. Start the backend and check API_BASE_URL.')
        except requests.RequestException:
            st.error('The prediction request could not be completed.')
    if 'last_result' in st.session_state:
        result=st.session_state.last_result
        st.markdown('### ✦ Prediction result')
        if result.get('predicted_class')==1: st.error('Predicted outcome: CHURN · This customer may leave.')
        else: st.success('Predicted outcome: NO CHURN · This customer is predicted to stay.')
        prob=result.get('churn_probability')
        if prob is not None and 0<=float(prob)<=1:
            p=float(prob);left,right=st.columns([1,2]);left.metric('Estimated churn probability',f'{p:.1%}')
            fig=go.Figure(go.Indicator(mode='gauge+number',value=p*100,number={'suffix':'%'},gauge={'axis':{'range':[0,100]},'bar':{'color':'#b59aff'},'bgcolor':'#242b4a','steps':[{'range':[0,50],'color':'#1b4b55'},{'range':[50,100],'color':'#573b65'}]}))
            fig.update_layout(height=240,paper_bgcolor='rgba(0,0,0,0)',font_color='#e7e6fb',margin=dict(t=20,b=15,l=20,r=20))
            with right: st.plotly_chart(fig,use_container_width=True)
            st.caption(result.get('probability_note') or 'Estimated model probability, not a guaranteed outcome.')
        else: st.info('The model did not provide a churn probability.')
    with st.expander('How does the prediction work?'):
        st.write('Streamlit sends the 12 inputs to FastAPI. The API validates them, computes Avg_Usage_Duration = Seconds of Use / (Frequency of use + 1), and runs the saved pipeline.')

elif page == 'Model Performance':
    section('Model Performance','Results from the capstone presentation; the held-out test metrics are not live API statistics.')
    st.markdown('### Final held-out test results · Tuned Random Forest')
    a,b,c,d=st.columns(4)
    a.metric('Precision','83.53%');b.metric('Recall','82.56%');c.metric('F1-score','83.04%');d.metric('PR-AUC','91.33%')
    left,right=st.columns(2)
    with left:
        fig=px.imshow([[465,14],[15,71]],text_auto=True,x=['Predicted: No Churn','Predicted: Churn'],y=['Actual: No Churn','Actual: Churn'],color_continuous_scale=[[0,'#1b2444'],[1,'#a98aff']],title='Held-out test confusion matrix')
        st.plotly_chart(chart_style(fig),use_container_width=True)
    with right:
        models=pd.DataFrame({'Model':['Logistic Regression','Random Forest','Gradient Boosting'],'F1':[.6152,.8725,.8306],'PR-AUC':[.7454,.9561,.9273],'Recall':[.8641,.8122,.7659]})
        metric=st.selectbox('Baseline cross-validation comparison',['F1','PR-AUC','Recall'])
        fig=px.bar(models,x='Model',y=metric,color='Model',text_auto='.3f',title=f'Baseline 5-fold CV · {metric}')
        st.plotly_chart(chart_style(fig),use_container_width=True)
    card('Important distinction','Baseline model comparison uses cross-validation. Final metrics above use the held-out test set. Do not compare them as if they were the same evaluation.')

elif page == 'About Project':
    section('About ChurnGuard AI','Intelligent Customer Churn Prediction · Machine Learning Capstone')
    st.markdown('''<div class="hero"><div class="eyebrow">THE MISSION</div><h1>Predict who may leave.<br><span class="accent">Understand the signals.</span></h1><p>Support informed customer-retention decisions with machine-learning estimates.</p></div>''',unsafe_allow_html=True)
    st.write('')
    a,b,c=st.columns(3)
    with a:card('Data','3,150 original records · 14 original columns')
    with b:card('Preparation','Duplicates and conflicting feature-label rows investigated and removed')
    with c:card('Learning','Logistic Regression · Random Forest · Gradient Boosting')
    st.markdown('### Project workflow')
    st.markdown('**EDA** → **Cleaning** → **Feature engineering** → **Stratified split** → **Model comparison** → **Tuning** → **Final evaluation** → **Saved pipeline** → **FastAPI + Streamlit**')
    st.markdown('### Technology stack')
    st.write('Python · pandas · scikit-learn · joblib · FastAPI · Streamlit · Plotly')
    st.info('The prediction is decision support, not proof of future behavior or the effectiveness of a retention intervention.')

st.divider()
st.caption('✦ CHURNGUARD AI   •   MACHINE LEARNING CAPSTONE   •   FROM DATA TO DECISIONS')

