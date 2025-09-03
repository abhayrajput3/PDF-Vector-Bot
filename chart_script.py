import plotly.graph_objects as go
import plotly.io as pio

# Data for the comparison table with clear advantage indicators
categories = [
    "<b>API Integration</b>",
    "<b>Models Used</b>", 
    "<b>Cost Structure</b>",
    "<b>Context Window</b>",
    "<b>Setup Complexity</b>",
    "<b>Performance</b>",
    "<b>Features</b>"
]

openai_details = [
    "ChatOpenAI<br>OpenAIEmbeddings<br>OPENAI_API_KEY",
    "GPT-3.5-turbo<br>text-embedding-ada-002", 
    "Paid only<br>($0.002/1K tokens)",
    "4K-16K tokens",
    "Medium complexity<br>(Credit card required)",
    "✅ Proven, reliable<br>Industry standard",
    "✅ Text-focused<br>✅ Mature ecosystem"
]

gemini_details = [
    "ChatGoogleGenerativeAI<br>GoogleGenerativeAIEmbeddings<br>GOOGLE_API_KEY",
    "gemini-2.0-flash<br>text-embedding-004",
    "✅ Free tier available<br>15 req/min, 1500/day<br>+ paid options",
    "✅ Up to 1M tokens<br>(Much larger)",
    "✅ Simple setup<br>✅ Free sign-up", 
    "✅ Fast processing<br>✅ Multimodal-native",
    "✅ Multimodal ready<br>✅ Large context"
]

# Create the table with improved styling and contrast
fig = go.Figure(data=[go.Table(
    columnwidth=[150, 300, 300],
    header=dict(
        values=['<b>Category</b>', '<b>OpenAI</b>', '<b>Google Gemini</b>'],
        fill_color='#1FB8CD',
        font=dict(color='white', size=18),
        align='center',
        height=60,
        line=dict(color='white', width=2)
    ),
    cells=dict(
        values=[categories, openai_details, gemini_details],
        fill_color=[['#f8f9fa']*len(categories), 
                   ['#f0f8f0']*len(categories), 
                   ['#f0f4f8']*len(categories)],
        font=dict(color='black', size=14),
        align=['center', 'left', 'left'],
        height=90,
        line=dict(color='#ddd', width=1)
    )
)])

fig.update_layout(
    title=dict(
        text="<b>OpenAI vs Google Gemini: PDF Analyzer Comparison</b>",
        font=dict(size=24),
        x=0.5,
        y=0.95
    ),
    font=dict(family="Arial, sans-serif"),
    paper_bgcolor='white'
)

# Save the chart
fig.write_image("openai_vs_gemini_comparison.png", width=1500, height=800)