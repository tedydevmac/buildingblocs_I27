import streamlit as st
import pandas as pd
import numpy as np

# ========== TEXT DISPLAY FUNCTIONS ==========
# st.title("Title!") --> h1
# st.header("Header!") --> h2
# st.subheader("Subheader!") --> h3
# st.write("Text!") --> p (versatile function that handles multiple data types)
# st.markdown(body="", unsafe_allow_html=False, help=None) --> write markdown
# st.text("Plain text") --> display plain text
# st.code(body="", language="", line_numbers="") --> display text in code format
# st.latex(r"\LaTeX") --> render LaTeX expressions
# st.caption("Small text") --> display small caption text

# ========== DATA DISPLAY FUNCTIONS ==========
# st.dataframe(df) --> display interactive dataframe
# st.table(df) --> display static table
# st.metric(label="", value="", delta="") --> display metric with optional delta
# st.json(object) --> display JSON object

# ========== CHART AND VISUALIZATION FUNCTIONS ==========
# st.line_chart(data) --> simple line chart
# st.area_chart(data) --> simple area chart
# st.bar_chart(data) --> simple bar chart
# st.pyplot(fig) --> display matplotlib figure
# st.plotly_chart(fig) --> display plotly figure
# st.altair_chart(chart) --> display altair chart
# st.map(data) --> display map with coordinates

# ========== MEDIA DISPLAY FUNCTIONS ==========
# st.image(image, caption="", width=None) --> display image
# st.audio(data) --> display audio player
# st.video(data) --> display video player

# ========== INPUT WIDGETS ==========
# st.button(label) --> clickable button, returns True when clicked
# st.checkbox(label, value=False) --> checkbox widget
# st.radio(label, options) --> radio button selection
# st.selectbox(label, options) --> dropdown selection
# st.multiselect(label, options) --> multiple selection widget
# st.slider(label, min_value, max_value, value) --> slider widget
# st.select_slider(label, options) --> slider with custom options
# st.text_input(label, value="") --> single line text input
# st.text_area(label, value="") --> multi-line text input
# st.number_input(label, min_value, max_value) --> number input
# st.date_input(label) --> date picker
# st.time_input(label) --> time picker
# st.file_uploader(label, type=None) --> file upload widget
# st.color_picker(label) --> color picker widget

# ========== LAYOUT FUNCTIONS ==========
# st.columns(spec) --> create columns layout
# st.container() --> group elements in a container
# st.expander(label) --> collapsible container
# st.sidebar --> add widgets to sidebar
# st.tabs(tab_names) --> create tabbed interface
# st.empty() --> placeholder that can be filled later

# ========== STATUS AND FEEDBACK FUNCTIONS ==========
# st.success("Success message") --> green success message
# st.info("Info message") --> blue info message
# st.warning("Warning message") --> yellow warning message
# st.error("Error message") --> red error message
# st.exception(exception) --> display exception with traceback
# st.balloons() --> show falling balloons animation
# st.snow() --> show falling snow animation
# st.toast("Toast message") --> show temporary toast notification

# ========== PROGRESS AND STATUS ==========
# st.progress(value) --> display progress bar
# st.spinner(text="Loading...") --> show loading spinner
# st.status(label) --> expandable status container

# ========== ADVANCED LAYOUT ==========
# st.form(key) --> create form container for batch submission
# st.form_submit_button(label) --> submit button for forms (only works inside forms)
# st.popover(label) --> create popover container
# st.dialog(title) --> create modal dialog

# ========== UTILITY FUNCTIONS ==========
# st.stop() --> stop execution at this point
# st.rerun() --> rerun the script from top
# st.cache_data --> decorator to cache function results
# st.cache_resource --> decorator to cache global resources
# st.session_state --> dictionary-like object to store state across reruns

x = np.arange(-100, 100) / 10
y1 = np.sin(x)
y2 = np.cos(x)

data = {
    'x value': x,
    'Sine': y1,
    'Cosine': y2,
    'x axis': 0
}

df = pd.DataFrame(data)
st.dataframe(df)
# st.table(df) == st.dataframe(df) except it is static (cannot change during runtime)

st.line_chart(data=data, x_label="x value", y_label="y value", color=["#0000FF", "#FF0000", "#000"], x="x value", y=["Sine", "Cosine", "x axis"])