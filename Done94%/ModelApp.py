import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta
from io import BytesIO

def calculate_percentage_change(original_cost, new_cost):
    return (new_cost - original_cost) / original_cost

def get_month_labels(start_month, end_month):
    current_date = datetime.now()
    return [(current_date + relativedelta(months=i)).strftime('%B-%Y') for i in range(start_month, end_month + 1)]

def to_excel(df1, df2, df3):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df1.to_excel(writer, sheet_name='Phase 1', index=False)
        df2.to_excel(writer, sheet_name='Phase 2', index=False)
        df3.to_excel(writer, sheet_name='Total', index=False)
    processed_data = output.getvalue()
    return processed_data

def predict_impressions():
    # Load the trained SARIMA models
    sarima_phase1_model = joblib.load('sarima_phase1_model.pkl')
    sarima_phase2_model = joblib.load('sarima_phase2_model.pkl')

    # User input: Enter the month range (0 to 60 months ahead) and costs
    start_month = 0  # Current month
    st.write("""
    ## TNDM-Impressions prediction based on the cost in phases.
    """)
    end_month = st.slider("Select number of months ahead for prediction (up to 60):", 0, 60, 12)
    phase1_cost = st.number_input("Enter Phase 1 cost($):", min_value=0.0,  step=2400.0)
    phase2_cost = st.number_input("Enter Phase 2 cost($):", min_value=0.0,  step=2400.0)

    # Calculate percentage changes in costs
    phase1_cost_change = calculate_percentage_change(142000, phase1_cost)
    phase2_cost_change = calculate_percentage_change(479000, phase2_cost)

    # Generate predictions for the specified months
    predicted_phase1 = sarima_phase1_model.predict(start=start_month, end=end_month, exog=[phase1_cost]*(end_month + 1))
    predicted_phase2 = sarima_phase2_model.predict(start=start_month, end=end_month, exog=[phase2_cost]*(end_month + 1))

    # Adjust predictions based on the percentage change in costs
    predicted_phase1 *= (1 + phase1_cost_change)
    predicted_phase2 *= (1 + phase2_cost_change)

    # Calculate the total impressions (sum of Phase 1 and Phase 2)
    total_predicted_impressions = predicted_phase1 + predicted_phase2

    # Get month labels
    months = get_month_labels(start_month, end_month)

    # Selecting points to label (e.g., first, last, and three equally spaced)
    indices_to_label = [0, end_month // 4, end_month // 2, 3 * end_month // 4, end_month]

    # Prepare data for Excel file
    df_phase1 = pd.DataFrame({'Month': months, 'Phase 1 Predictions': predicted_phase1})
    df_phase2 = pd.DataFrame({'Month': months, 'Phase 2 Predictions': predicted_phase2})
    df_total = pd.DataFrame({'Month': months, 'Total Predictions': total_predicted_impressions})

    # Plotting Phase 1 Predictions
    plt.figure(figsize=(10, 4))
    plt.plot(months, predicted_phase1, label='Phase 1 Predicted Impressions', color='blue')
    plt.scatter([months[i] for i in indices_to_label], [predicted_phase1[i] for i in indices_to_label], color='red')
    for i in indices_to_label:
        plt.annotate(f"{predicted_phase1[i]:.0f}", (months[i], predicted_phase1[i]), textcoords="offset points", xytext=(0,10), ha='center')
    plt.xlabel('Month-Year')
    plt.ylabel('Impressions')
    plt.title('Phase 1 Impressions')
    plt.xticks(rotation=270)
    plt.grid(True)
    st.pyplot(plt)

    # Plotting Phase 2 Predictions
    plt.figure(figsize=(10, 4))
    plt.plot(months, predicted_phase2, label='Phase 2 Predicted Impressions', color='orange')
    plt.scatter([months[i] for i in indices_to_label], [predicted_phase2[i] for i in indices_to_label], color='green')
    for i in indices_to_label:
        plt.annotate(f"{predicted_phase2[i]:.0f}", (months[i], predicted_phase2[i]), textcoords="offset points", xytext=(0,10), ha='center')
    plt.xlabel('Month-Year')
    plt.ylabel('Impressions')
    plt.title('Phase 2 Impressions')
    plt.xticks(rotation=270)
    plt.grid(True)
    st.pyplot(plt)

    # Checkbox to toggle the display of the total predicted impressions chart
    show_total_chart = st.checkbox("Show Total Predicted Impressions Chart", value=False)

    if show_total_chart:
        # Plotting Total Predicted Impressions
        plt.figure(figsize=(10, 4))
        plt.plot(months, total_predicted_impressions, label='Total Predicted Impressions', color='purple')
        plt.scatter([months[i] for i in indices_to_label], [total_predicted_impressions[i] for i in indices_to_label], color='darkred')
        for i in indices_to_label:
            plt.annotate(f"{total_predicted_impressions[i]:.0f}", (months[i], total_predicted_impressions[i]), textcoords="offset points", xytext=(0,10), ha='center')
        plt.xlabel('Month-Year')
        plt.ylabel('Impressions')
        plt.title('Total Impressions')
        plt.xticks(rotation=270)
        plt.grid(True)
        st.pyplot(plt)

    # Download button for Excel file
    excel_file = to_excel(df_phase1, df_phase2, df_total)
    st.download_button(label='Download Prediction Data as Excel',
                       data=excel_file,
                       file_name='predictions.xlsx',
                       mime='application/vnd.ms-excel')

def main_page():
    predict_impressions()

def explanation_page():
    st.title("About the Prediction Model")
    st.write("""
    ## How the Model Works
    
    
    - The **Model** is trained with SARIMA Algorithm packaged in Pickel File.
    - **Model Training**: the model was trained on the provided Data from an existing model.
    - **Predictions**: Describe how the model makes predictions and how costs impact these predictions.
    - **Use Case**: Discuss the specific use case of your model in context.

    Feel free to add more information, including diagrams or external links for users who might want to learn more.
    """)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a Page", ["Predictions", "Model Explanation"])

if page == "Predictions":
    main_page()
elif page == "Model Explanation":
    explanation_page()
