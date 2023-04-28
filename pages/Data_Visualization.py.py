import streamlit as st
import pandas as pd
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv("Mall_Customers.csv")

# Set up the page layout
st.set_page_config(page_title="Customer Segmentation", page_icon=":bar_chart:", layout="wide")

# Set up the sidebar
st.sidebar.title(":red[Customer Segmentation Visualization]")
genre = st.sidebar.selectbox("Select a Gender", df['Gender'].unique())
age_range = st.sidebar.slider("Select an age range", int(df['Age'].min()), int(df['Age'].max()), (int(df['Age'].min()), int(df['Age'].max())))

# Filter the data
df_filtered = df[(df['Gender'] == genre) & (df['Age'] >= age_range[0]) & (df['Age'] <= age_range[1])]

# Create the scatter plot for Annual Income and Spending Score
scatter_plot = alt.Chart(df_filtered).mark_circle().encode(
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    color='Gender',
    tooltip=['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']
).interactive()

# Create the histogram for Annual Income
histogram_income = alt.Chart(df_filtered).mark_bar().encode(
    x=alt.X('Annual Income (k$):Q', bin=True),
    y='count()',
    color='Gender',
    tooltip=['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']
).interactive()

# Create the histogram for Spending Score
histogram_spending = alt.Chart(df_filtered).mark_bar().encode(
    x=alt.X('Spending Score (1-100):Q', bin=True),
    y='count()',
    color='Gender',
    tooltip=['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']
).interactive()

# Create the KDE plot for Annual Income
sns.set_style('whitegrid')
fig, ax = plt.subplots()
sns.kdeplot(data=df_filtered, x='Annual Income (k$)', hue='Gender', fill=True, common_norm=False, alpha=.5, ax=ax)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Density')
plt.title('Density Plot for Annual Income')
kde_plot_income = st.pyplot(fig)

# Create the KDE plot for Spending Score
sns.set_style('whitegrid')
fig, ax = plt.subplots()
sns.kdeplot(data=df_filtered, x='Spending Score (1-100)', hue='Gender', fill=True, common_norm=False, alpha=.5, ax=ax)
plt.xlabel('Spending Score (1-100)')
plt.ylabel('Density')
plt.title('Density Plot for Spending Score')
kde_plot_spending = st.pyplot(fig)

# Display the visualizations
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.write(scatter_plot)
with col2:
    st.write(histogram_income)
with col3:
    st.write(histogram_spending)
#with col4:
    #st.write(kde_plot_income)
    # st.write(kde_plot_spending)
    # col2.write(kde_plot_income._repr_html_() + kde_plot_score._repr_html_())






# import streamlit as st
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt


# df = pd.read_csv("Mall_Customers.csv")

# # Set page title
# st.set_page_config(page_title="Data Analysis App")

# # Sidebar
# analysis_type = st.sidebar.selectbox("Select Analysis Type", ["Univariate Analysis", "Bivariate Analysis", "Multivariate Analysis"])

# # Display appropriate analysis
# if analysis_type == "Univariate Analysis":
#     st.write(df.describe())
#     st.write(df.hist())

# elif analysis_type == "Bivariate Analysis":
#     # Scatterplot
#     fig = sns.scatterplot(x="Annual Income (k$)", y="Spending Score (1-100)", hue="Gender", data=df)
#     st.pyplot(fig)

#     # Save the plot to disk
#     fig.savefig("scatterplot.png")

# elif analysis_type == "Multivariate Analysis":
#     # Pairplot
#     fig = sns.pairplot(df, hue="Gender")
#     st.pyplot(fig)

#     # Save the plot to disk
#     fig.savefig("pairplot.png")