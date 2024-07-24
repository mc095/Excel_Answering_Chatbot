import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    # Load the Excel file
    df = pd.read_excel("student_academics.xlsx")
    return df

def create_chart(df, x, y, title, kind='bar', figsize=(6, 4)):
    plt.figure(figsize=figsize)
    if kind == 'bar':
        sns.barplot(data=df, x=x, y=y)
    elif kind == 'scatter':
        sns.scatterplot(data=df, x=x, y=y)
    elif kind == 'pie':
        plt.pie(df[y], labels=df[x], autopct='%1.1f%%', startangle=90)
    plt.title(title, fontsize=12)
    if kind != 'pie':
        plt.xlabel(x, fontsize=10)
        plt.ylabel(y, fontsize=10)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
    plt.tight_layout()
    return plt

def show_gpa(student):
    st.write(f"**GPA:** {student['GPA']:.2f}")
    fig = create_chart(pd.DataFrame({'Category': ['GPA', 'Max GPA'], 'Value': [student['GPA'], 4.0]}), 
                       'Category', 'Value', "GPA Comparison", kind='pie')
    st.pyplot(fig)

def show_average(student):
    st.write(f"**Average:** {student['Average']:.2f}")
    fig = create_chart(pd.DataFrame({'Category': ['Your Average', 'Remaining'], 'Value': [student['Average'], 100 - student['Average']]}), 
                       'Category', 'Value', "Average Score", kind='pie')
    st.pyplot(fig)

def compare_with_class_average(df, student):
    subjects = ['Programming', 'DataStructures', 'Algorithms', 'Databases', 'ComputerNetworks']
    class_avg = df[subjects].mean()
    comparison_data = pd.DataFrame({
        'Subject': subjects,
        'Student': [student[subject] for subject in subjects],
        'Class Average': class_avg
    })
    comparison_data = comparison_data.melt('Subject', var_name='Category', value_name='Marks')
    fig = create_chart(comparison_data, 'Subject', 'Marks', "Comparison with Class Average")
    plt.legend(fontsize=8)
    st.pyplot(fig)

    # Pie chart for overall comparison
    overall_data = pd.DataFrame({
        'Category': ['Your Average', 'Class Average'],
        'Value': [student[subjects].mean(), class_avg.mean()]
    })
    fig_pie = create_chart(overall_data, 'Category', 'Value', "Overall Comparison", kind='pie')
    st.pyplot(fig_pie)

def compare_performance(student):
    subjects = ['Programming', 'DataStructures', 'Algorithms', 'Databases', 'ComputerNetworks']
    performance_data = pd.DataFrame({
        'Subject': subjects,
        'Marks': [student[subject] for subject in subjects]
    })
    fig = create_chart(performance_data, 'Subject', 'Marks', "Subject-wise Performance")
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

    # Pie chart for subject distribution
    fig_pie = create_chart(performance_data, 'Subject', 'Marks', "Subject Distribution", kind='pie')
    st.pyplot(fig_pie)

def compare_gpa_distribution(df, student):
    fig = create_chart(df, 'GPA', 'StudentID', "GPA Distribution", kind='scatter')
    plt.axhline(y=student.name, color='r', linestyle='--', linewidth=1)
    plt.axvline(x=student['GPA'], color='r', linestyle='--', linewidth=1)
    st.pyplot(fig)

    # Pie chart for GPA categories
    gpa_categories = pd.cut(df['GPA'], bins=[0, 2, 3, 3.5, 4], labels=['Low', 'Average', 'Good', 'Excellent'])
    gpa_distribution = gpa_categories.value_counts().reset_index()
    gpa_distribution.columns = ['Category', 'Count']
    fig_pie = create_chart(gpa_distribution, 'Category', 'Count', "GPA Category Distribution", kind='pie')
    st.pyplot(fig_pie)

def main():
    st.title("Student Statistics Generator")

    df = load_data()

    student_name = st.text_input("Enter your name")
    
    if student_name:
        student = df[df['Name'].str.lower() == student_name.lower()]
        if not student.empty:
            student = student.iloc[0]
            st.write(f"### Statistics for {student['Name']}")

            # Use st.columns with equal widths and remove gaps
            cols = st.columns(5)
            if cols[0].button("What's my GPA?"):
                show_gpa(student)
            if cols[1].button("What's my Average?"):
                show_average(student)
            if cols[2].button("Compare with class average"):
                compare_with_class_average(df, student)
            if cols[3].button("Compare my performance"):
                compare_performance(student)
            if cols[4].button("Compare GPA distribution"):
                compare_gpa_distribution(df, student)
        else:
            st.write("No data found for the given name. Please check the spelling and try again.")

if __name__ == "__main__":
    main()