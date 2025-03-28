import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pinecone import Pinecone
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# Access API keys from Streamlit's cloud secrets
genai_api_key = st.secrets["GENAI_API_KEY"]
pinecone_api_key = st.secrets["PINECONE_API_KEY"]

# Configure Generative AI and Pinecone with secure keys
genai.configure(api_key=genai_api_key)
pc = Pinecone(api_key=pinecone_api_key)

index_name = "bookings" 

index = pc.Index(index_name)


embedder = SentenceTransformer('all-mpnet-base-v2')


file_path = 'cleaned_hotel_bookings.csv'  
hotel_data = pd.read_csv(file_path)

hotel_data['reservation_status_date'] = pd.to_datetime(hotel_data['reservation_status_date'], format='%d-%m-%y')
hotel_data['arrival_date_month'] = hotel_data['arrival_date_month'].str.capitalize()
hotel_data['arrival_date'] = pd.to_datetime(
    hotel_data['arrival_date_year'].astype(str) + '-' +
    hotel_data['arrival_date_month'] + '-' +
    hotel_data['arrival_date_day_of_month'].astype(str)
)
hotel_data['total_nights'] = hotel_data['stays_in_weekend_nights'] + hotel_data['stays_in_week_nights']
hotel_data['total_revenue'] = hotel_data['adr'] * hotel_data['total_nights']

def retrieve_relevant_chunks(query, top_k=5):
    query_embedding = embedder.encode([query])
    results = index.query(vector=query_embedding.tolist(), top_k=top_k, include_metadata=True)
    relevant_chunks = [match['metadata']['text'] for match in results['matches']]
    return " ".join(relevant_chunks)

def question_text(retrieved_text, question):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content([f"Answer the following question based on the provided text:\n\nText: {retrieved_text}\n\nQuestion: {question}"])
        return response.text
    except Exception as e:
        return f"An error occurred: {e}"

st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a Page", ["Analytics Dashboard", "QA Bot"])

if page == "Analytics Dashboard":
    st.title("📊 Hotel Booking Analytics Dashboard")
    
    st.subheader("1. Monthly Revenue Trends")
    revenue_by_month = hotel_data.groupby(['arrival_date_year', 'arrival_date_month'])['total_revenue'].sum().unstack()
    fig, ax = plt.subplots(figsize=(14, 7))
    revenue_by_month.T.plot(kind='line', marker='o', ax=ax)
    ax.set_title('Monthly Revenue Trends')
    ax.set_xlabel('Month')
    ax.set_ylabel('Total Revenue')
    ax.grid(True)
    ax.legend(title='Year')
    st.pyplot(fig)
    
    st.subheader("2. Cancellation Rate")
    total_bookings = len(hotel_data)
    cancelled_bookings = hotel_data['is_canceled'].sum()
    cancellation_rate = (cancelled_bookings / total_bookings) * 100
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = ['Not Cancelled', 'Cancelled']
    sizes = [100 - cancellation_rate, cancellation_rate]
    colors = ['#66b3ff', '#ff9999']
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    ax.set_title(f'Cancellation Rate: {cancellation_rate:.2f}% of Total Bookings')
    st.pyplot(fig)
    
    st.subheader("3. Top 15 Countries by Number of Bookings")
    top_countries = hotel_data['country'].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.barplot(x=top_countries.values, y=top_countries.index, palette='viridis', ax=ax)
    ax.set_title('Top 15 Countries by Number of Bookings')
    ax.set_xlabel('Number of Bookings')
    ax.set_ylabel('Country')
    st.pyplot(fig)
    
    st.subheader("4. Booking Lead Time Distribution")
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.histplot(hotel_data['lead_time'], bins=50, kde=True, ax=ax)
    ax.set_title('Distribution of Booking Lead Time (Days)')
    ax.set_xlabel('Lead Time (Days)')
    ax.set_ylabel('Number of Bookings')
    ax.grid(True)
    st.pyplot(fig)
    
    st.subheader("5. Market Segment Distribution")
    market_segment_counts = hotel_data['market_segment'].value_counts()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=market_segment_counts.values, y=market_segment_counts.index, palette='rocket', ax=ax)
    ax.set_title('Distribution by Market Segment')
    ax.set_xlabel('Number of Bookings')
    ax.set_ylabel('Market Segment')
    st.pyplot(fig)
    
    st.subheader("6. Average Daily Rate by Hotel Type")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='hotel', y='adr', data=hotel_data, palette='Set2', ax=ax)
    ax.set_title('Average Daily Rate by Hotel Type')
    ax.set_xlabel('Hotel Type')
    ax.set_ylabel('Average Daily Rate (ADR)')
    st.pyplot(fig)
    
    st.subheader("7. Booking Changes Analysis")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='booking_changes', data=hotel_data[hotel_data['booking_changes'] < 5], palette='coolwarm', ax=ax)
    ax.set_title('Distribution of Booking Changes')
    ax.set_xlabel('Number of Booking Changes')
    ax.set_ylabel('Count')
    st.pyplot(fig)
    
    st.subheader("8. Special Requests Analysis")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='total_of_special_requests', data=hotel_data, palette='magma', ax=ax)
    ax.set_title('Distribution of Special Requests')
    ax.set_xlabel('Number of Special Requests')
    ax.set_ylabel('Count')
    st.pyplot(fig)
    
    st.subheader("9. Cancellation Rate by Deposit Type")
    cancellation_by_deposit = hotel_data.groupby('deposit_type')['is_canceled'].mean() * 100
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=cancellation_by_deposit.index, y=cancellation_by_deposit.values, palette='viridis', ax=ax)
    ax.set_title('Cancellation Rate by Deposit Type')
    ax.set_xlabel('Deposit Type')
    ax.set_ylabel('Cancellation Rate (%)')
    st.pyplot(fig)
    
    st.subheader("10. Repeated Guests Analysis")
    repeated_guests = hotel_data['is_repeated_guest'].value_counts(normalize=True) * 100
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=repeated_guests.index, y=repeated_guests.values, palette='Set1', ax=ax)
    ax.set_title('Percentage of Repeated Guests')
    ax.set_xlabel('Is Repeated Guest')
    ax.set_ylabel('Percentage')
    ax.set_xticklabels(['No', 'Yes'])
    st.pyplot(fig)

elif page == "QA Bot":
    st.title("🤖 RAG-based QA Bot")
    
    question = st.text_input("Enter your question about the document")
    if st.button("Get Answer"):
        if question:
            relevant_text = retrieve_relevant_chunks(question)
            answer = question_text(relevant_text, question)
            st.subheader("Answer")
            st.write(answer)
        else:
            st.warning("Please enter a question to get an answer.")
