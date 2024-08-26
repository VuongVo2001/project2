import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from wordcloud import WordCloud, STOPWORDS

# function cần thiết
def get_recommendations(df, hotel_id, cosine_sim, nums=8):
    # Get the index of the hotel that matches the hotel_id
    matching_indices = df.index[df['Hotel_ID'] == hotel_id].tolist()
    if not matching_indices:
        print(f"No hotel found with ID: {hotel_id}")
        return pd.DataFrame()  # Return an empty DataFrame if no match
    idx = matching_indices[0]

    # Get the pairwise similarity scores of all hotels with that hotel
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the hotels based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the nums most similar hotels (Ignoring the hotel itself)
    sim_scores = sim_scores[1:nums+1]

    # Get the hotel indices
    hotel_indices = [i[0] for i in sim_scores]

    # Return the top n most similar hotels as a DataFrame
    return df.iloc[hotel_indices]

def get_surprise_recommendations(df, hotel_id, model, nums=8):
    # Get the hotel title or ID from the DataFrame
    matching_indices = df.index[df['Hotel_ID'] == hotel_id].tolist()
    if not matching_indices:
        print(f"No hotel found with ID: {hotel_id}")
        return pd.DataFrame()  # Return an empty DataFrame if no match
    selected_hotel_title = df.loc[matching_indices[0], 'Title']

    # Create a list to store predictions
    predictions = []

    # Iterate over all other titles in the DataFrame to predict ratings
    for other_title in df['Title'].unique():
        if other_title != selected_hotel_title:
            pred = model.predict(selected_hotel_title, other_title).est
            predictions.append((other_title, pred))

    # Sort predictions by estimated rating
    predictions.sort(key=lambda x: x[1], reverse=True)

    # Get the top N recommended titles
    top_recommendations = [x[0] for x in predictions[:nums]]

    # Filter the DataFrame to include only the recommended hotels
    recommended_hotels = df[df['Title'].isin(top_recommendations)]

    return recommended_hotels

def display_recommended_hotels(recommended_hotels):
    if not recommended_hotels.empty:
        # Select the columns you want to display
        table_data = recommended_hotels[['Hotel_Name', 'Hotel_Description']].copy()

        # Optionally truncate the hotel description for the table
        table_data['Hotel_Description'] = table_data['Hotel_Description'].apply(
            lambda x: ' '.join(x.split()[:100]) + '...' if len(x.split()) > 100 else x
        )

        # Rename the columns for better readability
        table_data = table_data.rename(columns={
            'Hotel_Name': 'Tên khách sạn',
            'Hotel_Description': 'Mô tả'
        })

        # Display the table
        st.table(table_data)
    else:
        st.write("Không có khách sạn gợi ý.")


data_hotel = pd.read_csv('hotel_info_1.csv', encoding='utf-8')
data_comment = pd.read_csv('hotel_comments.csv', encoding='utf-8')
merged_data = pd.merge(data_hotel, data_comment, left_on="Hotel_ID", right_on="Hotel ID", how="inner")
final_data = merged_data.drop(["num_x","Hotel_ID", "Hotel ID", "Score Level","Group Name","Room Type","Review Date","Hotel_Rank"],axis=1)

# Đọc dữ liệu khách sạn
df_hotels = pd.read_csv('hotel_info_1.csv')
# Lấy 15 khách sạn đầu tiên
random_hotels = df_hotels.head(n=15)
# print(random_hotels)

st.session_state.random_hotels = random_hotels

# Open and read file
with open('cosine_model.pkl', 'rb') as f:
    cosine_model_new = pickle.load(f)

with open('cosine_model.pkl', 'rb') as f2:
    surprise_model_new = pickle.load(f2)

###### Giao diện Streamlit ######
st.title("Project 2 - Hotel Recommendation")
st.header("Võ Hùng Vương - Vũ Thúy Cầm")
menu = ["Home Page", "Business Objective", "Data Exploration","Prediction by Content-based Model","Predict by Collaborative Model"]
choice = st.sidebar.selectbox('Menu', menu)

if choice == 'Home Page':
    st.image('hotel.jpg', use_column_width=True)
elif choice == "Business Objective":
    st.subheader('Business Objective for a Machine Learning Project on Hotel Recommendation using Agoda Data')
    st.write("""

            **Objective:**

            The primary objective of this machine learning project is to develop a personalized hotel recommendation system for Agoda's platform. By leveraging customer behavior data, hotel attributes, and user reviews from Agoda, the goal is to enhance the booking experience for users by providing tailored hotel recommendations that align with their preferences and past behaviors.

            **Key Goals:**

            1. **Increase Customer Engagement:**
            - By offering highly relevant hotel suggestions based on individual user profiles, the project aims to increase customer interaction with the platform, leading to longer browsing sessions and higher conversion rates.

            2. **Boost Booking Conversions:**
            - Personalized recommendations are expected to drive an increase in hotel bookings, as users are more likely to find and book hotels that match their specific preferences and needs.

            3. **Enhance Customer Satisfaction and Loyalty:**
            - Delivering accurate and context-aware hotel recommendations will improve overall customer satisfaction, encouraging repeat bookings and fostering long-term loyalty to Agoda.

            4. **Optimize Revenue Streams:**
            - By recommending hotels that are more likely to be booked, the project can optimize the distribution of bookings across various hotel partners, potentially leading to higher commissions and better inventory management.

            5. **Data-Driven Insights for Marketing and Operations:**
            - The recommendation system will also provide valuable insights into customer preferences and trends, enabling Agoda's marketing and operations teams to tailor their strategies more effectively.
            
            **Implement:**

            1. **Content-based Model: Gensim  + Cosine similarity**
            
            2. **Collaborative Model: ALS  + Surprise**
            
            This project aligns with Agoda's strategic goal of leveraging data to provide a superior customer experience, driving both short-term revenue growth and long-term customer loyalty.
            """)
    st.image('content_vs_collab.png', use_column_width=True)


elif choice == 'Data Exploration':    
    st.subheader("Data Exploration")
    # Group by Hotel_Name and Hotel_Address and count the occurrences of a relevant column, e.g., 'Comment_ID'
    hotel_comment_counts = final_data.groupby(["Hotel_Name", "Hotel_Address"]).size().reset_index(name='count')

    # Now you can proceed to get the top 20 hotels
    top_20_hotels = hotel_comment_counts.sort_values(by="count", ascending=False).head(20)
    st.write("##### 1. Some data")
    st.dataframe(final_data.head(5))
    # Plotting 2
    st.write("##### 2. Phân bố số lượng comment theo từng khách sạn (Top 20)")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x="count", y="Hotel_Name", data=top_20_hotels, ax=ax)
    plt.title("Phân bố số lượng comment theo từng khách sạn (Top 20)")
    plt.xlabel("Số lượng comment")
    plt.ylabel("Tên khách sạn")
    st.pyplot(fig)
    st.write("The bar chart shows the distribution of the number of comments across the top 20 hotels. The Sata Hotel has received the highest number of comments, followed by Khách sạn Xavia (Xavia Hotel) and Khu nghỉ dưỡng Alma Cam Ranh (Alma Resort Cam Ranh). The chart provides insight into which hotels are the most commented on by customers, indicating potentially higher engagement or popularity.")
    
    # Plotting 3
    st.write("##### 3. Phân bố số lượng đánh giá cho mỗi khách sạn")
    # Grouping and counting the number of reviews per hotel
    hotel_review_counts = final_data.groupby(["Hotel_Name", "Hotel_Address"]).size().reset_index(name='count')
    # Creating a histogram of the review counts
    fig, ax = plt.subplots(figsize=(10, 6))
    hotel_review_counts['count'].hist(bins=50, ax=ax)
    ax.set_title("Phân bố số lượng đánh giá cho mỗi khách sạn")
    ax.set_xlabel("Số lượng đánh giá")
    ax.set_ylabel("Số lượng khách sạn")
    # Displaying the plot in Streamlit
    st.pyplot(fig)
    st.write("This chart shows the distribution of the number of reviews for each hotel. It is evident that most hotels receive very few reviews, with the majority having fewer than 200 reviews. Only a small number of hotels receive a higher number of reviews, and this number decreases as the review count increases. This suggests that the majority of hotels have relatively low engagement from customers, while only a few standout hotels have a high number of reviews.")

    # Plotting 4
    st.write("##### 4. Top 20 khách hàng thường xuyên comment")
    # Grouping and counting the number of comments per user
    user_review_counts = final_data.groupby(["Reviewer Name", "Nationality"]).size().reset_index(name='count')
    # Sorting by the count of comments in descending order
    user_review_counts = user_review_counts.sort_values(by="count", ascending=False)
    # Lấy top 20 người dùng thường xuyên comment
    top_20_users = user_review_counts.head(20)
    # Vẽ biểu đồ
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x="count", y="Reviewer Name", hue="Nationality", data=top_20_users, ax=ax)
    plt.title("Top 20 khách hàng thường xuyên comment")
    plt.xlabel("Số lượng comment")
    plt.ylabel("Tên khách hàng")
    st.pyplot(fig)
    st.write("We can not build the model base on the weighted of each comment from some active reviewer. Because it just use the family name, it is the same!")

    # Plotting 5
    st.write("##### 5. Phân bố của cột Score")
    # Grouping and counting the number of occurrences for each score
    score_counts = final_data.groupby("Score").size().reset_index(name='count')
    # Plotting the distribution of the Score column
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x="Score", y="count", data=score_counts, ax=ax)
    ax.set_title("Phân bố của cột Score")
    ax.set_xlabel("Score")
    ax.set_ylabel("Số lượng")
    # Display the plot in the Streamlit app
    st.pyplot(fig)
    st.write("So we will devide positive as 10, neutral as 9, the remaining is negative. It will reduce the overfitting to our model")

    # Plotting 6
    st.write("##### 6. WordCloud cho các comment")
    # Chuyển đổi cột "Review Body" thành một list các từ, loại bỏ giá trị null
    text = " ".join([str(review) for review in final_data["Body"] if pd.notnull(review)])
    # Tạo wordcloud
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(width=800, height=800,
                        background_color='white',
                        stopwords=stopwords,
                        min_font_size=10).generate(text)
    # Hiển thị wordcloud trong Streamlit
    fig, ax = plt.subplots(figsize=(8, 8), facecolor=None)
    ax.imshow(wordcloud)
    ax.axis("off")
    plt.tight_layout(pad=0)
    st.pyplot(fig)
    st.write("Các từ xuất hiện thường xuyên nhất như nhân viên, nhiệt tình, sạch sẽ,...")


elif choice == 'Prediction by Content-based Model':    
    st.header("Prediction by Content-based Model")

    # Kiểm tra xem 'selected_hotel_id' đã có trong session_state hay chưa
    if 'selected_hotel_id' not in st.session_state:
        # Nếu chưa có, thiết lập giá trị mặc định là None hoặc ID khách sạn đầu tiên
        st.session_state.selected_hotel_id = None

    # Theo cách cho người dùng chọn khách sạn từ dropdown
    # Tạo một tuple cho mỗi khách sạn, trong đó phần tử đầu là tên và phần tử thứ hai là ID
    hotel_options = [(row['Hotel_Name'], row['Hotel_ID']) for index, row in st.session_state.random_hotels.iterrows()]
    st.session_state.random_hotels
    # Tạo một dropdown với options là các tuple này
    selected_hotel = st.selectbox(
        "Chọn khách sạn",
        options=hotel_options,
        format_func=lambda x: x[0]  # Hiển thị tên khách sạn
    )
    # Display the selected hotel
    st.write("Bạn đã chọn:", selected_hotel)

    # Cập nhật session_state dựa trên lựa chọn hiện tại
    st.session_state.selected_hotel_id = selected_hotel[1]

    if st.session_state.selected_hotel_id:
        st.write("Hotel_ID: ", st.session_state.selected_hotel_id)
        # Hiển thị thông tin khách sạn được chọn
        selected_hotel = df_hotels[df_hotels['Hotel_ID'] == st.session_state.selected_hotel_id]

        if not selected_hotel.empty:
            st.write('#### Bạn vừa chọn:')
            st.write('### ', selected_hotel['Hotel_Name'].values[0])

            hotel_description = selected_hotel['Hotel_Description'].values[0]
            truncated_description = ' '.join(hotel_description.split()[:100])
            st.write('##### Thông tin:')
            st.write(truncated_description, '...')

            st.write('##### Các khách sạn khác bạn cũng có thể quan tâm:')
            recommendations = get_recommendations(df_hotels, st.session_state.selected_hotel_id, cosine_sim=cosine_model_new, nums=5) 
            display_recommended_hotels(recommendations)
        else:
            st.write(f"Không tìm thấy khách sạn với ID: {st.session_state.selected_hotel_id}")

elif choice == 'Predict by Collaborative Model':    
    # st.header("Predict by Collaborative Model")

    # # Kiểm tra xem 'selected_hotel_id' đã có trong session_state hay chưa
    # if 'selected_hotel_id' not in st.session_state:
    #     # Nếu chưa có, thiết lập giá trị mặc định là None hoặc ID khách sạn đầu tiên
    #     st.session_state.selected_hotel_id = None

    # # Theo cách cho người dùng chọn khách sạn từ dropdown
    # # Tạo một tuple cho mỗi khách sạn, trong đó phần tử đầu là tên và phần tử thứ hai là ID
    # hotel_options = [(row['Hotel_Name'], row['Hotel_ID']) for index, row in st.session_state.random_hotels.iterrows()]
    # st.session_state.random_hotels
    # # Tạo một dropdown với options là các tuple này
    # selected_hotel = st.selectbox(
    #     "Chọn khách sạn",
    #     options=hotel_options,
    #     format_func=lambda x: x[0]  # Hiển thị tên khách sạn
    # )
    # # Display the selected hotel
    # st.write("Bạn đã chọn:", selected_hotel)

    # # Cập nhật session_state dựa trên lựa chọn hiện tại
    # st.session_state.selected_hotel_id = selected_hotel[1]

    # if st.session_state.selected_hotel_id:
    #     st.write("Hotel_ID: ", st.session_state.selected_hotel_id)
    #     # Hiển thị thông tin khách sạn được chọn
    #     selected_hotel = df_hotels[df_hotels['Hotel_ID'] == st.session_state.selected_hotel_id]

    #     if not selected_hotel.empty:
    #         st.write('#### Bạn vừa chọn:')
    #         st.write('### ', selected_hotel['Hotel_Name'].values[0])

    #         hotel_description = selected_hotel['Hotel_Description'].values[0]
    #         truncated_description = ' '.join(hotel_description.split()[:100])
    #         st.write('##### Thông tin:')
    #         st.write(truncated_description, '...')

    #         st.write('##### Các khách sạn khác bạn cũng có thể quan tâm:')
    #         recommendations = get_surprise_recommendations(df_hotels, st.session_state.selected_hotel_id, surprise_model_new, nums=5) 
    #         display_recommended_hotels(recommendations)
    #     else:
    #         st.write(f"Không tìm thấy khách sạn với ID: {st.session_state.selected_hotel_id}")

    st.header("Predict by Collaborative Model")

    # Kiểm tra xem 'selected_hotel_id' đã có trong session_state hay chưa
    if 'selected_hotel_id' not in st.session_state:
        # Nếu chưa có, thiết lập giá trị mặc định là None hoặc ID khách sạn đầu tiên
        st.session_state.selected_hotel_id = None

    # Giả sử `df_hotels` là DataFrame chứa thông tin khách sạn
    # Tạo một tuple cho mỗi khách sạn, trong đó phần tử đầu là tên và phần tử thứ hai là ID
    hotel_options = [(row['Hotel_Name'], row['Hotel_ID']) for index, row in df_hotels.iterrows()]

    # Tạo một dropdown với options là các tuple này
    selected_hotel = st.selectbox(
        "Chọn khách sạn",
        options=hotel_options,
        format_func=lambda x: x[0]  # Hiển thị tên khách sạn
    )

    # Hiển thị khách sạn đã chọn
    st.write("Bạn đã chọn:", selected_hotel)

    # Cập nhật session_state dựa trên lựa chọn hiện tại
    st.session_state.selected_hotel_id = selected_hotel[1]

    if st.session_state.selected_hotel_id:
        st.write("Hotel_ID: ", st.session_state.selected_hotel_id)
        # Hiển thị thông tin khách sạn được chọn
        selected_hotel_info = df_hotels[df_hotels['Hotel_ID'] == st.session_state.selected_hotel_id]

        if not selected_hotel_info.empty:
            st.write('#### Bạn vừa chọn:')
            st.write('### ', selected_hotel_info['Hotel_Name'].values[0])

            hotel_description = selected_hotel_info['Hotel_Description'].values[0]
            truncated_description = ' '.join(hotel_description.split()[:100])
            st.write('##### Thông tin:')
            st.write(truncated_description, '...')

            st.write('##### Các khách sạn khác bạn cũng có thể quan tâm:')
            recommendations = get_surprise_recommendations(df_hotels, st.session_state.selected_hotel_id, surprise_model_new, nums=5)
            display_recommended_hotels(recommendations)
        else:
            st.write(f"Không tìm thấy khách sạn với ID: {st.session_state.selected_hotel_id}")
