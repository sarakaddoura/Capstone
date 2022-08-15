import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score,mean_squared_error,r2_score
from sklearn.preprocessing import MinMaxScaler
import datetime
from sklearn.preprocessing import OneHotEncoder
from streamlit_option_menu import option_menu
from sklearn.metrics.pairwise import cosine_similarity
import hydralit_components as hc
import plotly.express as px

### Importing Dataset

df = pd.read_csv('https://raw.githubusercontent.com/sarakaddoura/Capstone/main/Final_Data_4.csv')


### Page layout

st.set_page_config(
    page_title="Capstone Project",
    layout="wide",
)


### Option Menu

selected = option_menu(None, ["Home","Overview","Price Prediction","Recommendation System"],
    icons=['house',"eye","bar-chart", 'laptop'],
    menu_icon="cast", default_index=0, orientation="horizontal")

### Home page

if selected == "Home":
    st.markdown(f'<h1 style="color:red;font-size:35px;">{"E-commerce Clothing Store"}</h1>', unsafe_allow_html=True)

    col1,col2 = st.columns(2)

    col1.text_area('Capstone Project','''
The Capstone Project constitutes of 3 main parts:
- Analysis of Data using PowerBI
- Price Prediction model
- Recommendation System
    ''', height = 170)

    image = 'https://pixelplex.io/wp-content/uploads/2020/10/augmented-and-virtual-realities-what-is-the-difference.jpg'
    col2.image(image, use_column_width= True)

### Overview page

if selected == "Overview":

    ### Values for Cards

    value1 = df['Order Number'].nunique()
    value2 = "{:,}".format(value1)
    value3 = sum(df['Lineitem quantity'])
    value4 = "{:,}".format(value3)
    value5 = df['Total'].sum()
    value6 = "{:,.0f} $".format(value5)

    ### Columns

    info = st.columns(3)
    col1 = st.columns(1)

    ### Cards Themes

    theme_bad = {'bgcolor': '#f9f9f9','title_color': '#6E9BD0','content_color': '#EE7E71','icon_color': '#EE7E71', 'icon': 'fa fa-inbox'}
    theme_neutral = {'bgcolor': '#f9f9f9','title_color': '#6E9BD0','content_color': '#EE7E71','icon_color': '#EE7E71','icon': 'fa fa-hashtag'}
    theme_good = {'bgcolor': '#f9f9f9','title_color': '#6E9BD0','content_color': '#EE7E71','icon_color': '#EE7E71', 'icon': 'fa fa-credit-card'}

    ###datetype

    df['Date of Order'] = pd.to_datetime(df['Date of Order'], format='%d/%m/%Y')

    ###Figure 1

    fig1 = px.line(df, x="Date of Order", y="Total", title='Peaks of Sales')
    fig1.update_yaxes(title='y', visible=False, showticklabels=False)
    fig1.update_xaxes(title='', visible=True, showticklabels=True)
    st.plotly_chart(fig1,use_container_width = True)



    ### Cards
    with info[0]:
        hc.info_card(title= 'Unique Orders', content = value2, theme_override = theme_bad)

    with info[1]:
        hc.info_card(title = 'Total Quantity',content= value4 , theme_override = theme_neutral)

    with info[2]:
        hc.info_card(title = 'Amount Spent',content= value6 , sentiment='good', theme_override = theme_good)



### Price Prediction page

if selected == "Price Prediction":

    ### X and Y

    y = df['Lineitem price']
    x = df[["Shipping Country","Source","Category","Shipping Method"]]



    ### Columns

    col1,col2 = st.columns(2)

    with col1:
        st.write("")
        st.write("")
        st.markdown(f'<h1 style="color:red;font-size:28px;">{"Kindly enter the Following information:"}</h1>', unsafe_allow_html=True)
        st.write("")
        st.write("")
        st.write("")

    ### Manual encoding
        shipping_country = st.selectbox("Shipping Country",["AE","LB","US","FR","GB","QA","ES","CA"
            ,"KW","AZ","TR","BH","SA","GR","LV","DE","AU","LS","AL"])

        source = st.selectbox("Source",["Web","Shopify_draft_order","Iphone","Pos","Android"])
        category = st.selectbox("Category",["Top","Dress","Cardigan","Scarf","Pants","Suit","Skirt"
            ,"Jacket","Ensemble","Waistcoat","Abaya","Shirtdress","Sweat Pants","Cape","Sweater","Set"
            ,"Vest","Sweatsuit","Sweatdress","Kaftan","Gown","Coat","Kimono","Trenchcoat","Kafcket","Hoodie"
            ,"Beanie","Jumpsuit","Culottes","Bodysuit","Sweatshirt","Belt","Shorts","Sweat Coord","Socks"
            ,"Jeans","Leggings","Rope","Scrunchie","Box"])


        shipping_method = st.selectbox("shipping method",[
            "Local shipping","UPS","Free shipping","Aramex","Wakilni","Top speed",
            "Custom","DHL","Global shipping"])


        source_dict = {"Web":1,"Shopify_draft_order":2,"Iphone":3,"Pos":4,"Android":5}
        shipping_method_dict = {"Local shipping":1,"UPS":2,"Free shipping":3,"Aramex":4,"Wakilni":5,"Top speed":6,
            "Custom":7,"DHL":8,"Global shipping":9}
        shipping_country_dict = {"AE":1,"LB":2,"US":3,"FR":4,"GB":5,"QA":6,"ES":7,"CA":8
            ,"KW":9,"AZ":10,"TR":11,"BH":12,"SA":13,"GR":14,"LV":15,"DE":16,"AU":17,"LS":18,"AL":19}

        category_dict = {"Top":1,"Dress":2,"Cardigan":3,"Scarf":4,"Pants":5,"Suit":6,"Skirt":7
            ,"Jacket":8,"Ensemble":9,"Waistcoat":10,"Abaya":11,"Shirtdress":12,"Sweat Pants":13,"Cape":14,"Sweater":15,"Set":16
            ,"Vest":17,"Sweatsuit":18,"Sweatdress":19,"Kaftan":20,"Gown":21,"Coat":22,"Kimono":23,"Trenchcoat":24,"Kafcket":25,"Hoodie":26
            ,"Beanie":27,"Jumpsuit":28,"Culottes":29,"Bodysuit":30,"Sweatshirt":31,"Belt":32,"Shorts":33,"Sweat Coord":34,"Socks":35
            ,"Jeans":36,"Leggings":37,"Rope":38,"Scrunchie":39,"Box":40}


        df['Source'] = df['Source'].map(source_dict)
        df['Shipping Country']= df['Shipping Country'].map(shipping_country_dict)
        df['Category']=df['Category'].map(category_dict)
        df['Shipping Method'] = df['Shipping Method'].map(shipping_method_dict)

        ### X and Y

        Xfeatures = df[["Shipping Country","Source","Category","Shipping Method"]]
        y = df['Lineitem price']

        ### Splitting

        X_train, X_val, y_train, y_val= train_test_split(Xfeatures, y, test_size=0.2, random_state=42)
        dtree = DecisionTreeRegressor()
        dtree.fit(X_train,y_train)

        ### Predicting

        dt= dtree.predict(X_train)

        ### linking the categorical variables with encoded dictionaries

        def get_value(val,my_dict):
            for key,value in my_dict.items():
                if val == key:
                    return value

        source = get_value(source,source_dict)
        category = get_value(category,category_dict)
        shipping_method = get_value(shipping_method,shipping_method_dict)
        shipping_country = get_value(shipping_country,shipping_country_dict)
        single_sample = [shipping_country,source,category,shipping_method]

        ###Prediction button

        if st.button("Predict"):
            sample = np.array(single_sample).reshape(1,-1)
            prediction = dtree.predict(sample)
            st.info("Predicted Purchase")
            st.header("${}".format(prediction[0]))

    with col2:
        image = 'https://biznext.in/images/services/Money-Transfer-1.png'
        col2.image(image, use_column_width= True)


### Recommendation system page

if selected == "Recommendation System":
    st.markdown(f'<h1 style="color:red;font-size:25px;">{"Kindly enter your Order Number"}</h1>', unsafe_allow_html=True)

    ### Subset of dataset

    df_baskets = df[['Order Number', 'Lineitem name', 'Lineitem quantity']]
    #st.write(df_baskets.head())

    ### Pivot table

    customer_item_matrix = df_baskets.pivot_table(index='Order Number',
    columns=['Order Number'], values='Lineitem quantity').fillna(0)
    #st.write(customer_item_matrix.head())

### Cosine similarity

    user_user_sim_matrix = pd.DataFrame(cosine_similarity(customer_item_matrix))
    #st.write(user_user_sim_matrix.head())


    #Renaming index and column names

    user_user_sim_matrix.columns = customer_item_matrix.index

    user_user_sim_matrix['Order Number'] = customer_item_matrix.index
    user_user_sim_matrix = user_user_sim_matrix.set_index('Order Number')

    item_item_sim_matrix = pd.DataFrame(cosine_similarity(customer_item_matrix.T))
    item_item_sim_matrix.columns = customer_item_matrix.T.index

    item_item_sim_matrix['Order Number'] = customer_item_matrix.T.index
    item_item_sim_matrix = item_item_sim_matrix.set_index('Order Number')

    ### Text Input of Order Number

    var3= st.text_input(label="Order Number",value = 'R-S-5001')

    col1,col2 = st.columns(2)

    with col1:
        top_10_similar_items = list(item_item_sim_matrix.loc[var3].sort_values(ascending=False).iloc[:10].index)

        st.write(top_10_similar_items)

    with col2:
        st.write("Recommended Items")
        df.loc[
        df['Order Number'].isin(top_10_similar_items),
        ['Order Number', 'Lineitem name']
        ].drop_duplicates().set_index('Order Number').loc[top_10_similar_items]

    expander = st.expander("Key Insights")
    expander.write("""
Above you can see a list of suggested items based on the customers similarity of previous purchase :)
    """)
