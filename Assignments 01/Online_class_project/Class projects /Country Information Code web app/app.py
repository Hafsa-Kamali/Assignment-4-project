import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from PIL import Image
import io
import pycountry

# App configuration
st.set_page_config(
    page_title="Country Information Cards",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .card {
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        margin: 10px 0;
        background-color: white;
    }
    .flag-img {
        height: 120px;
        width: auto;
        object-fit: contain;
        margin-bottom: 15px;
        border: 1px solid #eee;
    }
    .country-name {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 10px;
        color: #1e3a8a;
    }
    .info-label {
        font-weight: bold;
        color: #555;
    }
    .stSelectbox div[data-baseweb="select"] > div {
        border-radius: 10px;
        padding: 5px;
    }
    .stButton button {
        border-radius: 10px;
        padding: 8px 16px;
    }
    .stTextInput input {
        border-radius: 10px;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Function to get country data from REST Countries API
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_country_data():
    try:
        response = requests.get("https://restcountries.com/v3.1/all")
        response.raise_for_status()
        data = response.json()
        
        # Process data into a DataFrame
        countries = []
        for country in data:
            try:
                currencies = ", ".join([curr['name'] for curr in country.get('currencies', {}).values()]) if 'currencies' in country else "N/A"
                languages = ", ".join(country.get('languages', {}).values()) if 'languages' in country else "N/A"
                
                countries.append({
                    'name': country.get('name', {}).get('common', 'N/A'),
                    'official_name': country.get('name', {}).get('official', 'N/A'),
                    'cca2': country.get('cca2', 'N/A'),
                    'cca3': country.get('cca3', 'N/A'),
                    'capital': ", ".join(country.get('capital', ['N/A'])),
                    'region': country.get('region', 'N/A'),
                    'subregion': country.get('subregion', 'N/A'),
                    'population': country.get('population', 0),
                    'area': country.get('area', 0),
                    'languages': languages,
                    'currencies': currencies,
                    'timezones': ", ".join(country.get('timezones', ['N/A'])),
                    'flag_url': country.get('flags', {}).get('png', ''),
                    'flag_emoji': country.get('flag', ''),
                    'latitude': country.get('latlng', [0, 0])[0] if 'latlng' in country and len(country['latlng']) > 0 else 0,
                    'longitude': country.get('latlng', [0, 0])[1] if 'latlng' in country and len(country['latlng']) > 1 else 0,
                    'independent': country.get('independent', False),
                    'un_member': country.get('unMember', False),
                    'landlocked': country.get('landlocked', False),
                    'borders': ", ".join(country.get('borders', ['N/A']))
                })
            except Exception as e:
                st.warning(f"Error processing country {country.get('name', {}).get('common', 'Unknown')}: {str(e)}")
                continue
        
        return pd.DataFrame(countries)
    except Exception as e:
        st.error(f"Failed to fetch country data: {str(e)}")
        return pd.DataFrame()

# Function to display country flag
@st.cache_data(ttl=86400)  # Cache for 1 day
def get_flag_image(url):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        return image
    except:
        return None

# Function to display country card
def display_country_card(country):
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if country['flag_url']:
            flag_img = get_flag_image(country['flag_url'])
            if flag_img:
                st.image(flag_img, use_container_width=True, caption=f"Flag of {country['name']}")
        st.markdown(f"<div style='text-align: center; font-size: 48px;'>{country['flag_emoji']}</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"<div class='country-name'>{country['name']}</div>", unsafe_allow_html=True)
        
        cols = st.columns(3)
        with cols[0]:
            st.markdown(f"<div class='info-label'>Official Name</div> {country['official_name']}", unsafe_allow_html=True)
            st.markdown(f"<div class='info-label'>Capital</div> {country['capital']}", unsafe_allow_html=True)
            st.markdown(f"<div class='info-label'>Region</div> {country['region']}", unsafe_allow_html=True)
            if country['subregion'] != 'N/A':
                st.markdown(f"<div class='info-label'>Subregion</div> {country['subregion']}", unsafe_allow_html=True)
        
        with cols[1]:
            st.markdown(f"<div class='info-label'>Population</div> {country['population']:,}", unsafe_allow_html=True)
            st.markdown(f"<div class='info-label'>Area</div> {country['area']:,.2f} km²", unsafe_allow_html=True)
            st.markdown(f"<div class='info-label'>Languages</div> {country['languages']}", unsafe_allow_html=True)
            st.markdown(f"<div class='info-label'>Currencies</div> {country['currencies']}", unsafe_allow_html=True)
        
        with cols[2]:
            st.markdown(f"<div class='info-label'>Country Code</div> {country['cca2']} / {country['cca3']}", unsafe_allow_html=True)
            st.markdown(f"<div class='info-label'>Timezones</div> {country['timezones']}", unsafe_allow_html=True)
            st.markdown(f"<div class='info-label'>UN Member</div> {'Yes' if country['un_member'] else 'No'}", unsafe_allow_html=True)
            st.markdown(f"<div class='info-label'>Landlocked</div> {'Yes' if country['landlocked'] else 'No'}", unsafe_allow_html=True)
        
        # Show map
        if country['latitude'] != 0 and country['longitude'] != 0:
            map_df = pd.DataFrame({
                'lat': [country['latitude']],
                'lon': [country['longitude']],
                'name': [country['name']]
            })
            st.map(map_df, zoom=3, use_container_width=True)

# Function to display country comparison
def display_comparison(selected_countries, df):
    if len(selected_countries) < 2:
        st.warning("Select at least 2 countries to compare")
        return
    
    compare_df = df[df['name'].isin(selected_countries)].copy()
    
    st.subheader("Country Comparison")
    
    # Population comparison
    fig_pop = px.bar(
        compare_df.sort_values('population', ascending=False),
        x='name',
        y='population',
        title='Population Comparison',
        labels={'name': 'Country', 'population': 'Population'},
        color='name',
        text='population'
    )
    fig_pop.update_traces(texttemplate='%{text:,}', textposition='outside')
    st.plotly_chart(fig_pop, use_container_width=True)
    
    # Area comparison
    fig_area = px.bar(
        compare_df.sort_values('area', ascending=False),
        x='name',
        y='area',
        title='Area Comparison (km²)',
        labels={'name': 'Country', 'area': 'Area (km²)'},
        color='name',
        text='area'
    )
    fig_area.update_traces(texttemplate='%{text:,.2f}', textposition='outside')
    st.plotly_chart(fig_area, use_container_width=True)
    
    # Scatter plot of population vs area
    fig_scatter = px.scatter(
        compare_df,
        x='area',
        y='population',
        size='population',
        color='name',
        hover_name='name',
        title='Population vs Area',
        labels={'area': 'Area (km²)', 'population': 'Population'},
        log_x=True,
        log_y=True,
        size_max=60
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Detailed comparison table
    st.subheader("Detailed Comparison")
    compare_display = compare_df[['name', 'capital', 'region', 'population', 'area', 'languages', 'currencies']].copy()
    compare_display['population'] = compare_display['population'].apply(lambda x: f"{x:,}")
    compare_display['area'] = compare_display['area'].apply(lambda x: f"{x:,.2f} km²")
    st.dataframe(compare_display.set_index('name').T, use_container_width=True)

# Main app
def main():
    st.title("🌍 Country Information Cards")
    st.markdown("Explore detailed information about countries around the world.")
    
    # Load data
    df = get_country_data()
    if df.empty:
        st.error("No country data available. Please try again later.")
        return
    
    # Sidebar filters
    with st.sidebar:
        st.header("Filters")
        
        # Search by name
        search_query = st.text_input("Search by country name", "")
        
        # Filter by region
        regions = ['All'] + sorted(df['region'].unique().tolist())
        selected_region = st.selectbox("Filter by region", regions)
        
        # Filter by population
        min_pop, max_pop = int(df['population'].min()), int(df['population'].max())
        population_range = st.slider(
            "Population range",
            min_pop, max_pop,
            (min_pop, max_pop),
            format="%d"
        )
        
        # Filter by area
        min_area, max_area = int(df['area'].min()), int(df['area'].max())
        area_range = st.slider(
            "Area range (km²)",
            min_area, max_area,
            (min_area, max_area),
            format="%d"
        )
        
        # Sort options
        sort_options = {
            "Name (A-Z)": "name",
            "Name (Z-A)": "name_desc",
            "Population (High-Low)": "population_desc",
            "Population (Low-High)": "population_asc",
            "Area (High-Low)": "area_desc",
            "Area (Low-High)": "area_asc"
        }
        sort_by = st.selectbox("Sort by", list(sort_options.keys()))
        
        # Comparison section
        st.header("Country Comparison")
        all_countries = sorted(df['name'].unique().tolist())
        selected_for_comparison = st.multiselect("Select countries to compare", all_countries)
        
        if st.button("Compare Selected Countries"):
            st.session_state['show_comparison'] = True
            st.session_state['selected_for_comparison'] = selected_for_comparison
    
    # Apply filters
    filtered_df = df.copy()
    
    if search_query:
        filtered_df = filtered_df[filtered_df['name'].str.contains(search_query, case=False) | 
                                filtered_df['official_name'].str.contains(search_query, case=False)]
    
    if selected_region != 'All':
        filtered_df = filtered_df[filtered_df['region'] == selected_region]
    
    filtered_df = filtered_df[
        (filtered_df['population'] >= population_range[0]) & 
        (filtered_df['population'] <= population_range[1])
    ]
    
    filtered_df = filtered_df[
        (filtered_df['area'] >= area_range[0]) & 
        (filtered_df['area'] <= area_range[1])
    ]
    
    # Apply sorting
    sort_column = sort_options[sort_by]
    if sort_column == "name":
        filtered_df = filtered_df.sort_values('name')
    elif sort_column == "name_desc":
        filtered_df = filtered_df.sort_values('name', ascending=False)
    elif sort_column == "population_desc":
        filtered_df = filtered_df.sort_values('population', ascending=False)
    elif sort_column == "population_asc":
        filtered_df = filtered_df.sort_values('population')
    elif sort_column == "area_desc":
        filtered_df = filtered_df.sort_values('area', ascending=False)
    elif sort_column == "area_asc":
        filtered_df = filtered_df.sort_values('area')
    
    # Display comparison if requested
    if st.session_state.get('show_comparison', False) and st.session_state.get('selected_for_comparison'):
        display_comparison(st.session_state['selected_for_comparison'], df)
        if st.button("Back to Country List"):
            st.session_state['show_comparison'] = False
        st.markdown("---")
    
    # Display filtered countries
    st.subheader(f"Showing {len(filtered_df)} countries")
    
    # Pagination
    items_per_page = 5
    total_pages = (len(filtered_df) // items_per_page) + (1 if len(filtered_df) % items_per_page != 0 else 0)
    
    if total_pages > 1:
        page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
        start_idx = (page - 1) * items_per_page
        end_idx = start_idx + items_per_page
        paginated_df = filtered_df.iloc[start_idx:end_idx]
    else:
        paginated_df = filtered_df
    
    # Display country cards
    for _, country in paginated_df.iterrows():
        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            display_country_card(country)
            st.markdown("</div>", unsafe_allow_html=True)
    
    # Display total pages if pagination exists
    if total_pages > 1:
        st.write(f"Page {page} of {total_pages}")

if __name__ == "__main__":
    main()