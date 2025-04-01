# 🌍 Country Information Cards - Streamlit Web App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://information-country-code-project-dlgivfv.streamlit.app/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repo-blue?logo=github)](https://github.com/Hafsa-Kamali/Information-country-code-project)

A beautiful, interactive web application that displays comprehensive information about countries worldwide with amazing visualizations and comparison tools.

## Features ✨

### 🗺️ Comprehensive Country Data
- Detailed country cards with flags and essential information
- Official names, capitals, regions, and subregions
- Population statistics and geographical area
- Languages, currencies, and timezones
- UN membership status and landlocked information

### 🔍 Powerful Search & Filtering
- Search by country name (common or official)
- Filter by world regions (Africa, Americas, Asia, etc.)
- Population range slider
- Area (km²) range slider
- Multiple sorting options (name, population, area)

### 📊 Comparison Tools
- Select multiple countries for side-by-side comparison
- Interactive bar charts for population and area
- Scatter plot visualization of population vs area
- Detailed comparison table with key metrics

### 🖼️ Visual Elements
- Country flags in high resolution
- Flag emojis for quick recognition
- Interactive maps showing country locations
- Beautifully styled cards with consistent design

### ⚙️ Technical Features
- Responsive design works on all devices
- Data caching for fast performance
- Pagination for browsing many countries
- Custom CSS styling for enhanced UI
- REST Countries API integration

## Live Demos 🚀

Try the app now:
- [Streamlit Cloud Deployment](https://information-country-code-project-dlgivfv.streamlit.app/)
- [GitHub Repository](https://github.com/Hafsa-Kamali/Information-country-code-project)

## Screenshots 📸

![Main Interface](screenshots/main.png)
*Main application interface with country cards*

![Comparison View](screenshots/comparison.png)
*Country comparison with interactive charts*

## Installation 💻

1. Clone the repository:
```bash
git clone https://github.com/yourusername/country-info-cards.git
cd country-info-cards
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run country_app.py
```

## Requirements 📦

- Python 3.7+
- Streamlit
- Pandas
- Requests
- Plotly
- Pillow
- PyCountry

## Configuration ⚙️

The app requires no special configuration. All data is fetched from the public [REST Countries API](https://restcountries.com/).

## Contributing 🤝

Contributions are welcome! Please open an issue or submit a pull request for any improvements.

## License 📄

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments 🙏

- Data provided by [REST Countries API](https://restcountries.com/)
- Built with [Streamlit](https://streamlit.io/)
- Flag images from country flags API
