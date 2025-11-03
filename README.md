# Australian EV Charging Monitor

A live map of electric vehicle chargers across Australia, updated automatically every 24 hours.  
Built using open data from the [Open Charge Map API](https://openchargemap.org/) and [OpenStreetMap OSRM](https://project-osrm.org/).

---

## ⚙️ Setup & Hosting

### 1 · Repository setup
Clone or fork this repository. It should contain:

- `build_ev_monitor_with_urban_centres_real.py` (main Python build script)  
- `requirements.txt`  
- `.github/workflows/rebuild.yml`  
- `data/processed/ocm_australia_latest.csv`  
- `outputs/index.html`

---

### 2 · Automatic rebuild
GitHub Actions (see `.github/workflows/rebuild.yml`) runs the Python script every 24 hours, regenerates the map, and commits `outputs/index.html`.

You can also trigger it manually in the Actions tab:  
**Actions → Rebuild and Deploy EV Charging Monitor → Run workflow**

---

### 3 · Hosting on GitHub Pages
The live map is hosted directly via GitHub Pages using the Actions deploy pipeline.  
After a successful workflow run, the public site is available at:  
👉 [https://hdia.github.io/ev_charging_monitor/](https://hdia.github.io/ev_charging_monitor/)

---

### 4 · Data sources & notes
- **Charger listings:** Open Charge Map API (Australia subset, refreshed daily)  
- **Urban centres:** ABS *Urban Centres and Localities 2021*  
- **Routing and proximity:** OpenStreetMap *Nominatim* + *OSRM*  
- Coverage metrics and equity tables derived from live charger counts within 5 km, 10 km, and 20 km of each population centre.  
- Snapshot time is shown in the “How to read this map” panel.
