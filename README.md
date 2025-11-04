# Australian EV Charging Monitor
*Visualising access, equity, and reach across Australia’s EV charging network.*

An interactive geospatial national EV charging monitor for Australia’s electric vehicle infrastructure, integrating live data from the [Open Charge Map API](https://openchargemap.org/), population centres from the ABS [Urban Centres and Localities (UCL)](https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/significant-urban-areas-urban-centres-and-localities-section-state/urban-centres-and-localities), and open-source routing via [OSRM (Open Source Routing Machine)](https://project-osrm.org/), powered by [OpenStreetMap](https://www.openstreetmap.org/) data. Features include dynamic coverage metrics, route-based charger proximity analysis, and population equity summaries, updated daily to reveal where Australia’s charging network is growing - and where gaps remain.


---

## ⚙️ Setup & Hosting

### 1 · Repository setup
Clone or fork this repository. It should contain:

- `ev_charging_monitor.py` (main Python build script)  
- `requirements.txt`  
- `.github/workflows/rebuild.yml`  
- `index.html`

---

### 2 · Automatic rebuild
GitHub Actions (see `.github/workflows/rebuild.yml`) runs the Python script every 24 hours, regenerates the map, and commits `index.html`.

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
- Coverage and equity metrics are derived from live charger counts within 5 km, 10 km, and 20 km of each population centre.  
- Data pull times and network metrics are shown in the "Australian EV charging snapshot" panel.
