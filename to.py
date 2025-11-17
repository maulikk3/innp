# ==========================================================================================
# FINAL STREAMLIT APP
# Features Included:
# 1) State-based postal filtering for ML performance
# 2) ML-based district prediction using pincode
# 3) Exact postal record lookup with SO → BO priority sort
# 4) City / Village / Locality search
# 5) Full India RTO Mapping (140+ codes)
# 6) Multi-RTO dropdown selection for user confirmation
# ==========================================================================================

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# -------------------------------- STREAMLIT CONFIG --------------------------------
st.set_page_config(page_title="India Pincode & RTO Intelligence", layout="wide")
st.title("📍 India Location & Vehicle Registration Intelligence System")

# -------------------------------- DATA INPUT --------------------------------------
st.sidebar.header("📂 Dataset Input")

choice = st.sidebar.radio("Select Input Method:", ["Use File Path", "Upload CSV"])

if choice == "Use File Path":
    file_path = r"5c2f62fe-5afa-4119-a499-fec9d604d5bd.csv"  # <-- CHANGE IF NEEDED
    df = pd.read_csv(file_path)
else:
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
    else:
        st.warning("Upload CSV first to continue.")
        st.stop()

# -------------------------------- DATA CLEANING ------------------------------------
df.columns = [c.strip().lower() for c in df.columns]

required = ["pincode", "district", "statename", "officename", "officetype"]
for col in required:
    if col not in df.columns:
        st.error(f"❌ Missing required column: {col}")
        st.stop()

df["pincode"] = pd.to_numeric(df["pincode"], errors="coerce")
df["district"] = df["district"].astype(str).str.strip().str.upper()
df["statename"] = df["statename"].astype(str).str.strip().str.upper()
df["officename"] = df["officename"].astype(str).str.strip().str.upper()
df["officetype"] = df["officetype"].astype(str).str.strip().str.upper()

df.dropna(subset=["pincode", "district", "statename"], inplace=True)

# -------------------------------- STATE FILTER -------------------------------------
st.subheader("🗂 State Filter For Faster & Accurate ML Prediction")

state = st.selectbox("Select State:", sorted(df["statename"].unique()))

state_df = df[df["statename"] == state]

st.info(f"📌 Loaded {len(state_df)} records for state: **{state}**")

# -------------------------------- TRAIN ML MODEL -----------------------------------
@st.cache_resource
def train_model(data):
    """Trains ML model using pincode → district prediction logic only within selected state."""

    d = data.copy()
    valid = d["district"].value_counts()
    valid_classes = valid[valid >= 2].index
    d = d[d["district"].isin(valid_classes)]

    if len(valid_classes) < 2:
        return None, 0.0

    if len(d) > 30000:
        d = d.sample(30000, random_state=42)

    X = d[["pincode"]]
    y = d["district"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(n_estimators=150, random_state=42))
    ])

    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))

    return model, accuracy

model, accuracy = train_model(state_df)

# ------------------------------------------------------------------------------------
# FULL INDIA RTO MAPPING (Over 140+ Codes)
# ------------------------------------------------------------------------------------
RTO_MAP = {
    # ==================== ANDHRA PRADESH ====================
    "AP01":"SRIKAKULAM","AP02":"VIZIANAGARAM","AP03":"VISAKHAPATNAM",
    "AP04":"RAJAHMUNDRY","AP05":"AMP","AP06":"NARASPUR",
    "AP07":"BHIMAVARAM","AP08":"ELURU","AP09":"VIJAYAWADA",
    "AP10":"MACHILIPATNAM","AP11":"GUNTUR","AP12":"ONGOLE",
    "AP13":"NELLORE","AP14":"CHITTOOR","AP15":"TIRUPATHI",
    "AP16":"KADAPA","AP17":"ANANTHAPUR","AP18":"KURNOOL",
    "AP19":"HYDERABAD","AP20":"R.R. DISTRICT","AP21":"MEDAK",
    "AP22":"NIZAMABAD","AP23":"ADILABAD","AP24":"KARIMNAGAR",
    "AP25":"WARANGAL","AP26":"KHAMMAM","AP27":"NALGONDA",
    "AP28":"MAHABOOBNAGAR","AP29":"RANGA REDDY","AP30":"SRIKAKULAM",

    # ==================== ARUNACHAL PRADESH ====================
    "AR01":"ITANAGAR","AR02":"NAHARLAGUN","AR03":"TAWANG","AR04":"BOMDILA",
    "AR05":"SEPPA","AR06":"ZIRO","AR07":"DAPORIJO","AR08":"ALONG",
    "AR09":"PASIGHAT","AR10":"TEZU","AR11":"ROING",

    # ==================== ASSAM ====================
    "AS01":"GUWAHATI","AS02":"NAGAON","AS03":"JORHAT","AS04":"SIVSAGAR",
    "AS05":"DIBRUGARH","AS06":"TINSUKIA","AS07":"GOLAGHAT","AS08":"DIMA HASAO",
    "AS09":"KARBI ANGLONG","AS10":"BARPETA","AS11":"KAMRUP","AS12":"DHUBRI",
    "AS13":"KOKRAJHAR","AS14":"UDALGURI","AS15":"MAJULI",

    # ==================== BIHAR ====================
    "BR01":"PATNA","BR02":"GAYA","BR03":"BOJHPUR","BR04":"CHAPRA",
    "BR05":"MOTIHARI","BR06":"MADHUBANI","BR07":"DARBHANGA","BR08":"MADHEPURA",
    "BR09":"SUPAUL","BR10":"BANKA","BR11":"BHAGALPUR","BR12":"MUNGER",
    "BR13":"SAMASTIPUR","BR14":"SIWAN","BR15":"GOPALGANJ","BR16":"SIKAR","BR22":"NALANDA",

    # ==================== CHANDIGARH ====================
    "CH01":"CHANDIGARH",

    # ==================== CHHATTISGARH ====================
    "CG01":"RAIPUR","CG02":"DHAMTARI","CG03":"RAJNANDGAON","CG04":"DURG",
    "CG05":"BILASPUR","CG06":"KORBA","CG07":"RAIGARH","CG08":"JANJGIR",
    "CG09":"JASHPUR","CG10":"BASTAR",

    # ==================== DELHI ====================
    "DL01":"NORTH DELHI","DL02":"SOUTH DELHI","DL03":"EAST DELHI",
    "DL04":"NEW DELHI","DL05":"WEST DELHI","DL06":"CENTRAL DELHI","DL07":"NORTH WEST",

    # ==================== GOA ====================
    "GA01":"PANAJI","GA02":"MARGAO",

    # ==================== GUJARAT ====================
    "GJ01":"AHMEDABAD","GJ02":"MEHSANA","GJ03":"RAJKOT","GJ04":"BHAVNAGAR",
    "GJ05":"SURAT","GJ06":"VADODARA","GJ07":"KHEDA","GJ08":"BANASKANTHA",
    "GJ09":"SABARKANTHA","GJ10":"JAMNAGAR","GJ11":"JUNAGADH","GJ12":"AMRELI",
    "GJ13":"GANDHINAGAR","GJ14":"ANAND","GJ15":"PATAN","GJ16":"NAVSARI","GJ17":"VALSAD",
    "GJ18":"DAHOD","GJ19":"NARMADA","GJ20":"TAPI","GJ21":"ARAVALLI",

    # ==================== HARYANA ====================
    "HR01":"AMBALA","HR02":"YAMUNA NAGAR","HR03":"KARNAL","HR04":"PANIPAT",
    "HR05":"SONIPAT","HR06":"ROHTAK","HR07":"BHIWANI","HR08":"HISAR",
    "HR09":"SIRSA","HR10":"FATEHABAD","HR11":"GURGAON","HR12":"FARIDABAD",
    "HR13":"PALWAL","HR14":"MEWAT","HR15":"JIND","HR16":"KAITHAL",
    "HR26":"GURGAON (NEW)",

    # ==================== HIMACHAL PRADESH ====================
    "HP01":"SHIMLA","HP02":"DHARAMSHALA","HP03":"MANDI","HP04":"KULLU","HP05":"HAMIRPUR",

    # ==================== JHARKHAND ====================
    "JH01":"RANCHI","JH02":"HAZARIBAGH","JH03":"DHANBAD","JH04":"BOKARO","JH05":"GIRIDIH",

    # ==================== KARNATAKA ====================
    "KA01":"KORAMANGALA","KA02":"RAJAJINAGAR","KA03":"INDIRANAGAR",
    "KA04":"YASHWANTHPURA","KA05":"JAYANAGAR","KA06":"MALLESHWARAM",
    "KA41":"ELECTRONIC CITY",

    # ==================== KERALA ====================
    "KL01":"THIRUVANANTHAPURAM","KL02":"NEYYATTINKARA","KL03":"NEDUMANGAD",
    "KL04":"KATTAKKADA","KL05":"ATTINGAL","KL06":"KOLLAM","KL07":"KOTTARAKKARA",

    # ==================== MADHYA PRADESH ====================
    "MP01":"BHOPAL","MP02":"INDORE","MP03":"UJJAIN","MP04":"RATLAM","MP05":"MANDSAUR",

    # ==================== MAHARASHTRA ====================
    "MH01":"MUMBAI SOUTH","MH02":"MUMBAI WEST","MH03":"MUMBAI EAST",
    "MH04":"THANE","MH05":"KALYAN","MH12":"PUNE","MH14":"PIMPRI-CHINCHWAD",
    "MH16":"AHMEDNAGAR","MH31":"NAGPUR","MH32":"AMRAVATI","MH34":"GONDIA","MH47":"MUMBAI T.T.",

    # ==================== MANIPUR ====================
    "MN01":"IMPHAL WEST","MN02":"IMPHAL EAST",

    # ==================== MEGHALAYA ====================
    "ML01":"SHILLONG",

    # ==================== MIZORAM ====================
    "MZ01":"AIZAWL",

    # ==================== NAGALAND ====================
    "NL01":"DIMAPUR",

    # ==================== ODISHA ====================
    "OD01":"BALASORE","OD02":"BHADRAK","OD03":"CUTTACK","OD04":"DHENKANAL","OD05":"ANGUL",

    # ==================== PUNJAB ====================
    "PB02":"AMRITSAR","PB03":"BATHINDA","PB04":"FARIDKOT","PB05":"FEROZEPUR",
    "PB06":"GURDASPUR","PB07":"HOSHIARPUR","PB08":"JALANDHAR","PB09":"KAPURTHALA",
    "PB10":"LUDHIANA","PB11":"MANSA","PB12":"MOGA","PB13":"MUKTSAR",
    "PB14":"NAWANSHAHR","PB15":"PATIALA","PB16":"RUPNAGAR","PB17":"SANGRUR",
    "PB18":"TARN TARAN","PB29":"MOGA","PB47":"ZIRA",

    # ==================== RAJASTHAN ====================
    "RJ01":"JAIPUR","RJ02":"JODHPUR","RJ03":"KOTA","RJ04":"UDAIPUR","RJ05":"BANSWARA",
    "RJ14":"JAIPUR (NEW)",

    # ==================== SIKKIM ====================
    "SK01":"GANGTOK",

    # ==================== TAMIL NADU ====================
    "TN01":"CHENNAI","TN02":"CHENNAI NORTH","TN03":"CHENNAI SOUTH","TN04":"CHENNAI EAST",
    "TN09":"CHENNAI (SPECIAL)","TN14":"KANJEEPURAM","TN22":"VELLORE",

    # ==================== TELANGANA ====================
    "TS01":"ADILABAD","TS02":"KARIMNAGAR","TS03":"WARANGAL","TS04":"KHAMMAM",
    "TS05":"NIZAMABAD","TS07":"HYDERABAD",

    # ==================== UTTAR PRADESH ====================
    "UP01":"MEERUT","UP02":"MUZAFFARNAGAR","UP03":"SAHARANPUR","UP04":"SHAHJAHANPUR",
    "UP05":"BAREILLY","UP06":"PILIBHIT","UP07":"BADAUN","UP08":"BUDAUN","UP09":"RAMPUR",
    "UP13":"AGRA","UP14":"ALIGARH","UP15":"MATHURA","UP16":"ETAH","UP17":"MAINPURI",
    "UP20":"MORADABAD","UP21":"BIJNOR","UP22":"MORADABAD (NEW)","UP23":"BULANDSHAHR",
    "UP24":"GHAZIABAD","UP25":"GHAZIABAD (NEW)","UP26":"NOIDA","UP78":"KANPUR",
    "UP80":"LUCKNOW",

    # ==================== UTTARAKHAND ====================
    "UK01":"DEHRADUN","UK02":"HARIDWAR","UK03":"ROORKEE","UK04":"UDHAM SINGH NAGAR",

    # ==================== WEST BENGAL ====================
    "WB01":"KOLKATA","WB02":"HOWRAH","WB03":"NAIHATI","WB04":"BARRACKPORE","WB05":"RISHRA",
}

# Helper function: district → possible RTO matches
def get_rto_list(district):
    district = district.upper()
    matches = [(code, location) for code, location in RTO_MAP.items() if district in location.upper()]
    return matches

# -------------------------------- NAVIGATION MENU -----------------------------------
page = st.sidebar.radio("Choose Action:", [
    "📊 Data Summary",
    "📫 Pincode Lookup + ML Prediction",
    "🏡 City / Village Search",
    "🚗 Vehicle Plate RTO Detection"
])

# =====================================================================================
# PAGE 1 – DATA SUMMARY
# =====================================================================================
if page == "📊 Data Summary":
    st.header("📊 Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", len(df))
    c2.metric("Filtered State Records", len(state_df))
    c3.metric("Unique Districts", state_df["district"].nunique())
    c4.metric("Model Accuracy", f"{accuracy*100:.2f}%" if accuracy else "N/A")

    st.write("### Sample of Filtered Data")
    st.dataframe(state_df.head())

# =====================================================================================
# PAGE 2 – PINCODE LOOKUP + ML PREDICTION
# =====================================================================================
if page == "📫 Pincode Lookup + ML Prediction":

    st.header("📫 Pincode → SO First + BO Sort + RTO Suggestion")
    pincode = st.text_input("Enter 6-digit Pincode:")

    if st.button("Search / Predict"):
        if not pincode.isdigit() or len(pincode) != 6:
            st.error("❌ Invalid pincode")
        else:
            pin = int(pincode)
            result = df[df["pincode"] == pin]

            if not result.empty:
                # SO → BO Ordering
                order = {"HO":0, "H.O":0, "SO":1, "S.O":1, "BO":2, "B.O":2}
                result["sort"] = result["officetype"].apply(lambda x: order.get(x, 9))
                result = result.sort_values("sort")

                cols = ["officetype","officename","pincode","divisionname","district","statename"]
                st.success("🎯 Exact Match Found (SO First)")
                st.dataframe(result[cols].drop_duplicates())

                district_name = result.iloc[0]["district"]
            else:
                if model:
                    district_name = model.predict([[pin]])[0]
                    st.warning(f"🤖 ML Predicted District: **{district_name}**")
                else:
                    st.error("❌ Not enough data to predict")
                    st.stop()

            # RTO Matching
            rto_matches = get_rto_list(district_name)

            if len(rto_matches) == 0:
                st.warning("❓ No matching RTO found for this district.")
            elif len(rto_matches) == 1:
                code, loc = rto_matches[0]
                st.success(f"🚗 Assigned RTO: **{code} → {loc}**")
            else:
                st.warning("⚠ Multiple RTOs Found – Choose one:")
                choice = st.selectbox("Select Your RTO:", [f"{c} – {l}" for c,l in rto_matches])
                st.success(f"🚗 Final Selected RTO: **{choice}**")

# =====================================================================================
# PAGE 3 – CITY / VILLAGE SEARCH
# =====================================================================================
if page == "🏡 City / Village Search":
    st.header("🏡 Search By Locality Name")
    name = st.text_input("Enter locality:")

    if st.button("Find"):
        result = df[df["officename"].str.contains(name.upper().strip())]

        if result.empty:
            st.error("❌ No locality found.")
        else:
            order = {"HO":0, "H.O":0, "SO":1, "S.O":1, "BO":2, "B.O":2}
            result["sort"] = result["officetype"].apply(lambda x: order.get(x, 9))
            result = result.sort_values("sort")

            st.success(f"🎯 Found {len(result)} matches")
            st.dataframe(result[["officetype","officename","pincode","district","statename"]].drop_duplicates())

# =====================================================================================
# PAGE 4 – VEHICLE PLATE RTO DETECTION
# =====================================================================================
if page == "🚗 Vehicle Plate RTO Detection":
    st.header("🚗 Vehicle Registration RTO Finder")
    plate = st.text_input("Enter Plate Number (Example: GJ05AB1234)")

    if st.button("Check RTO"):
        plate = plate.upper().replace(" ", "")
        if len(plate) < 4:
            st.error("❌ Invalid input")
        else:
            key = plate[:4]
            if key in RTO_MAP:
                st.success(f"🚘 RTO Found: **{key} → {RTO_MAP[key]}**")
            else:
                st.warning("❓ RTO Code Not Found in Database")

# =====================================================================================
# END
# =====================================================================================

